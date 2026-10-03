"""Local Qwen3-ASR QLoRA on reviewed original Korean targets and explicit train/validation files.
Only q/v adapters are trained and saved; the base stays frozen. No test discovery,
model downloads, uploads, or dependency installation."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import wave

PREFIX = 'language Korean<asr_text>'
MAX_AUDIO_SECONDS = 90
VERSIONS = {'mlx': '0.32.3', 'mlx-audio': '0.5.7', 'transformers': '5.18.0'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def target_positions(prompt_length, target_length):
    require(type(prompt_length) is int and type(target_length) is int and prompt_length > 0 and target_length > 0,
            'Prompt and target token lengths must be positive integers')
    length = prompt_length + target_length - 1
    return length, prompt_length - 1, length


def read_pcm(path):
    with wave.open(str(path), 'rb') as handle:
        require((handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()) ==
                (16000, 1, 2, 'NONE'), f'Expected mono 16 kHz PCM16 WAV: {path}')
        frames = handle.getnframes()
        require(0 < frames <= MAX_AUDIO_SECONDS * 16000, f'Clip must be between 0 and {MAX_AUDIO_SECONDS} seconds: {path}')
        pcm = handle.readframes(frames)
        require(len(pcm) == frames * 2, f'Truncated WAV: {path}')
    return pcm


def read_manifest(path):
    path = Path(path).resolve()
    require(path.name.lower() not in ('test.jsonl', 'held_out_test.jsonl'), 'Test data must remain withheld')
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        require(source.get('split') != 'test', 'Test record in a training/validation manifest')
        text = source.get('text')
        require(isinstance(text, str) and text.startswith(PREFIX) and bool(text[len(PREFIX):].strip()),
                'Each label must contain the Korean prefix and a nonempty original transcript')
        transcript = text[len(PREFIX):]
        require('<|' not in transcript and '<asr_text>' not in transcript, 'Unexpected control token in transcript')
        audio = Path(source['audio']).expanduser()
        audio = (path.parent / audio).resolve() if not audio.is_absolute() else audio.resolve()
        require(audio.is_file(), f'Missing audio: {audio}')
        pcm = read_pcm(audio)
        match = re.match(r'(lecture_[a-f0-9]{12})(?:_|$)', audio.stem)
        rows.append({**source, 'audio': str(audio), 'text': text, 'transcript': transcript,
                     'file_sha256': sha256(audio), 'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                     'lecture_id': source.get('lecture_id') or (match.group(1) if match else None),
                     'duration_seconds': len(pcm) / 32000})
    require(bool(rows), f'Empty manifest: {path}')
    for field in ('audio', 'file_sha256', 'pcm_sha256'):
        require(len({row[field] for row in rows}) == len(rows), f'Duplicate {field} within manifest: {path}')
    return rows


def validate_splits(train, validation):
    require(bool(train) and bool(validation), 'Train and validation must both be nonempty')
    require(all(isinstance(row.get('lecture_id'), str) and row['lecture_id'] for row in train + validation),
            'Lecture identity is required in metadata or lecture_<12hex> clip filenames to enforce held-out lectures')
    for field in ('audio', 'file_sha256', 'pcm_sha256', 'lecture_id'):
        left = {row.get(field) for row in train} - {None, ''}
        right = {row.get(field) for row in validation} - {None, ''}
        require(not left.intersection(right), f'Train/validation {field} overlap')


def require_ignored_output(path):
    path = Path(path).resolve()
    require(not path.exists() or not any(path.iterdir()), 'Output must be new or empty; use a fresh run directory')
    existing = path
    while not existing.exists():
        existing = existing.parent
    repo = subprocess.run(['git', '-C', str(existing), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if repo.returncode == 0:
        result = subprocess.run(['git', '-C', repo.stdout.strip(), 'check-ignore', '--quiet', str(path / 'adapters.safetensors')])
        require(result.returncode == 0, 'Training output inside a Git repository must be Git-ignored')


def runtime():
    global mx, nn, optim, np, tree_flatten, create_attention_mask, module_checkpoint
    for package, expected in VERSIONS.items():
        require(importlib.metadata.version(package) == expected, f'This implementation requires {package}=={expected}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.nn.utils import checkpoint as module_checkpoint
    from mlx.utils import tree_flatten
    from mlx_audio.lm.models.base import create_attention_mask


def load_base(path):
    from mlx_audio.stt.utils import load_model
    path = Path(path).expanduser().resolve()
    require(path.is_dir(), '--model must name an existing local model snapshot')
    wrapper = load_model(path, strict=True)
    model = getattr(wrapper, '_model', wrapper)
    require(isinstance(model, nn.Module) and model.__class__.__name__ == 'Qwen3ASRModel', 'Expected Qwen3-ASR model')
    model.freeze()
    model.eval()
    return model


def inject_lora(model, rank, layers, alpha=16.0):
    require(0 < layers <= len(model.model.layers) and rank > 0, 'Invalid rank or decoder layer count')
    class LoRALinear(nn.Module):
        def __init__(self, base, input_size, output_size):
            super().__init__()
            require(isinstance(base, nn.QuantizedLinear) and base.bits == 8, 'Expected an 8-bit decoder projection')
            self.base = base
            self.base.freeze()
            self.lora_a = mx.random.uniform(low=-1 / math.sqrt(input_size), high=1 / math.sqrt(input_size), shape=(input_size, rank))
            self.lora_b = mx.zeros((rank, output_size))
        def __call__(self, value):
            original = self.base(value)
            delta = (value.astype(mx.float32) @ self.lora_a) @ self.lora_b
            return original + (delta * (alpha / rank)).astype(original.dtype)
    for layer in model.model.layers[-layers:]:
        attention = layer.self_attn
        for name, heads in [('q_proj', attention.num_heads), ('v_proj', attention.num_kv_heads)]:
            setattr(attention, name, LoRALinear(getattr(attention, name), attention.hidden_size, heads * attention.head_dim))
    parameters = tree_flatten(model.trainable_parameters())
    require(len(parameters) == 4 * layers and all(name.endswith(('.lora_a', '.lora_b')) for name, _ in parameters),
            'Unexpected trainable base parameters')
    mx.eval(model.trainable_parameters())
    return sum(value.size for _, value in parameters)


def save_adapter(model, path):
    mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))


def restore_adapter(model, path):
    weights = mx.load(str(path))
    expected = dict(tree_flatten(model.trainable_parameters()))
    require(set(weights) == set(expected), 'Adapter keys differ from the configured LoRA modules')
    require(all(weights[key].shape == expected[key].shape for key in weights), 'Adapter shapes differ')
    require(all(bool(mx.all(mx.isfinite(value)).item()) for value in weights.values()), 'Nonfinite adapter weights')
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())


def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in tree_flatten(model.parameters()):
        if name.endswith(('.lora_a', '.lora_b')):
            continue
        # NumPy does not expose MLX bfloat16. Float32 preserves each value exactly.
        array = np.array(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
        digest.update(name.encode()); digest.update(str((array.shape, array.dtype)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def waveform(row):
    require(sha256(row['audio']) == row['file_sha256'], 'Audio changed after manifest validation')
    return np.frombuffer(read_pcm(row['audio']), dtype='<i2').astype(np.float32) / 32768


def prepare_example(model, row, frozen_layers, max_target_tokens):
    features, mask, audio_tokens = model._preprocess_audio(waveform(row))
    encoded = model.get_audio_features(features, mask)
    require(encoded.shape[0] == audio_tokens, 'Audio feature/prompt token mismatch')
    prompt = model._build_prompt(audio_tokens, language='Korean')
    targets = model._tokenizer.encode(row['transcript'] + '<|im_end|>', add_special_tokens=False)
    require(0 < len(targets) <= max_target_tokens, 'Original target is empty or exceeds token budget; never silently truncate labels')
    require(targets[-1] == model._tokenizer.convert_tokens_to_ids('<|im_end|>'), 'Missing target EOS token')
    length, start, stop = target_positions(prompt.shape[1], len(targets))
    ids = mx.concatenate([prompt, mx.array([targets[:-1]], dtype=prompt.dtype)], axis=1)
    require(ids.shape[1] == length, 'Teacher-forcing input shift mismatch')
    hidden = model._build_inputs_embeds(ids, encoded)
    attention_mask = create_attention_mask(hidden, None)
    for layer in model.model.layers[:frozen_layers]:
        hidden = layer(hidden, mask=attention_mask, cache=None)
    hidden = mx.stop_gradient(hidden)
    mx.eval(hidden)
    mx.clear_cache()
    return {'hidden': hidden, 'targets': mx.array([targets]), 'start': start, 'stop': stop, 'tokens': len(targets)}


def loss(model, example, first_layer, checkpoint=False):
    hidden = example['hidden']
    mask = create_attention_mask(hidden, None)
    for layer in model.model.layers[first_layer:]:
        hidden = module_checkpoint(layer)(hidden, mask=mask, cache=None) if checkpoint else layer(hidden, mask=mask, cache=None)
    hidden = model.model.norm(hidden)[:, example['start']:example['stop'], :]
    require(hidden.shape[:2] == example['targets'].shape, 'Loss must include first target through EOS exactly once')
    total = mx.array(0.0)
    for index in range(0, example['tokens'], 128):
        targets = example['targets'][:, index:index + 128]
        def project_and_score(states, labels=targets):
            logits = model.lm_head(states) if model.lm_head is not None else model.model.embed_tokens.as_linear(states)
            return nn.losses.cross_entropy(logits.astype(mx.float32), mx.stop_gradient(labels), reduction='sum')
        score = mx.checkpoint(project_and_score) if checkpoint else project_and_score
        total = total + score(hidden[:, index:index + 128])
    return total / example['tokens']


def hydrate(example):
    return {**example, **mx.load(example['cache'])}


def validation_ce(model, examples, first_layer):
    model.eval()
    total = tokens = 0
    for example in examples:
        value = float(loss(model, hydrate(example), first_layer).item())
        require(math.isfinite(value), 'Nonfinite validation loss')
        total += value * example['tokens']; tokens += example['tokens']
        mx.clear_cache()
    return total / tokens


def generation_metrics(model, rows, max_tokens):
    from qwen_alignment_checks import normalize_text, _edit_distance
    model.eval()
    results, errors, characters = [], 0, 0
    for index, row in enumerate(rows):
        result = model.generate(waveform(row), language='Korean', temperature=0.0, max_tokens=max_tokens, verbose=False)
        reference, hypothesis = normalize_text(row['transcript']), normalize_text(result.text)
        distance = _edit_distance(reference, hypothesis)
        errors += distance; characters += len(reference)
        results.append({'audio': row['audio'], 'reference': row['transcript'], 'prediction': result.text,
                        'cer': distance / max(1, len(reference)), 'truncated': result.generation_tokens >= max_tokens})
        print(f'VALIDATION GENERATION {index + 1}/{len(rows)}', flush=True)
        mx.clear_cache()
    return {'normalized_cer': errors / max(1, characters), 'edits': errors, 'reference_characters': characters,
            'truncated': sum(row['truncated'] for row in results), 'rows': results}


def load_adapter_model(model_path, run_directory, adapter_name='selected.safetensors'):
    runtime()
    config = json.loads((Path(run_directory) / 'adapter_config.json').read_text())
    require(config['base_model'] == str(Path(model_path).resolve()), 'Adapter base snapshot differs')
    require(config['base_config_sha256'] == sha256(Path(model_path) / 'config.json'), 'Base config changed')
    model = load_base(model_path)
    inject_lora(model, config['rank'], config['layers'], config['alpha'])
    report = json.loads((Path(run_directory) / 'training_report.json').read_text())
    require(report.get('base_unchanged') is True and frozen_fingerprint(model) == report['base_frozen_sha256'],
            'Base weights differ from the verified training snapshot')
    if adapter_name == 'selected.safetensors':
        require(sha256(Path(run_directory) / adapter_name) == report['selected_adapter_sha256'], 'Selected adapter changed')
    restore_adapter(model, Path(run_directory) / adapter_name)
    model.eval()
    return model


def train(args):
    train_rows, val_rows = read_manifest(args.train_file), read_manifest(args.validation_file)
    validate_splits(train_rows, val_rows)
    output, base = args.output.resolve(), args.model.expanduser().resolve()
    require(output != base and base not in output.parents and output not in base.parents, 'Output overlaps the base snapshot')
    require_ignored_output(output)
    runtime(); mx.random.seed(args.seed)
    model = load_base(base)
    first = len(model.model.layers) - args.layers
    require(0 <= first < len(model.model.layers), 'Invalid adapted layer count')
    output.mkdir(parents=True, exist_ok=True)
    provenance = {split: [{'audio': row['audio'], 'audio_sha256': row['file_sha256'],
                          'pcm_sha256': row['pcm_sha256'], 'lecture_id': row['lecture_id'],
                          'duration_seconds': row['duration_seconds'],
                          'text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
                          'transcript_sha256': hashlib.sha256(row['transcript'].encode('utf-8')).hexdigest()}
                         for row in rows] for split, rows in [('train', train_rows), ('validation', val_rows)]}
    write_json(output / 'input_provenance.json', provenance)
    config = {'schema': 1, 'base_model': str(base), 'base_config_sha256': sha256(base / 'config.json'),
              'versions': VERSIONS, 'rank': args.rank, 'layers': args.layers, 'alpha': 16.0,
              'seed': args.seed, 'epochs': args.epochs, 'learning_rate': args.learning_rate, 'max_steps': args.max_steps,
              'train_file': str(args.train_file.resolve()), 'validation_file': str(args.validation_file.resolve()),
              'train_sha256': sha256(args.train_file), 'validation_sha256': sha256(args.validation_file),
              'input_provenance_sha256': sha256(output / 'input_provenance.json'),
              'target_policy': 'Known Korean prompt; original transcript plus im_end; loss includes first target through EOS.',
              'test_policy': 'No test manifest read; validation CE alone selects checkpoint, with baseline eligible.'}
    write_json(output / 'adapter_config.json', config)
    prepared = []
    cache = output / 'frozen_features'; cache.mkdir()
    for index, row in enumerate(train_rows + val_rows):
        item = prepare_example(model, row, first, args.max_target_tokens)
        path = cache / f'{index:05d}.safetensors'
        mx.save_safetensors(str(path), {key: item.pop(key) for key in ('hidden', 'targets')})
        prepared.append({**item, 'cache': str(path)})
        print(f'PRECOMPUTE {index + 1}/{len(train_rows) + len(val_rows)}', flush=True)
    train_data, val_data = prepared[:len(train_rows)], prepared[len(train_rows):]
    unadapted = float(loss(model, hydrate(val_data[0]), first).item())
    parameters = inject_lora(model, args.rank, args.layers)
    zero_adapter = float(loss(model, hydrate(val_data[0]), first).item())
    require(math.isfinite(unadapted) and abs(unadapted - zero_adapter) <= 1e-6, 'Zero-initialized adapters change baseline loss')
    frozen_before = frozen_fingerprint(model)
    save_adapter(model, output / 'baseline.safetensors')
    baseline_ce = validation_ce(model, val_data, first)
    baseline_generation = generation_metrics(model, val_rows, args.generation_max_tokens)
    write_json(output / 'baseline_generation.json', baseline_generation)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    grad_fn = nn.value_and_grad(model, lambda example: loss(model, example, first, checkpoint=True))
    history, steps = [], 0
    best_ce, best_path, best_trained_ce, best_epoch = baseline_ce, 'baseline.safetensors', math.inf, 0
    began = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_data))); random.Random(args.seed + epoch).shuffle(order)
        model.train(); step_losses = []
        for index in order:
            value, gradients = grad_fn(hydrate(train_data[index]))
            gradients, norm = optim.clip_grad_norm(gradients, 1.0)
            mx.eval(value, norm, gradients)
            scalar, gradient_norm = float(value.item()), float(norm.item())
            require(math.isfinite(scalar) and math.isfinite(gradient_norm), 'Nonfinite training loss or gradient')
            require(steps != 0 or gradient_norm > 0, 'Initial adapter gradient is zero')
            optimizer.update(model, gradients); mx.eval(model.trainable_parameters(), optimizer.state)
            require(all(bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(model.trainable_parameters())), 'Nonfinite adapter update')
            steps += 1; step_losses.append(scalar)
            history.append({'step': steps, 'epoch': epoch, 'train_loss': scalar, 'gradient_norm': gradient_norm})
            write_json(output / 'history.json', history)
            print(f'TRAIN step={steps} epoch={epoch} loss={scalar:.6f} grad={gradient_norm:.6f}', flush=True)
            mx.clear_cache()
            if args.max_steps and steps >= args.max_steps:
                break
        current_ce = validation_ce(model, val_data, first)
        save_adapter(model, output / 'last.safetensors')
        if current_ce < best_trained_ce:
            best_trained_ce = current_ce; save_adapter(model, output / 'best_trained.safetensors')
        if current_ce < best_ce:
            best_ce, best_path, best_epoch = current_ce, 'best_trained.safetensors', epoch
        history.append({'epoch': epoch, 'step': steps, 'validation_ce': current_ce, 'epoch_complete': len(step_losses) == len(train_data)})
        write_json(output / 'history.json', history)
        print(f'EPOCH {epoch} validation_ce={current_ce:.6f} selected_ce={best_ce:.6f}', flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    require(frozen_fingerprint(model) == frozen_before, 'Frozen base parameters changed during training')
    require(sha256(args.train_file) == config['train_sha256'] and sha256(args.validation_file) == config['validation_sha256'], 'Input manifests changed during training')
    require(all(sha256(row['audio']) == row['file_sha256'] for row in train_rows + val_rows),
            'Training or validation audio changed during training')
    require(sha256(output / 'input_provenance.json') == config['input_provenance_sha256'], 'Input provenance sidecar changed')
    shutil.copy2(output / best_path, output / 'selected.safetensors')
    restore_adapter(model, output / 'selected.safetensors')
    reloaded_ce = validation_ce(model, val_data, first)
    require(abs(reloaded_ce - best_ce) < 1e-6, 'Saved adapter reload does not reproduce selected validation loss')
    selected_generation = generation_metrics(model, val_rows, args.generation_max_tokens)
    write_json(output / 'selected_generation.json', selected_generation)
    report = {'training_completed': True, 'steps': steps, 'train_examples': len(train_rows), 'validation_examples': len(val_rows),
              'trainable_parameters': parameters, 'base_frozen_sha256': frozen_before, 'base_unchanged': True,
              'input_provenance_sha256': config['input_provenance_sha256'], 'input_audio_unchanged': True,
              'zero_adapter_loss_difference': abs(unadapted - zero_adapter), 'baseline_validation_ce': baseline_ce,
              'best_trained_validation_ce': best_trained_ce, 'selected_validation_ce': best_ce, 'selected_epoch': best_epoch,
              'selected_source': best_path, 'baseline_validation_cer': baseline_generation['normalized_cer'],
              'selected_validation_cer': selected_generation['normalized_cer'],
              'generation_cer_improved': selected_generation['normalized_cer'] < baseline_generation['normalized_cer'],
              'frozen_cache_bytes': sum(path.stat().st_size for path in cache.iterdir()), 'vocabulary_projection_chunk_tokens': 128,
              'seconds': time.monotonic() - began, 'peak_memory_bytes': mx.get_peak_memory(), 'test_evaluated': False,
              'selected_adapter_sha256': sha256(output / 'selected.safetensors')}
    write_json(output / 'training_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--train-file', type=Path)
    parser.add_argument('--validation-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-target-tokens', type=int, default=2048)
    parser.add_argument('--generation-max-tokens', type=int, default=2048)
    parser.add_argument('--infer-audio', type=Path, help='Load selected adapters from --output and transcribe this WAV')
    args = parser.parse_args()
    require(args.generation_max_tokens > 0, 'Invalid training limits: --generation-max-tokens must be positive')
    if args.infer_audio:
        from dataclasses import replace
        from qwen_decode import DEFAULT_POLICY, decode_audio
        model = load_adapter_model(args.model, args.output)
        audio = np.frombuffer(read_pcm(args.infer_audio.resolve()), dtype='<i2').astype(np.float32) / 32768
        decoded = decode_audio(model, audio, policy=replace(DEFAULT_POLICY, full_max_tokens=args.generation_max_tokens),
                               after_attempt=mx.clear_cache)
        print(json.dumps(decoded, ensure_ascii=False, indent=2))
    else:
        require(args.train_file and args.validation_file, '--train-file and --validation-file are required')
        require(args.epochs > 0 and args.rank > 0 and args.layers > 0 and args.max_steps >= 0 and args.max_target_tokens > 0
                and math.isfinite(args.learning_rate) and args.learning_rate > 0, 'Invalid training limits')
        train(args)


if __name__ == '__main__':
    main()
