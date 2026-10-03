"""Evaluate a completed, fixed Qwen LoRA run on held-out original transcripts.

This command never trains, selects a checkpoint, downloads models, or overwrites
an evaluation. Test predictions are reported for the baseline and the already
selected adapter, including outputs that reach the fixed generation limit.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

import train_qwen_lora as training
import qwen_decode
from qwen_alignment_checks import normalize_text, _edit_distance

MAX_TOKENS = 2048
require = training.require
sha256 = training.sha256


def read_test_manifest(path):
    """Read only an explicitly supplied test file, preserving original targets."""
    path = Path(path).expanduser().resolve()
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        require(isinstance(source, dict), 'Each test record must be an object')
        require(source.get('split', 'test') == 'test', 'Non-test record in held-out manifest')
        text = source.get('text')
        require(isinstance(text, str) and text.startswith(training.PREFIX), 'Expected original Korean target prefix')
        transcript = text[len(training.PREFIX):]
        require(bool(normalize_text(transcript)), 'Empty normalized test reference')
        require('<|' not in transcript and '<asr_text>' not in transcript, 'Unexpected control token in reference')
        require(isinstance(source.get('audio'), str), 'Missing test audio path')
        audio = Path(source['audio']).expanduser()
        audio = (path.parent / audio).resolve() if not audio.is_absolute() else audio.resolve()
        require(audio.is_file(), f'Missing test audio: {audio}')
        audio_hash = sha256(audio)
        pcm = training.read_pcm(audio)
        require(sha256(audio) == audio_hash, f'Test audio changed while reading: {audio}')
        match = re.match(r'(lecture_[a-f0-9]{12})(?:_|$)', audio.stem)
        lecture = source.get('lecture_id') or (match.group(1) if match else None)
        require(isinstance(lecture, str) and bool(lecture.strip()), 'Missing held-out lecture identity')
        if match:
            require(lecture == match.group(1), 'Test lecture metadata disagrees with clip filename')
        rows.append({'audio': str(audio), 'lecture_id': lecture, 'text': text, 'transcript': transcript,
                     'file_sha256': audio_hash, 'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                     'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                     'duration_seconds': len(pcm) / 32000})
    require(bool(rows), 'Empty held-out manifest')
    for field in ('audio', 'file_sha256', 'pcm_sha256'):
        require(len({row[field] for row in rows}) == len(rows), f'Duplicate test {field}')
    return rows


def read_completed_run(run, model):
    run, model = Path(run).expanduser().resolve(), Path(model).expanduser().resolve()
    names = ('adapter_config.json', 'training_report.json', 'input_provenance.json', 'selected.safetensors')
    files = {str(run / name): sha256(run / name) for name in names}
    config = json.loads((run / 'adapter_config.json').read_text(encoding='utf-8'))
    report = json.loads((run / 'training_report.json').read_text(encoding='utf-8'))
    require(report.get('training_completed') is True, 'Training has not completed')
    require(report.get('base_unchanged') is True and report.get('input_audio_unchanged') is True,
            'Training did not verify unchanged base weights and source audio')
    require(config.get('base_model') == str(model), 'Evaluation base differs from the training snapshot')
    require(config.get('base_config_sha256') == sha256(model / 'config.json'), 'Base config changed')
    provenance_hash = files[str(run / 'input_provenance.json')]
    require(config.get('input_provenance_sha256') == provenance_hash and
            report.get('input_provenance_sha256') == provenance_hash, 'Training provenance hash mismatch')
    require(report.get('selected_adapter_sha256') == files[str(run / 'selected.safetensors')],
            'Selected adapter changed after checkpoint selection')
    provenance = json.loads((run / 'input_provenance.json').read_text(encoding='utf-8'))
    require(isinstance(provenance, dict), 'Invalid training provenance')
    for split, count in [('train', 'train_examples'), ('validation', 'validation_examples')]:
        records = provenance.get(split)
        require(isinstance(records, list) and bool(records) and len(records) == report.get(count),
                f'Missing or inconsistent {split} provenance')
        for row in records:
            require(isinstance(row, dict), 'Invalid provenance record')
            require(isinstance(row.get('lecture_id'), str) and bool(row['lecture_id'].strip()),
                    'Missing provenance lecture identity')
            require(isinstance(row.get('audio'), str) and bool(row['audio']), 'Missing provenance audio path')
            for field in ('audio_sha256', 'pcm_sha256'):
                require(isinstance(row.get(field), str) and re.fullmatch(r'[a-f0-9]{64}', row[field]),
                        f'Invalid provenance {field}')
    return config, report, provenance, files


def validate_held_out(rows, provenance):
    """Check source identity and decoded PCM against both optimization partitions."""
    for split in ('train', 'validation'):
        for field, source_field in [('audio', 'audio'), ('lecture_id', 'lecture_id'),
                                    ('file_sha256', 'audio_sha256'), ('pcm_sha256', 'pcm_sha256')]:
            left = {row[field] for row in rows}
            right = {row[source_field] for row in provenance[split]}
            if field == 'audio':
                left = {str(Path(value).expanduser().resolve()) for value in left}
                right = {str(Path(value).expanduser().resolve()) for value in right}
            require(not left.intersection(right), f'Test/{split} {field} overlap')


def assert_files_unchanged(files):
    for path, digest in files.items():
        require(Path(path).is_file() and sha256(path) == digest, f'Evaluation input changed: {path}')


def score_prediction(row, prediction, generation_tokens, *, unresolved=None):
    require(isinstance(prediction, str), 'Model prediction must be text')
    require(type(generation_tokens) is int and generation_tokens >= 0, 'Invalid generation token count')
    reference, hypothesis = normalize_text(row['transcript']), normalize_text(prediction)
    require(bool(reference), 'Empty normalized test reference')
    edits = _edit_distance(reference, hypothesis)
    cer = edits / len(reference)
    require(math.isfinite(cer), 'Nonfinite CER')
    return {**row, 'prediction': prediction, 'normalized_reference': reference,
            'normalized_prediction': hypothesis, 'edits': edits, 'reference_characters': len(reference),
            'normalized_cer': cer, 'generation_tokens': generation_tokens,
            'truncated': generation_tokens >= MAX_TOKENS if unresolved is None else bool(unresolved)}


def score_decoded(row, decoded):
    """Score every final output, retaining raw failure evidence and unresolved leaves."""
    require(decoded['policy'] == qwen_decode.policy_proof(), 'Unexpected evaluation decoding policy')
    final = score_prediction(row, decoded['text'], decoded['generation_tokens'], unresolved=decoded['unresolved'])
    initial = score_prediction(row, decoded['initial_text'], decoded['initial_generation_tokens'])
    return {**final, 'initial_prediction': decoded['initial_text'], 'initial_edits': initial['edits'],
            'initial_normalized_cer': initial['normalized_cer'],
            'initial_token_limit': 'generation_token_limit' in decoded['initial_problem'],
            'initial_repetition': 'consecutive_repetition' in decoded['initial_problem'],
            'unresolved': decoded['unresolved'], 'retry_attempts': decoded['retries'],
            'fallback_used': decoded['fallback_used'], 'decode_evidence': decoded}


def generate_predictions(model, rows, label):
    model.eval()
    predictions = []
    for index, row in enumerate(rows):
        result = qwen_decode.decode_audio(model, training.waveform(row), after_attempt=training.mx.clear_cache)
        predictions.append(score_decoded(row, result))
        print(f'{label.upper()} TEST {index + 1}/{len(rows)}', flush=True)
        training.mx.clear_cache()
    edits = sum(row['edits'] for row in predictions)
    characters = sum(row['reference_characters'] for row in predictions)
    cer = edits / characters
    require(math.isfinite(cer), 'Nonfinite aggregate CER')
    return {'normalized_cer': cer, 'edits': edits, 'reference_characters': characters,
            'initial_normalized_cer': sum(row['initial_edits'] for row in predictions) / characters,
            'initial_token_limit': sum(row['initial_token_limit'] for row in predictions),
            'initial_repetition': sum(row['initial_repetition'] for row in predictions),
            'retry_attempts': sum(row['retry_attempts'] for row in predictions),
            'fallback_predictions': sum(row['fallback_used'] for row in predictions),
            'unresolved': sum(row['unresolved'] for row in predictions),
            'truncated': sum(row['truncated'] for row in predictions),
            'generation_policy': qwen_decode.policy_proof(), 'rows': predictions}


def require_fresh_output(output, run, model):
    output, run, model = (Path(value).expanduser().resolve() for value in (output, run, model))
    require(not output.exists(), 'Evaluation output already exists; choose a fresh directory')
    for protected in (run, model):
        require(output != protected and output not in protected.parents and protected not in output.parents,
                'Evaluation output must be separate from checkpoint and base model directories')
    training.require_ignored_output(output)
    return output


def evaluate(args):
    run, model, test_file = (Path(value).expanduser().resolve() for value in (args.run, args.model, args.test_file))
    output = require_fresh_output(args.output, run, model)
    config, report, provenance, immutable = read_completed_run(run, model)
    immutable[str(test_file)] = sha256(test_file)
    rows = read_test_manifest(test_file)
    validate_held_out(rows, provenance)
    immutable.update({row['audio']: row['file_sha256'] for row in rows})
    # Pin every local snapshot file before model loading; no network or remote model IDs.
    immutable.update({str(path): sha256(path) for path in sorted(model.iterdir()) if path.is_file()})
    immutable.update({str(Path(path).resolve()): sha256(path) for path in
                      (__file__, qwen_decode.__file__, training.__file__)})
    assert_files_unchanged(immutable)
    output.mkdir(parents=True, exist_ok=False)
    training.write_json(output / 'evaluation_inputs.json', {
        'schema': 1, 'run': str(run), 'test_file': str(test_file), 'files_sha256': immutable,
        'examples': rows, 'generation': qwen_decode.policy_proof(),
        'policy': 'Fixed selected checkpoint; no training, checkpoint selection, or threshold tuning.'})
    began = time.monotonic()
    # Verify the trained base and selected adapter before exposing any test results.
    selected = training.load_adapter_model(model, run)
    del selected
    gc.collect()
    training.mx.clear_cache()
    baseline_model = training.load_base(model)
    baseline = generate_predictions(baseline_model, rows, 'baseline')
    assert_files_unchanged(immutable)
    training.write_json(output / 'baseline_predictions.json', baseline)
    del baseline_model
    gc.collect()
    training.mx.clear_cache()
    selected_model = training.load_adapter_model(model, run)
    selected = generate_predictions(selected_model, rows, 'selected')
    assert_files_unchanged(immutable)
    training.write_json(output / 'selected_predictions.json', selected)
    result = {'schema': 1, 'evaluation_completed': True, 'partition': 'held_out_test',
              'test_examples': len(rows), 'test_lectures': sorted({row['lecture_id'] for row in rows}),
              'duration_seconds': sum(row['duration_seconds'] for row in rows),
              'baseline_normalized_cer': baseline['normalized_cer'], 'selected_normalized_cer': selected['normalized_cer'],
              'selected_minus_baseline_cer': selected['normalized_cer'] - baseline['normalized_cer'],
              'baseline_edits': baseline['edits'], 'selected_edits': selected['edits'],
              'reference_characters': baseline['reference_characters'],
              'baseline_truncated': baseline['truncated'], 'selected_truncated': selected['truncated'],
              'baseline_raw_initial_cer': baseline['initial_normalized_cer'], 'selected_raw_initial_cer': selected['initial_normalized_cer'],
              'baseline_initial_token_limit': baseline['initial_token_limit'], 'selected_initial_token_limit': selected['initial_token_limit'],
              'baseline_initial_repetition': baseline['initial_repetition'], 'selected_initial_repetition': selected['initial_repetition'],
              'baseline_retry_attempts': baseline['retry_attempts'], 'selected_retry_attempts': selected['retry_attempts'],
              'baseline_fallback_predictions': baseline['fallback_predictions'], 'selected_fallback_predictions': selected['fallback_predictions'],
              'baseline_unresolved': baseline['unresolved'], 'selected_unresolved': selected['unresolved'],
              'generation_policy': qwen_decode.policy_proof(), 'unresolved_outputs_included': True,
              'generation_max_tokens': MAX_TOKENS, 'truncated_outputs_included': True,
              'selected_adapter_sha256': report['selected_adapter_sha256'],
              'training_report_sha256': immutable[str(run / 'training_report.json')],
              'input_provenance_sha256': config['input_provenance_sha256'],
              'test_manifest_sha256': immutable[str(test_file)], 'inputs_unchanged': True,
              'checkpoint_selection_performed': False, 'training_performed': False,
              'seconds': time.monotonic() - began,
              'metric': 'Corpus character error rate after NFKC, case folding, and removal of non-alphanumeric characters.',
              'limitation': 'Fixed held-out labels only. Bounded retry changes inference behavior, not the selected weights. '
                            'Raw initial predictions and final unresolved outputs remain visible; audio intervals are not word alignments.'}
    require(all(math.isfinite(result[key]) for key in ('baseline_normalized_cer', 'selected_normalized_cer',
                                                      'selected_minus_baseline_cer', 'seconds')), 'Nonfinite evaluation metric')
    training.write_json(output / 'evaluation_report.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--test-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
