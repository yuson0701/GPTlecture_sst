"""Export every Qwen transcription and compare with the original lecture scripts.

Reuses the complete independent ASR pass; retries token-limited windows locally
in shorter intervals. Never uses reference words as model prompts, filters by
alignment eligibility, modifies the frozen dataset, or launches training.
"""
import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from align_qwen_dataset import (fingerprint, read_audio, read_jsonl, sha256_file,
                                validate_inventory, write_json)
from qwen_alignment_checks import character_error_rate, normalize_text

ROOT = Path(__file__).resolve().parents[1]


def timestamp(seconds):
    total = int(seconds)
    return f'{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}'


def diff_texts(original, qwen):
    """Lossless word/punctuation display diff; CER is calculated separately."""
    token_pattern = r'\s+|\w+|[^\w\s]'
    left, right = re.findall(token_pattern, original), re.findall(token_pattern, qwen)
    return [{'tag': tag, 'original': ''.join(left[a:b]), 'qwen': ''.join(right[c:d])}
            for tag, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()]


def validate_windows(windows, duration):
    cursor = 0.0
    for row in windows:
        if abs(row['start'] - cursor) > 1 / 16000 or not row['start'] < row['end']:
            raise ValueError('ASR windows must cover the lecture in chronological order without gaps')
        if not isinstance(row['text'], str):
            raise ValueError('Invalid ASR text')
        cursor = row['end']
    if abs(cursor - duration) > 1 / 16000:
        raise ValueError('ASR duration differs from source audio')


def export(args):
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError('Use a separate output directory, outside the frozen dataset')
    lectures = read_jsonl(dataset / 'lectures.jsonl')
    validate_inventory(dataset, lectures)
    caches = {}
    for lecture in lectures:
        path = dataset / 'asr' / (lecture['lecture_id'] + '.json')
        cache = json.loads(path.read_text())
        if not cache.get('complete'):
            raise ValueError(f'Incomplete ASR pass: {path}')
        options = SimpleNamespace(window=30.0, max_tokens=768)
        if cache['fingerprint'] != fingerprint(lecture, options, Path(cache['model'])):
            raise ValueError(f'Stale ASR cache: {path}')
        validate_windows(cache['windows'], lecture['duration_seconds'])
        caches[lecture['lecture_id']] = cache
    lock = {'schema_version': 1, 'retry_policy': '10s/384 tokens; halve capped subwindows to 2.5s',
            'lectures_sha256': sha256_file(dataset / 'lectures.jsonl'),
            'asr_sha256': {lid: sha256_file(dataset / 'asr' / (lid + '.json')) for lid in caches}}
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'source_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError('Output provenance differs; choose a new output directory')
    elif any(output.iterdir()):
        raise ValueError('Refusing to overwrite an unknown nonempty output directory')
    else:
        write_json(lock_path, lock)
    retry_path = output / 'retry_cache.json'
    retries = json.loads(retry_path.read_text()) if retry_path.exists() else {}
    model, mx = None, None

    def retry(lecture, source_window):
        nonlocal model, mx
        cache_key = f"{lecture['lecture_id']}_{source_window['window_index']:05d}"
        if cache_key in retries:
            validate_windows([
                {**s, 'start': s['start'] - source_window['start'], 'end': s['end'] - source_window['start']}
                for s in retries[cache_key]], source_window['end'] - source_window['start'])
            return retries[cache_key]
        if model is None:
            os.environ['HF_HUB_OFFLINE'] = '1'
            import mlx.core as mlx_core
            from mlx_audio.stt import load
            mx = mlx_core
            model = load(caches[lecture['lecture_id']]['model'], strict=True)
        audio = read_audio(lecture['audio'])

        def decode(begin, end):
            result = model.generate(audio[round(begin * 16000):round(end * 16000)], language='Korean',
                                    max_tokens=384, temperature=0.0, verbose=False)
            capped = result.generation_tokens >= 384
            mx.clear_cache()
            if capped and end - begin > 2.5 + 1e-6:
                middle = round((begin + end) * 8000) / 16000
                return decode(begin, middle) + decode(middle, end)
            return [{'start': begin, 'end': end, 'text': result.text.strip(), 'truncated': capped,
                     'retried': True, 'parent_window_index': source_window['window_index'],
                     'generation_tokens': result.generation_tokens}]

        result = []
        start = source_window['start']
        while start < source_window['end'] - 1e-6:
            end = min(start + 10, source_window['end'])
            result.extend(decode(start, end))
            start = end
        retries[cache_key] = result
        write_json(retry_path, retries)
        print(f"Retried {lecture['lecture_id']} {timestamp(source_window['start'])}: "
              f"{len(result)} shorter windows; {sum(s['truncated'] for s in result)} still capped", flush=True)
        return result

    transcript_dir = output / 'transcripts'
    transcript_dir.mkdir(exist_ok=True)
    compiled = []
    for lecture in lectures:
        lid = lecture['lecture_id']
        segments = []
        for window in caches[lid]['windows']:
            if window['truncated']:
                segments.extend(retry(lecture, window))
            else:
                segments.append({key: window[key] for key in ('start', 'end', 'text', 'truncated')}
                                | {'retried': False, 'parent_window_index': window['window_index']})
        validate_windows(segments, lecture['duration_seconds'])
        qwen_text = '\n\n'.join(s['text'] for s in segments if s['text'])
        original = lecture['text']
        files = {'qwen_txt': f'transcripts/{lecture["title"]} Qwen.txt',
                 'original_txt': f'transcripts/{lecture["title"]} Original.txt',
                 'qwen_timed_txt': f'transcripts/{lecture["title"]} Qwen timed.txt'}
        for filename in files.values():
            if Path(filename).parent != Path('transcripts'):
                raise ValueError('Unsafe transcript title')
        (output / files['qwen_txt']).write_text(qwen_text + '\n', encoding='utf-8')
        (output / files['original_txt']).write_text(original + '\n', encoding='utf-8')
        timed = ['Qwen3-ASR transcription. Times identify input windows, not aligned word boundaries.', '']
        for s in segments:
            label = f"[{timestamp(s['start'])}–{timestamp(s['end'])}]"
            if s['truncated']:
                label += ' [OUTPUT LIMIT — text may be incomplete or repetitive]'
            timed.extend([label, s['text'] or '[No text returned by Qwen]', ''])
        (output / files['qwen_timed_txt']).write_text('\n'.join(timed), encoding='utf-8')
        warnings = ['Input windows can split words; timestamps are not forced alignment.']
        capped_count = sum(s['truncated'] for s in segments)
        if capped_count:
            warnings.append(f'{capped_count} shorter windows still hit the output limit; these are retained and marked.')
        empty_count = sum(not s['text'] for s in segments)
        if empty_count:
            warnings.append(f'{empty_count} windows returned no text. They remain visible in the timed transcript.')
        compiled.append({'lecture_id': lid, 'title': lecture['title'], 'duration_seconds': lecture['duration_seconds'],
                         'source_split': lecture['split'], 'cer': character_error_rate(original, qwen_text),
                         'reference_characters': len(normalize_text(original)), 'qwen_characters': len(normalize_text(qwen_text)),
                         'reference_text': original, 'qwen_text': qwen_text, 'segments': segments,
                         'diff': diff_texts(original, qwen_text), 'files': files,
                         'audio': os.path.relpath(lecture['audio'], output), 'warnings': warnings,
                         'retried_parent_windows': sum(w['truncated'] for w in caches[lid]['windows']),
                         'unresolved_truncated_windows': capped_count})
        print(f"Exported {lid}: {100 * compiled[-1]['cer']:.1f}% character difference", flush=True)
    report = {'model': 'Qwen3-ASR-1.7B (local MLX 8-bit)',
              'method_note': 'Character difference is Levenshtein edits / original character count after NFKC, case folding, '
                             'and removal of spaces/punctuation. This is disagreement, not accuracy: neither transcript '
                             'has been established as ground truth. Highlighted text also shows spelling, punctuation, '
                             'and spacing differences. Original export metadata, speaker labels, and non-speech '
                             'annotations were removed for comparison; spoken wording was preserved.',
              'source_note': 'Reuses every completed Qwen ASR window, regardless of prior alignment rejection. '
                             'Token-capped windows are retranscribed in shorter chunks without reference text prompts.',
              'lectures': compiled,
              'weighted_character_difference': sum(r['cer'] * r['reference_characters'] for r in compiled)
                                               / sum(r['reference_characters'] for r in compiled)}
    examples_path = output / 'examples.json'
    if examples_path.exists():
        examples = json.loads(examples_path.read_text(encoding='utf-8'))
        by_id = {lecture['lecture_id']: lecture for lecture in compiled}
        for example in examples:
            lecture = by_id[example['lecture_id']]
            segment = next(s for s in lecture['segments'] if s['start'] == example['window_start'])
            if example['original'] not in lecture['reference_text'] or example['qwen'] not in segment['text']:
                raise ValueError('A comparison example is stale or does not quote the current transcripts')
        report['examples'] = examples
    write_json(output / 'comparison.json', report)
    print(json.dumps({'output': str(output), 'lectures': len(compiled),
                      'weighted_character_difference': report['weighted_character_difference'],
                      'unresolved_truncated_windows': sum(r['unresolved_truncated_windows'] for r in compiled)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/qwen3_asr')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/qwen_transcripts')
    export(parser.parse_args())


if __name__ == '__main__':
    main()
