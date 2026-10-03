"""Local acoustic checks for untimed lecture transcripts. Never starts training.

Run prepare_qwen_dataset.py first. All model output is cached under the ignored
dataset directory; restart resumes completed windows after checking provenance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import unicodedata
import wave

SCHEMA = 1


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_audio(path):
    import numpy as np
    with wave.open(str(path), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f'Expected mono 16 kHz PCM16 WAV: {path}')
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768


def fingerprint(lecture, args, model_path):
    payload = {key: lecture[key] for key in ('lecture_id', 'source_audio_sha256', 'source_transcript_sha256', 'text', 'split')}
    payload.update(schema=SCHEMA, window=args.window, model=str(model_path),
                   max_tokens=args.max_tokens, mlx_audio=importlib.metadata.version('mlx-audio'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def transcribe(args, lectures):
    os.environ['HF_HUB_OFFLINE'] = '1'
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_audio.stt import load
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        model_path = Path(snapshot_download(args.model, local_files_only=True,
                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken']))
    model = load(str(model_path.resolve()), strict=True)
    cache = args.dataset / 'asr'
    cache.mkdir(exist_ok=True)
    for lecture in lectures:
        path = cache / (lecture['lecture_id'] + '.json')
        stamp = fingerprint(lecture, args, model_path.resolve())
        saved = json.loads(path.read_text()) if path.exists() else {'fingerprint': stamp, 'model': str(model_path.resolve()), 'windows': []}
        if saved['fingerprint'] != stamp:
            raise ValueError(f'Stale ASR cache: {path}; use a new dataset output directory')
        audio = read_audio(lecture['audio'])
        width = round(args.window * 16000)
        count = math.ceil(len(audio) / width)
        for index in range(len(saved['windows']), count):
            start, end = index * width, min(len(audio), (index + 1) * width)
            began = time.monotonic()
            result = model.generate(audio[start:end], language='Korean', max_tokens=args.max_tokens,
                                    temperature=0.0, verbose=False)
            saved['windows'].append({'window_index': index, 'start': start / 16000, 'end': end / 16000,
                                     'text': result.text.strip(), 'truncated': result.generation_tokens >= args.max_tokens,
                                     'decode_seconds': round(time.monotonic() - began, 3)})
            write_json(path, saved)
            mx.clear_cache()
            if index % 10 == 0 or index == count - 1:
                print(f"ASR {lecture['lecture_id']} {index + 1}/{count}", flush=True)
        saved['complete'] = True
        write_json(path, saved)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_jsonl(path, rows):
    tmp = path.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def normalized(text):
    return ''.join(c.lower() for c in unicodedata.normalize('NFC', text) if c.isalnum())


def candidate_eligible(row):
    return not row['issues'] or (row.get('review_kind') == 'sparse_internal_timestamp_anomalies'
                                and set(row['issues']) == {'zero_duration_alignment_token'})


def validate_inventory(dataset, lectures):
    from prepare_qwen_dataset import prepare, clean_transcript
    report = json.loads((dataset / 'report.json').read_text())
    prepare(report['source_directory'], dataset)
    if not lectures:
        raise ValueError('Empty lecture inventory')
    source_splits = {}
    for lecture in lectures:
        if Path(lecture['transcript']).read_text(encoding='utf-8') != lecture['text'] + '\n':
            raise ValueError('Manifest text differs from normalized transcript')
        cleaned = clean_transcript(Path(lecture['source_transcript']).read_text(encoding='utf-8'))[0]
        if cleaned != lecture['text']:
            raise ValueError('Normalized transcript differs from source transcript cleaning')
        key = lecture['source_audio_sha256']
        if key in source_splits and source_splits[key] != lecture['split']:
            raise ValueError('Identical lecture audio crosses dataset partitions')
        source_splits[key] = lecture['split']


def align(args, lectures):
    """Create auditable candidates, withhold official train manifests until review."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import mlx.core as mx
    from mlx_audio.stt import load
    from qwen_alignment_checks import (map_windows_to_reference, prepare_crop_candidate,
                                       audit_alignment, boundary_review_eligible)
    if args.aligner is None or not args.aligner.is_dir():
        raise ValueError('--aligner must be a downloaded local Qwen3 forced-aligner snapshot')
    aligner_path = args.aligner.resolve()
    model = load(str(aligner_path), strict=True)
    cache = args.dataset / 'alignment'
    clips = args.dataset / 'clips'
    manifests = args.dataset / 'candidates'
    boundary_manifests = args.dataset / 'boundary_review'
    for folder in (cache, clips, manifests, boundary_manifests):
        folder.mkdir(exist_ok=True)
    all_rows = []
    for lecture in lectures:
        asr_path = args.dataset / 'asr' / (lecture['lecture_id'] + '.json')
        asr = json.loads(asr_path.read_text())
        if not asr.get('complete'):
            raise ValueError(f'Incomplete ASR cache: {asr_path}')
        if asr['fingerprint'] != fingerprint(lecture, args, Path(asr['model'])):
            raise ValueError(f'Stale ASR cache: {asr_path}')
        stamp = hashlib.sha256(json.dumps({'schema': SCHEMA, 'asr': sha256_file(asr_path),
                  'aligner': str(aligner_path), 'max_cer': args.max_cer,
                  'checker': sha256_file(Path(__file__).with_name('qwen_alignment_checks.py')),
                  'pipeline': sha256_file(__file__)}, sort_keys=True).encode()).hexdigest()
        path = cache / (lecture['lecture_id'] + '.json')
        saved = json.loads(path.read_text()) if path.exists() else {'fingerprint': stamp, 'rows': []}
        if saved['fingerprint'] != stamp:
            raise ValueError(f'Stale alignment cache: {path}; archive it before rerunning')
        audio = read_audio(lecture['audio'])
        mapped = map_windows_to_reference(lecture['text'], asr['windows'], max_cer=args.max_cer)
        for index in range(len(saved['rows']), len(mapped)):
            row = prepare_crop_candidate(lecture['text'], mapped[index], max_cer=args.max_cer)
            row.update(clip_id=f"{lecture['lecture_id']}_{index:05d}", lecture_id=lecture['lecture_id'],
                       split=lecture['split'], human_review='pending',
                       source_audio_sha256=lecture['source_audio_sha256'],
                       source_transcript_sha256=lecture['source_transcript_sha256'])
            row['issues'] = list(row['issues'])
            if not row['issues']:
                context_start = max(0, round((row['start'] - 1) * 16000))
                context_end = min(len(audio), round((row['end'] + 1) * 16000))
                context = audio[context_start:context_end]
                result = model.generate(context, text=row['reference_text'], language='Korean')
                items = [{'text': item.text, 'start_time': float(item.start_time),
                          'end_time': float(item.end_time)} for item in result.items]
                row['alignment_context_start'] = context_start / 16000
                row['alignment_context_duration_seconds'] = len(context) / 16000
                row['aligned_items'] = items
                row['issues'].extend(audit_alignment(items, row['reference_text'], len(context) / 16000))
                if boundary_review_eligible(items, row['reference_text'], len(context) / 16000,
                                            row['cer'], row['match_coverage']):
                    row['review_kind'] = 'sparse_internal_timestamp_anomalies'
                if items and candidate_eligible(row):
                    if items[0]['start_time'] < 0.08 or items[-1]['end_time'] > len(context) / 16000 - 0.08:
                        row['issues'].append('alignment_touches_context_boundary')
                    begin = max(context_start, context_start + round((items[0]['start_time'] - 0.03) * 16000))
                    end = min(context_end, context_start + round((items[-1]['end_time'] + 0.03) * 16000))
                    duration = (end - begin) / 16000
                    row.update(audio_start=begin / 16000, audio_end=end / 16000, duration_seconds=duration)
                    if not 1 <= duration <= 32:
                        row['issues'].append('clip_duration_out_of_bounds')
                    rate = len(normalized(row['reference_text'])) / max(duration, 0.001)
                    if not 1 <= rate <= 22:
                        row['issues'].append('implausible_text_rate')
                    samples = audio[begin:end]
                    row['audio_rms'] = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0
                    row['clipping_fraction'] = float(np.mean(np.abs(samples) >= 0.999)) if len(samples) else 0
                    if row['audio_rms'] < 0.0005:
                        row['issues'].append('near_silent_audio')
                    if row['clipping_fraction'] > 0.01:
                        row['issues'].append('excessive_clipping')
                    previous = next((r for r in reversed(saved['rows']) if candidate_eligible(r)), None)
                    if previous and begin / 16000 < previous['audio_end']:
                        row['issues'].append('overlapping_audio')
                    if previous and row['reference_start'] < previous['reference_end']:
                        row['issues'].append('overlapping_reference_span')
                    if candidate_eligible(row):
                        clip_path = clips / (row['clip_id'] + '.wav')
                        pcm = np.clip(np.rint(samples * 32768), -32768, 32767).astype('<i2').tobytes()
                        with wave.open(str(clip_path), 'wb') as stream:
                            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                            stream.writeframes(pcm)
                        row.update(audio=str(clip_path.resolve()), audio_sha256=sha256_file(clip_path),
                                   pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                                   text_sha256=hashlib.sha256(row['reference_text'].encode()).hexdigest())
                mx.clear_cache()
            row['automatic_status'] = ('pass' if not row['issues'] else
                                       'review_required' if candidate_eligible(row) else 'quarantined')
            saved['rows'].append(row)
            write_json(path, saved)
            if index % 10 == 0 or index == len(mapped) - 1:
                print(f"ALIGN {lecture['lecture_id']} {index + 1}/{len(mapped)}", flush=True)
        saved['complete'] = True
        write_json(path, saved)
        for row in saved['rows']:
            if candidate_eligible(row):
                clip_path = clips / (row['clip_id'] + '.wav')
                if row['audio'] != str(clip_path.resolve()) or not clip_path.is_file():
                    raise ValueError(f'Cached alignment clip missing or moved: {clip_path}')
                if sha256_file(clip_path) != row['audio_sha256']:
                    raise ValueError(f'Cached alignment clip changed: {clip_path}')
        all_rows.extend(saved['rows'])
    # Exact repeated audio is excluded across ALL partitions, conservatively.
    seen = {}
    for row in all_rows:
        if row['automatic_status'] not in ('pass', 'review_required'):
            continue
        key = row['pcm_sha256']
        if key in seen:
            row['issues'].append('duplicate_audio')
            row['automatic_status'] = 'quarantined'
            if row['split'] != seen[key]['split']:
                seen[key]['issues'].append('duplicate_audio_across_splits')
                seen[key]['automatic_status'] = 'quarantined'
        else:
            seen[key] = row
    save_jsonl(args.dataset / 'segments.jsonl', all_rows)
    save_jsonl(args.dataset / 'quarantine.jsonl', [r for r in all_rows if r['automatic_status'] == 'quarantined'])
    counts = {}
    for split in ('train', 'validation', 'test'):
        accepted = [r for r in all_rows if r['split'] == split and not r['issues']]
        boundary_review = [r for r in all_rows if r['split'] == split and r['automatic_status'] == 'review_required']
        save_jsonl(manifests / f'{split}.jsonl', [
            {'audio': r['audio'], 'text': 'language Korean<asr_text>' + r['reference_text']} for r in accepted])
        save_jsonl(boundary_manifests / f'{split}.jsonl', [
            {'audio': r['audio'], 'text': 'language Korean<asr_text>' + r['reference_text']} for r in boundary_review])
        counts[split] = {'lectures': [r['lecture_id'] for r in lectures if r['split'] == split],
                         'candidates': len(accepted), 'seconds': round(sum(r['duration_seconds'] for r in accepted), 3),
                         'boundary_review_candidates': len(boundary_review),
                         'boundary_review_seconds': round(sum(r['duration_seconds'] for r in boundary_review), 3)}
    review = [{'clip_id': r['clip_id'], 'audio_sha256': r['audio_sha256'], 'text_sha256': r['text_sha256'],
               'status': 'pending', 'notes': ''} for r in all_rows if candidate_eligible(r)]
    review_path = args.dataset / 'review_template.jsonl'
    save_jsonl(review_path, review)
    from collections import Counter
    report = {'alignment_complete': True,
              'segments_sha256': sha256_file(args.dataset / 'segments.jsonl'),
              'split_lock_sha256': sha256_file(args.dataset / 'split_lock.json'),
              'lectures_sha256': sha256_file(args.dataset / 'lectures.jsonl'),
              'training_ready': False, 'reason': 'Automated candidate checks complete; human transcript/boundary review is pending. No training launched.',
              'source_lectures': len(lectures), 'source_seconds': sum(r['duration_seconds'] for r in lectures),
              'windows': len(all_rows), 'quarantined': sum(r['automatic_status'] == 'quarantined' for r in all_rows),
              'boundary_review_required': sum(r['automatic_status'] == 'review_required' for r in all_rows),
              'issue_counts': dict(Counter(issue for r in all_rows for issue in r['issues'])),
              'coverage_warning_counts': dict(Counter(issue for r in all_rows for issue in r.get('coverage_warnings', []))),
              'splits': counts, 'max_normalized_cer': args.max_cer,
              'asr_model': args.model, 'aligner_snapshot': str(aligner_path),
              'speaker_disjoint': False,
              'limitations': ['Tiro labels are unverified machine transcripts, not human ground truth.',
                             'ASR agreement and forced timestamps cannot prove correctness.',
                             'Boundary-review clips retain sparse zero-duration internal tokens; they have NOT passed full alignment and require listening review.',
                             'Final evaluation needs reviewed labels; test is excluded from model selection.',
                             'Exact audio duplicates checked; semantic or near-duplicate recordings require review.']}
    write_json(args.dataset / 'alignment_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parents[1] / 'data/qwen3_asr')
    parser.add_argument('--stage', choices=['asr', 'align', 'all'], default='all')
    parser.add_argument('--model', default='mlx-community/Qwen3-ASR-1.7B-8bit')
    parser.add_argument('--aligner', type=Path)
    parser.add_argument('--window', type=float, default=30.0)
    parser.add_argument('--max-tokens', type=int, default=768)
    parser.add_argument('--max-cer', type=float, default=0.25)
    args = parser.parse_args()
    if not 5 <= args.window <= 30:
        parser.error('--window must be between 5 and 30 seconds')
    args.dataset = args.dataset.resolve()
    lectures = read_jsonl(args.dataset / 'lectures.jsonl')
    validate_inventory(args.dataset, lectures)
    if args.stage in ('asr', 'all'):
        transcribe(args, lectures)
    if args.stage in ('align', 'all'):
        align(args, lectures)


if __name__ == '__main__':
    main()
