"""Pair original transcripts with their source paragraph timestamps, locally.

No ASR predictions, interpolated timings, network, or training are used. Full
paragraphs without usable source intervals are quarantined rather than split.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave

from prepare_qwen_dataset import canonical_title, clean_transcript, sha256
from fetch_tiro_timestamps import extract_page

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')
OWNED = ('clips', 'segments.jsonl', 'quarantine.jsonl', 'train.jsonl', 'validation.jsonl', 'test.jsonl', 'report.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def text_normalized(text):
    # Only Unicode composition and whitespace; preserve every spoken character.
    return ' '.join(unicodedata.normalize('NFC', text).split())


def unique(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, f'Duplicate {key}: {row[key]}')
        result[row[key]] = row
    return result


def cleaned_piece(value):
    try:
        text, _, events, _, unknown = clean_transcript(value)
        return text_normalized(text), events, unknown
    except ValueError as error:
        if str(error) == 'Transcript has no speech after metadata and event removal':
            return '', [{'event': event} for event in re.findall(r'\[[^\]\n]+\]', value)], []
        raise


def paragraph_text(paragraph):
    """Respect source user overrides; never use summaries or generated Qwen text."""
    for field in ('userDiarizedTranscript', 'userTranscript', 'diarizedTranscript', 'refinedTranscript', 'rawTranscript'):
        value = paragraph.get(field)
        if value is None or value == '' or value == {}:
            continue
        if field.endswith('DiarizedTranscript') or field == 'diarizedTranscript':
            require(isinstance(value, dict) and isinstance(value.get('segments'), list), f'Malformed {field}')
            segments = value['segments']
            if not segments:
                continue
            require(all(isinstance(s.get('content'), str) for s in segments), f'Malformed {field} content')
            value = '\n'.join(s['content'] for s in segments)
        require(isinstance(value, str), f'Malformed transcript field: {field}')
        text, events, unknown = cleaned_piece(value)
        return text, field, events, unknown
    return '', None, [], []


def ensure_private_output(output):
    existing = output
    while not existing.exists():
        existing = existing.parent
    result = subprocess.run(['git', '-C', str(existing), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if result.returncode == 0:
        check = subprocess.run(['git', '-C', result.stdout.strip(), 'check-ignore', '--quiet', str(output / 'train.jsonl')])
        require(check.returncode == 0, 'Dataset output inside Git must be ignored')


def validate_sources(base_dataset, output):
    lock_path, manifest_path = base_dataset / 'split_lock.json', base_dataset / 'lectures.jsonl'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    locked = unique(lock['lectures'], 'lecture_id')
    lectures = unique(read_jsonl(manifest_path), 'lecture_id')
    require(set(locked) == set(lectures) and bool(lectures), 'Lecture inventory differs from frozen split lock')
    source_manifest = output / 'sources.json'
    sources = unique(json.loads(source_manifest.read_text(encoding='utf-8'))['sources'], 'lecture_id')
    require(set(sources) == set(lectures), 'Timestamp source inventory differs from frozen lectures')
    groups, seen_audio, raw_sources, provenance = {}, {}, {}, {}
    for lid, lecture in lectures.items():
        require(re.fullmatch(r'lecture_[a-f0-9]{12}', lid) is not None, 'Invalid lecture ID')
        require(all(lecture.get(k) == v for k, v in locked[lid].items()), f'Split/source provenance changed: {lid}')
        require(lecture['split'] in SPLITS, 'Unknown frozen split')
        group = re.sub(r'^\d+_', '', canonical_title(lecture['title']))
        require(group not in groups or groups[group] == lecture['split'], 'Repeated lecturer title crosses splits')
        groups[group] = lecture['split']
        for kind in ('audio', 'transcript'):
            require(sha256(lecture['source_' + kind]) == lecture['source_' + kind + '_sha256'], f'Changed source {kind}: {lid}')
        for key in ('source_audio_sha256', 'audio_sha256'):
            marker = (key, lecture[key])
            require(marker not in seen_audio or seen_audio[marker] == lecture['split'], 'Duplicate recording crosses splits')
            seen_audio[marker] = lecture['split']
        expected_audio = base_dataset / 'lectures' / (lid + '.wav')
        expected_text = base_dataset / 'lectures' / (lid + '.txt')
        require(lecture['audio'] == str(expected_audio) and lecture['transcript'] == str(expected_text), 'Unexpected normalized artifact path')
        require(not expected_audio.is_symlink() and not expected_text.is_symlink(), 'Symlinked normalized artifact')
        require(sha256(expected_audio) == lecture['audio_sha256'], f'Normalized audio changed: {lid}')
        require(sha256(expected_text) == lecture['transcript_sha256'], f'Normalized transcript changed: {lid}')
        source_text, paragraphs, *_ = clean_transcript(Path(lecture['source_transcript']).read_text(encoding='utf-8'))
        require(source_text == lecture['text'] and paragraphs == lecture['paragraphs'] and
                expected_text.read_text(encoding='utf-8') == source_text + '\n', f'Original labels changed: {lid}')
        path = output / 'raw' / (lid + '.json')
        require(not path.is_symlink(), 'Symlinked timestamp source')
        raw = json.loads(path.read_text(encoding='utf-8'))
        require(raw['lecture_id'] == lid and raw['url'] == sources[lid]['url'], f'Timestamp source identity mismatch: {lid}')
        require(re.fullmatch(r'[a-f0-9]{64}', raw.get('page_sha256', '')) is not None, 'Invalid page provenance hash')
        page = output / 'raw' / (lid + '.html')
        require(not page.is_symlink() and sha256(page) == raw['page_sha256'], f'Cached source page hash mismatch: {lid}')
        require(extract_page(page.read_bytes(), sources[lid]) == raw,
                f'Timestamp JSON differs from cached source page: {lid}')
        require(canonical_title(raw['source']['title']) == canonical_title(lecture['title']), f'Share title differs from original: {lid}')
        require(isinstance(raw['source'].get('paragraphs'), list), 'Missing source paragraphs')
        raw_sources[lid] = raw
        provenance[lid] = {'raw_json_sha256': sha256(path), 'page_sha256': raw['page_sha256'], 'url': raw['url']}
    return lectures, raw_sources, {'split_lock_sha256': sha256(lock_path), 'lectures_sha256': sha256(manifest_path),
                                   'sources_sha256': sha256(source_manifest), 'raw_sources': provenance}


def flag_overlaps(rows):
    valid = [r for r in rows if type(r['source_start_ms']) is int and type(r['source_end_ms']) is int
             and 0 <= r['source_start_ms'] < r['source_end_ms']]
    for before, after in zip(valid, valid[1:]):
        if after['source_start_ms'] < before['source_start_ms']:
            before['issues'].append('unordered_source_timestamps')
            after['issues'].append('unordered_source_timestamps')
    active = []
    for row in sorted(valid, key=lambda r: r['source_start_ms']):
        active = [r for r in active if r['source_end_ms'] > row['source_start_ms']]
        for other in active:
            other['issues'].append('overlapping_source_timestamps')
            row['issues'].append('overlapping_source_timestamps')
        active.append(row)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in values), encoding='utf-8')


def prepare(base_dataset, output, max_duration=90.0):
    import numpy as np
    base_dataset, output = Path(base_dataset).resolve(), Path(output).resolve()
    require(type(max_duration) in (int, float) and math.isfinite(max_duration) and 0 < max_duration <= 120,
            'max-duration must be positive and at most 120 seconds')
    require(base_dataset != output and base_dataset not in output.parents and output not in base_dataset.parents,
            'Timestamp output and base dataset must not overlap')
    ensure_private_output(output)
    lectures, sources, provenance = validate_sources(base_dataset, output)
    stamp = {**provenance, 'max_duration_seconds': max_duration, 'schema_version': 1}
    existing = set(p.name for p in output.iterdir())
    if 'report.json' in existing:
        report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        require(report['input_provenance'] == stamp, 'Prepared timestamp dataset is stale; use a new output directory')
        for name, digest in report['artifact_sha256'].items():
            require(sha256(output / name) == digest, f'Existing timestamp artifact changed: {name}')
        for row in read_jsonl(output / 'segments.jsonl'):
            if row['automatic_status'] == 'pass':
                require(row['audio'] == str(output / 'clips' / (row['clip_id'] + '.wav')) and
                        sha256(row['audio']) == row['audio_sha256'], 'Existing timestamp clip changed')
        return report
    require(not (existing - {'sources.json', 'raw'}), 'Unknown output artifacts; refusing overwrite')
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        (stage / 'clips').mkdir()
        rows, lecture_reports = [], []
        for lid, lecture in lectures.items():
            raw = sources[lid]
            paragraphs = raw['source']['paragraphs']
            require(len({p.get('uuid') for p in paragraphs}) == len(paragraphs) and all(p.get('uuid') for p in paragraphs),
                    f'Missing or duplicate paragraph UUID: {lid}')
            texts = [paragraph_text(paragraph) for paragraph in paragraphs]
            full_text = text_normalized(' '.join(item[0] for item in texts))
            require(full_text == text_normalized(lecture['text']), f'Timestamp transcript does not match entire original transcript: {lid}')
            lecture_rows, warnings = [], []
            with wave.open(lecture['audio'], 'rb') as stream:
                require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) ==
                        (16000, 1, 2, 'NONE'), 'Expected normalized mono 16 kHz PCM16 WAV')
                frames = stream.getnframes()
                require(abs(frames / 16000 - lecture['duration_seconds']) < 1e-8, 'Normalized duration mismatch')
                note_duration = raw['source'].get('totalRecordingDurationInMillis')
                if type(note_duration) not in (int, float) or not math.isfinite(note_duration):
                    warnings.append('missing_note_duration_metadata')
                    note_delta = None
                else:
                    note_delta = note_duration / 1000 - frames / 16000
                    if abs(note_delta) > 0.5:
                        warnings.append('stale_note_duration_metadata')
                for index, (paragraph, (text, field, events, unknown)) in enumerate(zip(paragraphs, texts)):
                    start, end = paragraph.get('audioStartInMillis'), paragraph.get('audioEndInMillis')
                    row = {'clip_id': f'{lid}_{index:05d}', 'lecture_id': lid, 'split': lecture['split'],
                           'paragraph_index': index, 'paragraph_uuid': paragraph['uuid'],
                           'source_start_ms': start, 'source_end_ms': end,
                           'reference_text': text, 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                           'transcript_source_field': field, 'removed_events': events, 'unrecognized_brackets': unknown,
                           'alignment_origin': 'source_timestamps', 'label_origin': 'original_tiro_transcript',
                           'source_audio_sha256': lecture['source_audio_sha256'],
                           'source_transcript_sha256': lecture['source_transcript_sha256'],
                           'source_timestamps_sha256': provenance['raw_sources'][lid]['raw_json_sha256'],
                           'human_review': 'pending', 'issues': []}
                    if not text:
                        row['issues'].append('no_speech_label')
                    if type(start) is not int or type(end) is not int:
                        row['issues'].append('missing_or_invalid_source_timestamps')
                    elif not 0 <= start < end:
                        row['issues'].append('invalid_source_interval')
                    else:
                        begin, finish = start * 16, end * 16
                        row.update(audio_start=start / 1000, audio_end=end / 1000, duration_seconds=(end - start) / 1000,
                                   start_frame=begin, end_frame=finish)
                        if finish > frames:
                            row['issues'].append('source_interval_out_of_bounds')
                        if (end - start) / 1000 > max_duration:
                            row['issues'].append('paragraph_exceeds_max_duration')
                    lecture_rows.append(row)
                flag_overlaps(lecture_rows)
                for row in lecture_rows:
                    row['issues'] = list(dict.fromkeys(row['issues']))
                    if row['issues']:
                        continue
                    stream.setpos(row['start_frame'])
                    pcm = stream.readframes(row['end_frame'] - row['start_frame'])
                    require(len(pcm) == 2 * (row['end_frame'] - row['start_frame']), 'Truncated normalized audio')
                    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float64)
                    rms = float(np.sqrt(np.mean((samples / 32768) ** 2)))
                    clipping = float(np.mean(np.abs(samples) >= 32760))
                    row['audio_validation'] = {'rms': rms, 'rms_dbfs': 20 * math.log10(rms) if rms else None,
                                               'clipping_fraction': clipping, 'frames': len(samples),
                                               'exact_normalized_pcm_slice': True}
                    if rms < 0.0005:
                        row['issues'].append('near_silent_audio')
                    if clipping > 0.01:
                        row['issues'].append('excessive_clipping')
                    if row['issues']:
                        continue
                    path = stage / 'clips' / (row['clip_id'] + '.wav')
                    with wave.open(str(path), 'wb') as clip:
                        clip.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                        clip.writeframes(pcm)
                    row.update(audio=str(output / 'clips' / path.name), audio_sha256=sha256(path),
                               pcm_sha256=hashlib.sha256(pcm).hexdigest())
            rows.extend(lecture_rows)
            lecture_reports.append({'lecture_id': lid, 'split': lecture['split'], 'source_paragraphs': len(lecture_rows),
                                    'full_original_transcript_matches': True, 'source_seconds': lecture['duration_seconds'],
                                    'note_metadata_duration_delta_seconds': note_delta, 'warnings': warnings,
                                    'last_source_interval_end_seconds': max((r['audio_end'] for r in lecture_rows if 'audio_end' in r), default=None)})
        by_pcm = {}
        for row in rows:
            if not row['issues']:
                by_pcm.setdefault(row['pcm_sha256'], []).append(row)
        for duplicates in by_pcm.values():
            if len(duplicates) > 1:
                for row in duplicates:
                    row['issues'].append('exact_duplicate_audio')
                    (stage / 'clips' / (row['clip_id'] + '.wav')).unlink()
                    for key in ('audio', 'audio_sha256', 'pcm_sha256'):
                        row.pop(key, None)
        for row in rows:
            row['automatic_status'] = 'quarantined' if row['issues'] else 'pass'
        accepted = [r for r in rows if not r['issues']]
        write_jsonl(stage / 'segments.jsonl', rows)
        write_jsonl(stage / 'quarantine.jsonl', [r for r in rows if r['issues']])
        for split in SPLITS:
            write_jsonl(stage / (split + '.jsonl'), [{'audio': r['audio'], 'text': 'language Korean<asr_text>' + r['reference_text']}
                                                      for r in accepted if r['split'] == split])
        # Original files and timestamp exports must remain unchanged through construction.
        _, _, after = validate_sources(base_dataset, output)
        require(after == provenance, 'Sources changed during timestamp preparation')
        report = {'schema_version': 1, 'pairing_complete': True, 'training_launched': False,
                  'alignment_origin': 'source_timestamps', 'human_review_of_new_clips_asserted': False,
                  'input_provenance': stamp, 'frozen_split_lock': str(base_dataset / 'split_lock.json'),
                  'source_lectures': len(lectures), 'source_paragraphs': len(rows), 'accepted_paragraphs': len(accepted),
                  'quarantined_paragraphs': len(rows) - len(accepted),
                  'issue_counts': dict(Counter(issue for row in rows for issue in row['issues'])),
                  'source_seconds': sum(l['duration_seconds'] for l in lectures.values()),
                  'accepted_seconds': sum(r['duration_seconds'] for r in accepted),
                  'lectures': lecture_reports,
                  'splits': {split: {'lecture_ids': [lid for lid, l in lectures.items() if l['split'] == split],
                                     'clips': sum(r['split'] == split for r in accepted),
                                     'seconds': sum(r['duration_seconds'] for r in accepted if r['split'] == split)} for split in SPLITS},
                  'train_file': str(output / 'train.jsonl'), 'eval_file': str(output / 'validation.jsonl'),
                  'held_out_test_file': str(output / 'test.jsonl'),
                  'limitations': ['Source paragraph timestamps are provided by the original transcript service, not new manual word-level verification.',
                                  'No times were interpolated, clamped, or inferred from Qwen ASR predictions.',
                                  'Long or invalid paragraphs are withheld whole; no guessed subparagraph cuts.',
                                  'Note-level duration may be stale; exact paragraph intervals must still fit the waveform.',
                                  'Preserved transcript labels may contain source transcription errors. Test is reserved for final evaluation.'],
                  'artifact_sha256': {name: sha256(stage / name) for name in OWNED if name not in ('clips', 'report.json')}}
        write_json(stage / 'report.json', report)
        require(not (set(p.name for p in output.iterdir()) - {'raw', 'sources.json'}), 'Output changed during construction')
        for name in OWNED:
            (stage / name).rename(output / name)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, default=ROOT / 'data/qwen3_asr')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/tiro_timestamps')
    parser.add_argument('--max-duration', type=float, default=90.0)
    args = parser.parse_args()
    try:
        report = prepare(args.base_dataset, args.output, args.max_duration)
    except (ValueError, OSError, KeyError, TypeError, wave.Error) as error:
        parser.exit(1, f'Timestamp preparation refused: {error}\n')
    print(json.dumps({key: report[key] for key in ('source_lectures', 'source_paragraphs', 'accepted_paragraphs',
                                                 'quarantined_paragraphs', 'accepted_seconds', 'issue_counts', 'splits')}, indent=2))


if __name__ == '__main__':
    main()
