"""Prepare local lecture inventory; deliberately never create training examples.

Run with .venv-qwen/bin/python. This decodes audio but does not download models,
infer transcript timestamps, or claim that exported transcripts are verified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import wave


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
EVENTS = {
    '박수 소리', '박수', '웃음', '숨을 내쉬는 소리', '영상 소리', '문 닫는 소리',
    '헛기침', '기침', '기지개 켜는 소리', 'background sneeze', '종이 소리',
    '전동 드릴 소리', '목 가다듬는 소리', '의자 소리', 'background noise',
}


def canonical_title(value):
    return re.sub(r'^(\d+)_\s*', r'\1_', unicodedata.normalize('NFC', value).strip())


def lecture_id(title):
    return 'lecture_' + hashlib.sha256(canonical_title(title).encode('utf-8')).hexdigest()[:12]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_split_config(value):
    """Keep lecture titles, partition decisions, and audit notes in private data."""
    if not isinstance(value, dict):
        value = json.loads(Path(value).read_text(encoding='utf-8'))
    if not isinstance(value.get('splits'), dict) or not value['splits']:
        raise ValueError('Private split configuration requires a nonempty splits mapping')
    splits, issues = {}, {}
    for title, split in value['splits'].items():
        title = canonical_title(title)
        if title in splits or split not in ('train', 'validation', 'test'):
            raise ValueError('Ambiguous title or unsupported split in private configuration')
        splits[title] = split
    for title, entries in value.get('lecture_issues', {}).items():
        title = canonical_title(title)
        if title not in splits or not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
            raise ValueError('Invalid lecture_issues in private split configuration')
        issues[title] = entries
    return {'splits': splits, 'lecture_issues': issues}


def discover_pairs(source, split_config):
    config = read_split_config(split_config)
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError(f'Source directory does not exist: {source}')
    audio, scripts, ignored = {}, {}, []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        normalized = unicodedata.normalize('NFC', path.name)
        if path.suffix.lower() == '.m4a':
            title, target = canonical_title(path.stem), audio
        elif normalized.endswith(' 스크립트.txt'):
            title, target = canonical_title(normalized[:-len(' 스크립트.txt')]), scripts
        else:
            ignored.append(path.name)
            continue
        if path.is_symlink():
            raise ValueError(f'Source symlinks are not supported: {path}')
        if title in target:
            raise ValueError(f'Ambiguous normalized filename for {title}')
        target[title] = path.resolve()
    if not audio or set(audio) != set(scripts):
        raise ValueError(f'Unpaired audio/transcripts: audio-only={sorted(set(audio)-set(scripts))}; '
                         f'transcript-only={sorted(set(scripts)-set(audio))}')
    unknown = set(audio) - set(config['splits'])
    if unknown:
        raise ValueError(f'No frozen lecture split for: {sorted(unknown)}. Review the private split configuration explicitly.')
    # A filename is only a grouping hint, never a verified speaker identity.
    groups = {}
    pairs = []
    for title in sorted(audio):
        group = re.sub(r'^\d+_', '', title)
        split = config['splits'][title]
        if group in groups and groups[group] != split:
            raise ValueError(f'Repeated lecturer title crosses splits: {group}')
        groups[group] = split
        pairs.append({'lecture_id': lecture_id(title), 'title': title, 'split': split,
                      'source_audio': str(audio[title]), 'source_transcript': str(scripts[title]),
                      'source_audio_sha256': sha256(audio[title]),
                      'source_transcript_sha256': sha256(scripts[title])})
    if len({p['lecture_id'] for p in pairs}) != len(pairs):
        raise ValueError('Lecture ID collision')
    return pairs, ignored


def reject_cross_split_duplicates(rows, key):
    seen = {}
    for row in rows:
        digest = row[key]
        if digest in seen and seen[digest] != row['split']:
            raise ValueError(f'Exact duplicate {key} crosses splits')
        seen[digest] = row['split']


def clean_transcript(raw):
    """Return NFC text, speech paragraphs, exact removal audit, and export metadata."""
    lines = unicodedata.normalize('NFC', raw).lstrip('\ufeff').splitlines()
    metadata = {}
    start = 0
    if lines and lines[0].strip() == '---':
        closing = next((i for i in range(1, len(lines)) if lines[i].strip() == '---'), None)
        if closing is None:
            raise ValueError('Unclosed transcript frontmatter')
        for line in lines[1:closing]:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip().strip('"')
        start = closing + 1
    paragraphs, audit, unknown = [], [], []
    pending = None
    first_content = True

    def finish():
        nonlocal pending
        if pending is not None:
            value = ' '.join(pending.pop('parts')).strip()
            value = re.sub(r'[ \t]+', ' ', value)
            if value:
                pending['text'] = value
                paragraphs.append(pending)
            pending = None

    for i in range(start, len(lines)):
        line = lines[i].strip()
        if not line:
            finish()
            continue
        if first_content and re.match(r'^#\s+', line):
            first_content = False
            metadata['markdown_title'] = re.sub(r'^#\s+', '', line)
            continue
        first_content = False
        match = re.match(r'^([A-Z]):\s*', line)
        if match:
            finish()
            speaker = match.group(1)
            line = line[match.end():]
        else:
            speaker = None
        if pending is None:
            pending = {'source_line': i + 1, 'source_line_end': i + 1, 'speaker': speaker,
                       'parts': [], 'removed_events': []}
        pending['source_line_end'] = i + 1

        def remove_event(match):
            event = match.group(0)
            entry = {'source_line': i + 1, 'event': event}
            if match.group(1).strip().lower() in EVENTS:
                audit.append(entry)
                pending['removed_events'].append(event)
                return ' '
            unknown.append(entry)
            return event

        pending['parts'].append(re.sub(r'\[([^\]\n]+)\]', remove_event, line))
    finish()
    if not paragraphs:
        raise ValueError('Transcript has no speech after metadata and event removal')
    text = '\n\n'.join(p['text'] for p in paragraphs)
    cursor = 0
    for index, paragraph in enumerate(paragraphs):
        paragraph.update(paragraph_id=f'p{index:04d}', char_start=cursor,
                         char_end=cursor + len(paragraph['text']))
        cursor = paragraph['char_end'] + 2
    return text, paragraphs, audit, metadata, unknown


def normalize_audio(source, destination):
    """Decode every frame locally, resample to mono PCM16, and measure integrity."""
    import av
    import numpy as np
    count = near_zero = clipped = 0
    squares = 0.0
    peak = 0
    with av.open(str(source)) as container:
        streams = list(container.streams.audio)
        if len(streams) != 1:
            raise ValueError(f'Expected exactly one audio stream: {source}')
        stream = streams[0]
        header_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        source_metadata = {'codec': stream.codec_context.name,
                           'sample_rate': stream.codec_context.sample_rate,
                           'channels': stream.codec_context.channels,
                           'header_duration_seconds': header_duration}
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        with wave.open(str(destination), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)

            def write_frame(frame):
                nonlocal count, near_zero, clipped, squares, peak
                samples = frame.to_ndarray().reshape(-1).astype('<i2', copy=False)
                absolute = np.abs(samples.astype(np.int32))
                count += len(samples)
                near_zero += int(np.count_nonzero(absolute <= 32))
                clipped += int(np.count_nonzero(absolute >= 32760))
                peak = max(peak, int(absolute.max(initial=0)))
                squares += float(np.sum(samples.astype(np.float64) ** 2))
                output.writeframesraw(samples.tobytes())

            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    write_frame(converted)
            for converted in resampler.resample(None):
                write_frame(converted)
    if not count or peak == 0:
        raise ValueError(f'Empty or entirely silent audio: {source}')
    duration = count / 16000
    if header_duration is not None and abs(duration - header_duration) > max(0.25, header_duration * 0.001):
        raise ValueError(f'Decoded/header duration mismatch: {duration} vs {header_duration}: {source}')
    with wave.open(str(destination), 'rb') as check:
        if (check.getnchannels(), check.getsampwidth(), check.getframerate(), check.getnframes()) != (1, 2, 16000, count):
            raise ValueError('Normalized WAV failed structural validation')
    rms = math.sqrt(squares / count) / 32768
    metrics = {'decoded_all_frames': True, 'frames': count, 'peak_amplitude': peak / 32768,
               'rms_dbfs': 20 * math.log10(rms), 'near_silence_sample_fraction': near_zero / count,
               'clipped_sample_fraction': clipped / count,
               'near_silence_threshold_pcm16': 32, 'clipping_threshold_pcm16': 32760,
               'note': 'Sample amplitude checks only; not speech detection or transcript alignment.'}
    issues = []
    if near_zero / count > 0.5:
        issues.append('high_near_silence_sample_fraction')
    if clipped / count > 0.001:
        issues.append('clipping_review_required')
    if rms < 0.003:
        issues.append('low_audio_level_review_required')
    return duration, source_metadata, metrics, issues


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare(source, output, split_config=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output and source directories must not overlap')
    if split_config is None:
        existing_lock = output / 'split_lock.json'
        if not existing_lock.is_file():
            raise ValueError('An explicit private --split-config is required before the first preparation')
        prior = json.loads(existing_lock.read_text())
        split_config = {'splits': {row['title']: row['split'] for row in prior['lectures']}}
    config = read_split_config(split_config)
    pairs, ignored = discover_pairs(source, config)
    reject_cross_split_duplicates(pairs, 'source_audio_sha256')
    lock = {'schema_version': SCHEMA_VERSION,
            'policy': 'Frozen whole-lecture splits; no random segment split; titles are not verified speaker IDs.',
            'speaker_disjoint_verified': False, 'lectures': pairs}
    entries = set(p.name for p in output.iterdir()) if output.exists() else set()
    allowed_existing = {'model_cache', 'source_config.json'}
    cache_only = not (entries - allowed_existing) and all(not (output / name).is_symlink() for name in entries)
    if 'model_cache' in entries:
        cache_only = cache_only and (output / 'model_cache').is_dir()
    if 'source_config.json' in entries:
        cache_only = cache_only and (output / 'source_config.json').is_file()
    if entries and not cache_only:
        lock_path = output / 'split_lock.json'
        if not lock_path.is_file() or json.loads(lock_path.read_text()) != lock:
            raise ValueError('Existing split lock missing or mismatched; refusing to overwrite or silently re-split')
        manifest_path = output / 'lectures.jsonl'
        if not manifest_path.is_file():
            raise ValueError('Existing preparation is incomplete; refusing to overwrite unknown artifacts')
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
        if {r['lecture_id'] for r in rows} != {p['lecture_id'] for p in pairs} or len(rows) != len(pairs):
            raise ValueError('Existing lecture manifest does not match split lock')
        for row in rows:
            pair = next(p for p in pairs if p['lecture_id'] == row['lecture_id'])
            if any(row.get(key) != value for key, value in pair.items()):
                raise ValueError('Existing manifest provenance/split differs from lock')
            expected_audio = output / 'lectures' / (row['lecture_id'] + '.wav')
            expected_text = output / 'lectures' / (row['lecture_id'] + '.txt')
            if row['audio'] != str(expected_audio) or row['transcript'] != str(expected_text):
                raise ValueError('Unexpected managed artifact path')
            if sha256(expected_audio) != row['audio_sha256'] or sha256(expected_text) != row['transcript_sha256']:
                raise ValueError('Existing normalized artifact changed; refusing to overwrite')
            text, paragraphs, *_ = clean_transcript(Path(row['source_transcript']).read_text(encoding='utf-8'))
            if text != row['text'] or paragraphs != row['paragraphs'] or expected_text.read_text(encoding='utf-8') != text + '\n':
                raise ValueError('Existing manifest text or paragraph boundaries differ from source transcript')
        reject_cross_split_duplicates(rows, 'audio_sha256')
        report_path = output / 'report.json'
        if not report_path.is_file():
            raise ValueError('Existing preparation report is missing')
        return json.loads(report_path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        (stage / 'lectures').mkdir()
        rows = []
        for pair in pairs:
            lid = pair['lecture_id']
            text, paragraphs, events, metadata, unknown = clean_transcript(Path(pair['source_transcript']).read_text(encoding='utf-8'))
            wav = stage / 'lectures' / (lid + '.wav')
            transcript = stage / 'lectures' / (lid + '.txt')
            duration, original_audio, validation, audio_issues = normalize_audio(pair['source_audio'], wav)
            transcript.write_text(text + '\n', encoding='utf-8')
            issues = ['no_source_timestamps', 'transcript_accuracy_unverified', 'speaker_identity_unverified'] + audio_issues
            if unknown:
                issues.append('unrecognized_bracket_content_preserved')
            issues = list(dict.fromkeys(issues + config['lecture_issues'].get(pair['title'], [])))
            rows.append({**pair, 'schema_version': SCHEMA_VERSION,
                         'audio': str(output / 'lectures' / wav.name), 'audio_sha256': sha256(wav),
                         'transcript': str(output / 'lectures' / transcript.name), 'transcript_sha256': sha256(transcript),
                         'text': text, 'paragraphs': paragraphs, 'removed_events': events,
                         'unrecognized_brackets': unknown, 'export_metadata': metadata,
                         'duration_seconds': duration, 'sample_rate': 16000, 'channels': 1, 'sample_width_bytes': 2,
                         'source_audio_metadata': original_audio, 'audio_validation': validation,
                         'alignment_status': 'pending', 'eligible_for_training': False,
                         'issues': issues})
        reject_cross_split_duplicates(rows, 'audio_sha256')
        # Ensure original files did not change while decoding.
        for pair in pairs:
            for kind in ('audio', 'transcript'):
                if sha256(pair['source_' + kind]) != pair['source_' + kind + '_sha256']:
                    raise ValueError('Source changed during preparation')
        write_json(stage / 'split_lock.json', lock)
        (stage / 'lectures.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        report = {'schema_version': SCHEMA_VERSION, 'source_directory': str(source), 'output_directory': str(output),
                  'lecture_count': len(rows), 'duration_seconds': sum(r['duration_seconds'] for r in rows),
                  'speech_paragraph_count': sum(len(r['paragraphs']) for r in rows),
                  'removed_event_count': sum(len(r['removed_events']) for r in rows),
                  'ignored_source_files': ignored, 'alignment_status': 'pending', 'training_example_count': 0,
                  'source_segment_timestamps_available': False, 'speaker_disjoint_verified': False,
                  'note': 'Export creation timestamps are metadata, not alignment timestamps. '
                          'Full lectures and paragraphs are pending alignment and reference review. No training manifests are created.',
                  'splits': {split: {'lecture_ids': [r['lecture_id'] for r in rows if r['split'] == split],
                                     'duration_seconds': sum(r['duration_seconds'] for r in rows if r['split'] == split)}
                             for split in ('train', 'validation', 'test')},
                  'lectures': [{k: r[k] for k in ('lecture_id', 'title', 'split', 'duration_seconds', 'audio_validation', 'issues')}
                               for r in rows]}
        write_json(stage / 'report.json', report)
        if output.exists():
            remaining = set(p.name for p in output.iterdir())
            if remaining - allowed_existing:
                raise ValueError('Output changed during preparation; refusing to overwrite')
            # Preserve the separate alignment tool's local model cache. Publish the
            # manifest last, after every file it references is ready.
            for name in ('lectures', 'split_lock.json', 'report.json', 'lectures.jsonl'):
                (stage / name).rename(output / name)
        else:
            stage.rename(output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'voiceinput_output')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/qwen3_asr')
    parser.add_argument('--split-config', type=Path,
                        help='Private JSON containing splits and optional lecture_issues; reruns use the existing frozen lock by default')
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output, args.split_config)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Preparation refused: {error}\n')
    print(json.dumps({key: report[key] for key in ('lecture_count', 'duration_seconds', 'speech_paragraph_count',
                                                  'alignment_status', 'training_example_count', 'splits')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
