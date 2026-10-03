"""Export only human-approved, provenance-checked Qwen3-ASR examples.

Corrections require realignment and fresh review hashes. This command never
launches training, uploads data, or uses the held-out test split for selection.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import wave

from prepare_qwen_dataset import clean_transcript
from qwen_alignment_checks import boundary_review_eligible

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')
BOUNDARY_REVIEW_KIND = 'sparse_internal_timestamp_anomalies'


def review_category(row):
    """Keep strict automatic passes separate from exceptional human boundary review."""
    if row.get('automatic_status') == 'pass' and not row.get('issues'):
        return 'automatic_pass'
    if (row.get('automatic_status') == 'review_required'
            and row.get('issues') == ['zero_duration_alignment_token']
            and row.get('review_kind') == BOUNDARY_REVIEW_KIND
            and isinstance(row.get('aligned_items'), list)
            and boundary_review_eligible(row['aligned_items'], row.get('reference_text'),
                                         row.get('alignment_context_duration_seconds'),
                                         row.get('cer'), row.get('match_coverage'))):
        return 'boundary_review'
    return None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def index_unique(rows, key, label):
    result = {}
    for row in rows:
        ident = row[key]
        require(isinstance(ident, str) and bool(ident), f'Invalid {label} ID')
        require(ident not in result, f'Duplicate {label} ID: {ident}')
        result[ident] = row
    return result


def pcm_stream(path, stack):
    stream = stack.enter_context(wave.open(str(path), 'rb'))
    require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
            == (16000, 1, 2, 'NONE'), f'Expected mono 16 kHz PCM16 audio: {path}')
    require(stream.getnframes() > 0, f'Empty audio: {path}')
    return stream


def finite_number(value, label):
    require(type(value) in (float, int) and math.isfinite(value), f'Invalid {label}')
    return float(value)


def sample_index(value, label):
    value = finite_number(value, label) * 16000
    require(abs(value - round(value)) < 1e-6, f'{label} is not sample-aligned')
    return round(value)


def assert_hash_split_unique(rows, key):
    seen = {}
    for row in rows:
        digest = row[key]
        require(digest not in seen or seen[digest] == row['split'],
                f'Duplicate {key} across splits')
        seen[digest] = row['split']


def validate_inputs(dataset, review_file):
    paths = {name: dataset / name for name in
             ('alignment_report.json', 'lectures.jsonl', 'split_lock.json', 'segments.jsonl')}
    require(all(p.is_file() for p in paths.values()), 'Alignment is incomplete: required artifacts are missing')
    report = json.loads(paths['alignment_report.json'].read_text())
    require(report.get('alignment_complete') is True, 'Alignment is not marked complete')
    for name, field in (('segments.jsonl', 'segments_sha256'), ('lectures.jsonl', 'lectures_sha256'),
                        ('split_lock.json', 'split_lock_sha256')):
        require(report.get(field) == file_hash(paths[name]), f'Stale alignment report: {name} changed')
    lock = json.loads(paths['split_lock.json'].read_text())
    locked = index_unique(lock['lectures'], 'lecture_id', 'locked lecture')
    lectures = index_unique(read_jsonl(paths['lectures.jsonl']), 'lecture_id', 'lecture')
    segments = index_unique(read_jsonl(paths['segments.jsonl']), 'clip_id', 'clip')
    require(set(locked) == set(lectures) and bool(lectures), 'Lecture inventory differs from split lock')
    require(report.get('source_lectures') == len(lectures), 'Alignment lecture count mismatch')
    require(report.get('windows') == len(segments), 'Alignment segment count mismatch')
    groups = {}
    for lid, lecture in lectures.items():
        require(re.fullmatch(r'[a-zA-Z0-9_-]+', lid) is not None, 'Unsafe lecture ID')
        require(all(lecture.get(key) == value for key, value in locked[lid].items()),
                f'Lecture provenance or split differs from frozen lock: {lid}')
        require(lecture['split'] in SPLITS, 'Unknown frozen split')
        group = re.sub(r'^\d+_', '', lecture['title'])
        require(group not in groups or groups[group] == lecture['split'], 'Repeated lecturer title crosses splits')
        groups[group] = lecture['split']
        for kind in ('audio', 'transcript'):
            require(file_hash(lecture['source_' + kind]) == lecture['source_' + kind + '_sha256'],
                    f'Source {kind} changed: {lid}')
        expected_audio = dataset / 'lectures' / f'{lid}.wav'
        expected_text = dataset / 'lectures' / f'{lid}.txt'
        require(lecture['audio'] == str(expected_audio) and lecture['transcript'] == str(expected_text),
                f'Unexpected normalized lecture artifact path: {lid}')
        require(not expected_audio.is_symlink() and not expected_text.is_symlink(), 'Symlinked normalized artifact')
        require(file_hash(expected_audio) == lecture['audio_sha256'], f'Normalized lecture audio changed: {lid}')
        require(file_hash(expected_text) == lecture['transcript_sha256'], f'Normalized transcript changed: {lid}')
        require(expected_text.read_text(encoding='utf-8') == lecture['text'] + '\n',
                f'Manifest text differs from normalized transcript: {lid}')
        source_text, source_paragraphs, *_ = clean_transcript(Path(lecture['source_transcript']).read_text(encoding='utf-8'))
        require(source_text == lecture['text'] and source_paragraphs == lecture['paragraphs'],
                f'Cleaned labels differ from source transcript: {lid}')
        cache_path = dataset / 'alignment' / f'{lid}.json'
        require(cache_path.is_file(), f'Missing lecture alignment cache: {lid}')
        cache = json.loads(cache_path.read_text())
        require(cache.get('complete') is True, f'Incomplete lecture alignment cache: {lid}')
        cached_rows = index_unique(cache['rows'], 'clip_id', 'cached clip')
        actual_ids = {key for key, value in segments.items() if value['lecture_id'] == lid}
        require(set(cached_rows) == actual_ids, f'Stale lecture alignment cache: {lid}')
        for cid in actual_ids:
            row = segments[cid]
            require(row['split'] == lecture['split'], f'Clip split differs from frozen lecture split: {cid}')
            require(row.get('source_audio_sha256') == lecture['source_audio_sha256'] and
                    row.get('source_transcript_sha256') == lecture['source_transcript_sha256'],
                    f'Stale segment source provenance: {cid}')
            if row.get('automatic_status') in ('pass', 'review_required'):
                category = review_category(row)
                require(category is not None, f'Clip fails declared review eligibility: {cid}')
                for key in ('reference_text', 'audio', 'audio_sha256', 'text_sha256', 'pcm_sha256',
                            'audio_start', 'audio_end', 'duration_seconds', 'automatic_status', 'issues'):
                    require(row.get(key) == cached_rows[cid].get(key), f'Stale cached alignment: {cid}/{key}')
                if category == 'boundary_review':
                    for key in ('review_kind', 'aligned_items', 'alignment_context_start',
                                'alignment_context_duration_seconds', 'cer', 'match_coverage'):
                        require(row.get(key) == cached_rows[cid].get(key), f'Stale boundary review evidence: {cid}/{key}')
    require(all(row['lecture_id'] in lectures for row in segments.values()), 'Unknown lecture in segments')
    for key in ('source_audio_sha256', 'audio_sha256'):
        assert_hash_split_unique(lectures.values(), key)
    for split in SPLITS:
        split_report = report.get('splits', {}).get(split, {})
        require(set(split_report.get('lectures', [])) == {lid for lid, l in lectures.items() if l['split'] == split},
                f'Alignment split inventory mismatch: {split}')
        require(split_report.get('candidates') == sum(r.get('automatic_status') == 'pass' and r['split'] == split
                                                     for r in segments.values()), f'Candidate count mismatch: {split}')
        require(split_report.get('boundary_review_candidates') == sum(r.get('automatic_status') == 'review_required'
                                                                    and r['split'] == split for r in segments.values()),
                f'Boundary review candidate count mismatch: {split}')
    reviews = index_unique(read_jsonl(review_file), 'clip_id', 'review')
    approved = []
    for cid, review in reviews.items():
        require(set(review) <= {'clip_id', 'audio_sha256', 'text_sha256', 'status', 'notes'},
                'Review files cannot correct labels; corrections require realignment and fresh hashes')
        require(cid in segments, f'Unknown review clip ID: {cid}')
        require(review.get('status') in ('approved', 'rejected'), f'Review status must be approved or rejected: {cid}')
        row = segments[cid]
        require(review_category(row) is not None, f'Cannot review non-pass clip outside the strict boundary-review exception: {cid}')
        require(review.get('audio_sha256') == row.get('audio_sha256') and
                review.get('text_sha256') == row.get('text_sha256'), f'Stale review hashes: {cid}')
        if review['status'] == 'approved':
            approved.append(row)
    require(all(any(r['split'] == split for r in approved) for split in SPLITS),
            'At least one approved clip is required in train, validation, and test')
    return lectures, segments, reviews, approved


def validate_approved_audio(dataset, lectures, approved):
    pcm_seen, last_end = {}, {}
    ordered = sorted(approved, key=lambda r: (r['lecture_id'], r['audio_start'], r['clip_id']))
    with ExitStack() as stack:
        sources = {lid: pcm_stream(lecture['audio'], stack) for lid, lecture in lectures.items()}
        for lid, stream in sources.items():
            require(abs(stream.getnframes() / 16000 - finite_number(lectures[lid]['duration_seconds'], 'lecture duration')) < 1e-8,
                    f'Normalized lecture duration mismatch: {lid}')
        for row in ordered:
            cid, lid = row['clip_id'], row['lecture_id']
            require(re.fullmatch(r'[a-zA-Z0-9_-]+', cid) is not None, 'Unsafe clip ID')
            expected = dataset / 'clips' / (cid + '.wav')
            require(row['audio'] == str(expected) and not expected.is_symlink(), f'Unexpected clip path: {cid}')
            require(file_hash(expected) == row['audio_sha256'], f'Clip file changed after review: {cid}')
            reference = row['reference_text']
            require(isinstance(reference, str) and bool(reference.strip()), f'Empty clip label: {cid}')
            require(hashlib.sha256(reference.encode('utf-8')).hexdigest() == row['text_sha256'], f'Clip label hash mismatch: {cid}')
            start = sample_index(row['audio_start'], 'clip start')
            end = sample_index(row['audio_end'], 'clip end')
            require(0 <= start < end <= sources[lid].getnframes(), f'Clip out of lecture bounds: {cid}')
            if review_category(row) == 'boundary_review':
                context_start = sample_index(row['alignment_context_start'], 'alignment context start')
                context_frames = sample_index(row['alignment_context_duration_seconds'], 'alignment context duration')
                context_end = context_start + context_frames
                require(0 <= context_start < context_end <= sources[lid].getnframes(), f'Invalid boundary-review context: {cid}')
                items = row['aligned_items']
                expected_start = max(context_start, context_start + round((items[0]['start_time'] - 0.03) * 16000))
                expected_end = min(context_end, context_start + round((items[-1]['end_time'] + 0.03) * 16000))
                require(start == expected_start and end == expected_end,
                        f'Boundary-review clip changed the original aligned word span: {cid}')
            duration = (end - start) / 16000
            require(1 <= duration <= 32 and abs(finite_number(row['duration_seconds'], 'clip duration') - duration) < 1e-8,
                    f'Clip duration mismatch or out of range: {cid}')
            require(lid not in last_end or start >= last_end[lid], f'Overlapping approved clips: {cid}')
            last_end[lid] = end
            with ExitStack() as clip_stack:
                clip = pcm_stream(expected, clip_stack)
                require(clip.getnframes() == end - start, f'Clip frame count mismatch: {cid}')
                pcm = clip.readframes(clip.getnframes())
                require(len(pcm) == 2 * clip.getnframes(), f'Truncated clip audio: {cid}')
            digest = hashlib.sha256(pcm).hexdigest()
            require(digest == row['pcm_sha256'], f'Clip PCM hash mismatch: {cid}')
            require(digest not in pcm_seen, f'Exact duplicate approved PCM: {cid} and {pcm_seen.get(digest)}')
            pcm_seen[digest] = cid
            sources[lid].setpos(start)
            require(pcm == sources[lid].readframes(end - start), f'Clip does not match its normalized lecture interval: {cid}')
    return ordered


def export_reviewed(dataset, review_file):
    dataset, review_file = Path(dataset).resolve(), Path(review_file).resolve()
    lectures, segments, reviews, approved = validate_inputs(dataset, review_file)
    approved = validate_approved_audio(dataset, lectures, approved)
    output = dataset / 'reviewed'
    fingerprints = {name: file_hash(dataset / name) for name in
                    ('alignment_report.json', 'lectures.jsonl', 'split_lock.json', 'segments.jsonl')}
    fingerprints['review_file'] = file_hash(review_file)
    manifests = {split: ''.join(json.dumps({'audio': r['audio'], 'text': 'language Korean<asr_text>' + r['reference_text']},
                                         ensure_ascii=False) + '\n' for r in approved if r['split'] == split)
                 for split in SPLITS}
    readiness = {'schema_version': 1, 'training_ready': True, 'training_launched': False,
                 'approval_policy': 'Only explicit approved review records with matching audio/text hashes are exported. '
                                    'Approvals attest that the full clip boundaries and verbatim text were checked. '
                                    'Boundary-review approvals also verify every word with an internal timestamp anomaly.',
                 'review_file': str(review_file), 'input_sha256': fingerprints,
                 'reviewed_records': len(reviews), 'approved_records': len(approved),
                 'automatic_pass_approved': sum(review_category(r) == 'automatic_pass' for r in approved),
                 'boundary_review_approved': sum(review_category(r) == 'boundary_review' for r in approved),
                 'rejected_records': sum(r['status'] == 'rejected' for r in reviews.values()),
                 'unreviewed_records': len(segments) - len(reviews),
                 'approved_clip_ids': [r['clip_id'] for r in approved],
                 'train_file': str(output / 'train.jsonl'), 'eval_file': str(output / 'validation.jsonl'),
                 'held_out_test_file': str(output / 'test.jsonl'),
                 'test_policy': 'Keep test.jsonl out of eval_file and model selection; use only for final evaluation.',
                 'speaker_disjoint_verified': False,
                 'splits': {split: {'clips': sum(r['split'] == split for r in approved),
                                    'automatic_pass_approved': sum(r['split'] == split and review_category(r) == 'automatic_pass' for r in approved),
                                    'boundary_review_approved': sum(r['split'] == split and review_category(r) == 'boundary_review' for r in approved),
                                    'lecture_ids': sorted({r['lecture_id'] for r in approved if r['split'] == split}),
                                    'seconds': sum(r['duration_seconds'] for r in approved if r['split'] == split)}
                            for split in SPLITS},
                 'manifest_sha256': {split: hashlib.sha256(value.encode('utf-8')).hexdigest()
                                     for split, value in manifests.items()}}
    if output.exists():
        expected_names = {'readiness_report.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'}
        require(not output.is_symlink() and set(p.name for p in output.iterdir()) == expected_names,
                'Existing reviewed directory has unknown artifacts; refusing overwrite')
        require(json.loads((output / 'readiness_report.json').read_text()) == readiness,
                'Reviewed export differs; archive the previous reviewed directory explicitly before exporting new approvals')
        require(all((output / f'{split}.jsonl').read_text(encoding='utf-8') == value for split, value in manifests.items()),
                'Reviewed manifest changed; refusing overwrite')
        return readiness
    stage = Path(tempfile.mkdtemp(prefix='.reviewed.preparing-', dir=dataset))
    try:
        for split, value in manifests.items():
            (stage / f'{split}.jsonl').write_text(value, encoding='utf-8')
        (stage / 'readiness_report.json').write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Refuse replacement if another process has published in the meantime.
        require(not output.exists(), 'Reviewed output appeared during validation; refusing overwrite')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/qwen3_asr')
    parser.add_argument('--review-file', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = export_reviewed(args.dataset, args.review_file)
    except (ValueError, OSError, KeyError, TypeError, wave.Error) as error:
        parser.exit(1, f'Reviewed export refused: {error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
