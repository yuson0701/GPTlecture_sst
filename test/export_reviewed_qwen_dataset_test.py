import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_qwen_dataset as prep
import export_reviewed_qwen_dataset as export


def save_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def write_wav(path, pcm, rate=16000, channels=1):
    with wave.open(str(path), 'wb') as handle:
        handle.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
        handle.writeframes(pcm)


class ExportTests(unittest.TestCase):
    text = '하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 cartilage'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        for index, title in enumerate(('0_샘플가 교수님', '1_샘플나 교수님', '1_샘플다 교수님')):
            samples = (np.sin(2 * np.pi * (301 + index * 73) * np.arange(384000) / 48000) * 8000).astype('<i2')
            write_wav(self.source / (title + '.m4a'), samples.tobytes(), rate=48000)
            (self.source / (title + ' 스크립트.txt')).write_text('A: ' + self.text + ' [기침]\n', encoding='utf-8')
        self.dataset = self.root / 'dataset'
        prep.prepare(self.source, self.dataset, {'splits': {'0_샘플가 교수님': 'validation', '1_샘플나 교수님': 'test', '1_샘플다 교수님': 'train'}})
        self.lectures = export.read_jsonl(self.dataset / 'lectures.jsonl')
        (self.dataset / 'clips').mkdir()
        (self.dataset / 'alignment').mkdir()
        self.rows = [self.clip(lecture, '00000', 0.25, 1.5) for lecture in self.lectures]
        self.review_file = self.dataset / 'my_reviews.jsonl'
        self.reviews = []
        self.refresh()

    def clip(self, lecture, suffix, start, end):
        cid = lecture['lecture_id'] + '_' + suffix
        path = self.dataset / 'clips' / (cid + '.wav')
        with wave.open(lecture['audio'], 'rb') as stream:
            stream.setpos(round(start * 16000))
            pcm = stream.readframes(round((end - start) * 16000))
        write_wav(path, pcm)
        text = lecture['text']
        return {'clip_id': cid, 'lecture_id': lecture['lecture_id'], 'split': lecture['split'],
                'audio': str(path), 'audio_sha256': export.file_hash(path),
                'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                'reference_text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'audio_start': start, 'audio_end': end, 'duration_seconds': end - start,
                'automatic_status': 'pass', 'human_review': 'pending', 'issues': [],
                'source_audio_sha256': lecture['source_audio_sha256'],
                'source_transcript_sha256': lecture['source_transcript_sha256']}

    def refresh(self):
        save_jsonl(self.dataset / 'lectures.jsonl', self.lectures)
        save_jsonl(self.dataset / 'segments.jsonl', self.rows)
        for lecture in self.lectures:
            cache = {'complete': True, 'rows': [r for r in self.rows if r['lecture_id'] == lecture['lecture_id']]}
            (self.dataset / 'alignment' / (lecture['lecture_id'] + '.json')).write_text(json.dumps(cache))
        self.report = {'alignment_complete': True, 'source_lectures': len(self.lectures), 'windows': len(self.rows),
                       'splits': {split: {'lectures': [l['lecture_id'] for l in self.lectures if l['split'] == split],
                                          'candidates': sum(r['split'] == split and r['automatic_status'] == 'pass' for r in self.rows),
                                          'boundary_review_candidates': sum(r['split'] == split and r['automatic_status'] == 'review_required' for r in self.rows)}
                                  for split in export.SPLITS}}
        for name, field in (('lectures.jsonl', 'lectures_sha256'), ('segments.jsonl', 'segments_sha256'),
                            ('split_lock.json', 'split_lock_sha256')):
            self.report[field] = export.file_hash(self.dataset / name)
        (self.dataset / 'alignment_report.json').write_text(json.dumps(self.report))
        self.reviews = [{key: row[key] for key in ('clip_id', 'audio_sha256', 'text_sha256')} | {'status': 'approved', 'notes': ''}
                        for row in self.rows]
        self.save_reviews()

    def save_reviews(self):
        save_jsonl(self.review_file, self.reviews)

    def export(self):
        return export.export_reviewed(self.dataset, self.review_file)

    def test_approved_export_uses_official_schema_and_validation_for_eval(self):
        report = self.export()
        self.assertTrue(report['training_ready'])
        self.assertFalse(report['training_launched'])
        self.assertTrue(report['eval_file'].endswith('/validation.jsonl'))
        self.assertTrue(report['held_out_test_file'].endswith('/test.jsonl'))
        for split in export.SPLITS:
            rows = export.read_jsonl(self.dataset / 'reviewed' / (split + '.jsonl'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {'audio', 'text'})
            self.assertEqual(rows[0]['text'], 'language Korean<asr_text>' + self.text)
        self.assertEqual(self.export(), report)

    def boundary_review_row(self):
        row = self.clip(self.lectures[0], '00000', 0.25, 4.21)
        items = [{'text': word, 'start_time': round(0.28 + index * 0.4, 2),
                  'end_time': round(0.28 + index * 0.4 + (0 if index == 4 else 0.3), 2)}
                 for index, word in enumerate(self.text.split())]
        row.update(automatic_status='review_required', issues=['zero_duration_alignment_token'],
                   review_kind='sparse_internal_timestamp_anomalies', aligned_items=items,
                   alignment_context_start=0.0, alignment_context_duration_seconds=8.0,
                   cer=0.08, match_coverage=0.92)
        self.rows[0] = row
        return row

    def test_sparse_internal_zero_requires_explicit_human_approval_and_separate_counts(self):
        row = self.boundary_review_row()
        self.refresh()
        self.reviews[0]['status'] = 'pending'
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'approved or rejected'):
            self.export()
        self.reviews[0]['status'] = 'approved'
        self.reviews[0]['notes'] = 'Listened to the full clip; all words and both boundaries verified.'
        self.save_reviews()
        report = self.export()
        self.assertEqual(report['automatic_pass_approved'], 2)
        self.assertEqual(report['boundary_review_approved'], 1)
        self.assertEqual(row['issues'], ['zero_duration_alignment_token'])
        self.assertEqual(row['automatic_status'], 'review_required')
        self.assertEqual(export.read_jsonl(self.dataset / 'reviewed' / 'validation.jsonl')[0]['text'],
                         'language Korean<asr_text>' + self.text)

    def test_collapsed_boundary_additional_issues_and_poor_match_never_export(self):
        for mutation in ('collapsed_boundary', 'extra_issue', 'poor_cer', 'wrong_review_kind'):
            with self.subTest(mutation=mutation):
                row = self.boundary_review_row()
                if mutation == 'collapsed_boundary':
                    row['aligned_items'][0]['end_time'] = row['aligned_items'][0]['start_time']
                elif mutation == 'extra_issue':
                    row['issues'].append('long_internal_alignment_gap')
                elif mutation == 'poor_cer':
                    row['cer'] = 0.151
                else:
                    row['review_kind'] = 'generic_override'
                self.refresh()
                with self.assertRaisesRegex(ValueError, 'review eligibility'):
                    self.export()
        self.assertFalse((self.dataset / 'reviewed').exists())

    def test_boundary_review_cannot_shrink_the_full_aligned_reference_span(self):
        row = self.boundary_review_row()
        changed_clip = self.clip(self.lectures[0], '00000', 0.35, 4.21)
        for key in ('audio_start', 'audio_end', 'duration_seconds', 'audio_sha256', 'pcm_sha256'):
            row[key] = changed_clip[key]
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'original aligned word span'):
            self.export()

    def test_missing_completion_marker_and_stale_report_refuse_output(self):
        self.report.pop('alignment_complete')
        (self.dataset / 'alignment_report.json').write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'not marked complete'):
            self.export()
        self.refresh()
        with (self.dataset / 'segments.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'Stale alignment report'):
            self.export()
        self.assertFalse((self.dataset / 'reviewed').exists())

    def test_unknown_and_duplicate_review_ids_are_rejected(self):
        self.reviews.append(dict(self.reviews[0]))
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'Duplicate review'):
            self.export()
        self.reviews[-1]['clip_id'] = 'unknown_clip'
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'Unknown review'):
            self.export()

    def test_pending_reviews_and_missing_approved_partition_are_rejected(self):
        self.reviews[0]['status'] = 'pending'
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'approved or rejected'):
            self.export()
        self.reviews[0]['status'] = 'rejected'
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'At least one approved'):
            self.export()
        self.reviews.pop(0)
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'At least one approved'):
            self.export()

    def test_stale_review_hashes_and_inline_label_corrections_fail(self):
        self.reviews[0]['text_sha256'] = '0' * 64
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'Stale review hashes'):
            self.export()
        self.refresh()
        self.reviews[0]['reference_text'] = 'Corrected label'
        self.save_reviews()
        with self.assertRaisesRegex(ValueError, 'corrections require realignment'):
            self.export()

    def test_nonpass_approval_and_changed_split_fail(self):
        self.rows[0]['automatic_status'] = 'quarantined'
        self.rows[0]['issues'] = ['high_cer']
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'non-pass'):
            self.export()
        self.rows[0]['automatic_status'] = 'pass'
        self.rows[0]['issues'] = []
        self.rows[0]['split'] = 'train'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'split differs'):
            self.export()

    def test_changed_source_and_manifest_labels_fail(self):
        source = Path(self.lectures[0]['source_transcript'])
        original = source.read_text()
        source.write_text('A: changed source')
        with self.assertRaisesRegex(ValueError, 'Source transcript changed'):
            self.export()
        source.write_text(original)
        self.lectures[0]['text'] = 'tampered label'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Manifest text differs'):
            self.export()

    def test_changed_pcm_and_wrong_format_fail_even_with_fresh_review_hashes(self):
        row = self.rows[0]
        with wave.open(row['audio'], 'rb') as handle:
            pcm = handle.readframes(handle.getnframes())
        write_wav(row['audio'], pcm, channels=2)
        row['audio_sha256'] = export.file_hash(row['audio'])
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'PCM16'):
            self.export()
        samples = np.frombuffer(pcm, dtype='<i2').copy()
        samples[200] += 1
        changed_pcm = samples.tobytes()
        write_wav(row['audio'], changed_pcm)
        row['audio_sha256'] = export.file_hash(row['audio'])
        row['pcm_sha256'] = hashlib.sha256(changed_pcm).hexdigest()
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'does not match its normalized lecture interval'):
            self.export()

    def test_altered_frame_count_fails(self):
        row = self.rows[0]
        with wave.open(row['audio'], 'rb') as handle:
            pcm = handle.readframes(handle.getnframes() - 1)
        write_wav(row['audio'], pcm)
        row['audio_sha256'] = export.file_hash(row['audio'])
        row['pcm_sha256'] = hashlib.sha256(pcm).hexdigest()
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'frame count mismatch'):
            self.export()

    def test_overlapping_clips_and_exact_duplicate_pcm_fail(self):
        self.rows.append(self.clip(self.lectures[0], '00001', 0.5, 1.75))
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Overlapping approved'):
            self.export()
        self.rows.pop()
        first, second = self.rows[:2]
        pcm = None
        with wave.open(first['audio'], 'rb') as handle:
            pcm = handle.readframes(handle.getnframes())
        write_wav(second['audio'], pcm)
        second['audio_sha256'] = export.file_hash(second['audio'])
        second['pcm_sha256'] = hashlib.sha256(pcm).hexdigest()
        lecture = next(l for l in self.lectures if l['lecture_id'] == second['lecture_id'])
        with wave.open(lecture['audio'], 'rb') as handle:
            original = bytearray(handle.readframes(handle.getnframes()))
        start = round(second['audio_start'] * 16000) * 2
        original[start:start + len(pcm)] = pcm
        write_wav(lecture['audio'], bytes(original))
        lecture['audio_sha256'] = export.file_hash(lecture['audio'])
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Exact duplicate approved PCM'):
            self.export()

    def test_unknown_existing_export_is_preserved(self):
        output = self.dataset / 'reviewed'
        output.mkdir()
        marker = output / 'user_notes.txt'
        marker.write_text('preserve this')
        with self.assertRaisesRegex(ValueError, 'unknown artifacts'):
            self.export()
        self.assertEqual(marker.read_text(), 'preserve this')

    def test_cross_split_identical_source_recording_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'across splits'):
            export.assert_hash_split_unique([{'split': 'train', 'hash': 'same'}, {'split': 'test', 'hash': 'same'}], 'hash')


if __name__ == '__main__':
    unittest.main()
