"""Integrity and source-time boundary tests; all fixture names/text are fictional."""
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
import prepare_qwen_dataset as base
import prepare_tiro_dataset as tiro


class SourceTimestampTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.base = self.root / 'base'
        self.output = self.root / 'timestamps'
        self.titles = {'0_샘플가': 'validation', '1_샘플나': 'test', '1_샘플다': 'train'}
        for index, title in enumerate(self.titles):
            t = np.arange(6 * 16000) / 16000
            # Each lecture and interval is distinct, avoiding accidental duplicates.
            pcm = (8000 * np.sin(2 * np.pi * ((301 + index * 73) * t + 3.7 * t * t))).astype('<i2').tobytes()
            with wave.open(str(self.source / (title + '.m4a')), 'wb') as handle:
                handle.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                handle.writeframes(pcm)
            (self.source / (title + ' 스크립트.txt')).write_text(
                'A: 첫 문장 alpha.\n\nB: 두 번째 beta. [기침]\n', encoding='utf-8')
        base.prepare(self.source, self.base, {'splits': self.titles})
        self.lectures = tiro.read_jsonl(self.base / 'lectures.jsonl')
        (self.output / 'raw').mkdir(parents=True)
        sources = []
        for lecture in self.lectures:
            lid = lecture['lecture_id']
            url = 'https://example.invalid/' + lid
            sources.append({'lecture_id': lid, 'url': url})
            self.save_raw(lid, {'lecture_id': lid, 'url': url, 'page_sha256': hashlib.sha256(lid.encode()).hexdigest(),
                               'source': {'title': lecture['title'], 'sourceType': 'fixture',
                                          'totalRecordingDurationInMillis': 6000,
                                          'paragraphs': [
                                              self.paragraph('one', 250, 2000, '첫 문장 alpha.'),
                                              self.paragraph('two', 3000, 5000, '두 번째 beta. [기침]')]}})
        tiro.write_json(self.output / 'sources.json', {'sources': sources})
        self.lid = self.lectures[0]['lecture_id']

    @staticmethod
    def paragraph(identifier, start, end, text):
        return {'uuid': identifier, 'audioStartInMillis': start, 'audioEndInMillis': end,
                'diarizedTranscript': {'segments': [{'content': text, 'speaker': {'label': 'A'}}]},
                'rawTranscript': 'Never prefer this alternate field over the selected original.', 'hasAudio': False}

    def save_raw(self, lid, raw):
        data = '<script id="__NEXT_DATA__" type="application/json">' + json.dumps(
            {'props': {'pageProps': {'sharedNote': raw['source']}}}, ensure_ascii=False) + '</script>'
        page = self.output / 'raw' / (lid + '.html')
        page.write_bytes(data.encode('utf-8'))
        raw['page_sha256'] = base.sha256(page)
        tiro.write_json(self.output / 'raw' / (lid + '.json'), raw)

    def change_raw(self, mutate, lid=None):
        lid = lid or self.lid
        path = self.output / 'raw' / (lid + '.json')
        raw = json.loads(path.read_text())
        mutate(raw)
        self.save_raw(lid, raw)

    def run_preparation(self, **kwargs):
        return tiro.prepare(self.base, self.output, **kwargs)

    def rows(self):
        return tiro.read_jsonl(self.output / 'segments.jsonl')

    def assert_no_manifest(self):
        self.assertFalse((self.output / 'train.jsonl').exists())
        self.assertFalse((self.output / 'clips').exists())
        self.assertEqual(list(self.root.glob('.timestamps.preparing-*')), [])

    def test_exact_pcm_slices_original_labels_disjoint_splits_and_gaps(self):
        report = self.run_preparation()
        self.assertEqual((report['source_paragraphs'], report['accepted_paragraphs']), (6, 6))
        self.assertEqual(report['eval_file'], str(self.output / 'validation.jsonl'))
        self.assertFalse(report['human_review_of_new_clips_asserted'])
        self.assertEqual(self.run_preparation(), report)
        for row in self.rows():
            lecture = next(l for l in self.lectures if l['lecture_id'] == row['lecture_id'])
            self.assertEqual(row['split'], lecture['split'])
            self.assertEqual(row['human_review'], 'pending')
            self.assertEqual(row['alignment_origin'], 'source_timestamps')
            self.assertNotIn('[기침]', row['reference_text'])
            with wave.open(lecture['audio'], 'rb') as handle:
                handle.setpos(row['source_start_ms'] * 16)
                original = handle.readframes((row['source_end_ms'] - row['source_start_ms']) * 16)
            with wave.open(row['audio'], 'rb') as handle:
                self.assertEqual(original, handle.readframes(handle.getnframes()))
            self.assertEqual(row['duration_seconds'], 1.75 if row['paragraph_index'] == 0 else 2)
        for split in tiro.SPLITS:
            official = tiro.read_jsonl(self.output / (split + '.jsonl'))
            self.assertEqual(len(official), 2)
            self.assertTrue(all(set(row) == {'audio', 'text'} for row in official))
            self.assertEqual(official[0]['text'], 'language Korean<asr_text>첫 문장 alpha.')

    def test_absent_time_is_quarantined_without_guessing(self):
        self.change_raw(lambda r: r['source']['paragraphs'][0].pop('audioEndInMillis'))
        self.assertEqual(self.run_preparation()['accepted_paragraphs'], 5)
        row = next(r for r in self.rows() if r['lecture_id'] == self.lid and r['paragraph_index'] == 0)
        self.assertEqual(row['issues'], ['missing_or_invalid_source_timestamps'])
        self.assertNotIn('audio', row)
        self.assertIsNone(row['source_end_ms'])

    def test_both_overlapping_paragraphs_are_withheld(self):
        self.change_raw(lambda r: r['source']['paragraphs'][0].update(audioEndInMillis=3500))
        report = self.run_preparation()
        self.assertEqual(report['accepted_paragraphs'], 4)
        self.assertEqual(report['issue_counts']['overlapping_source_timestamps'], 2)
        self.assertTrue(all('audio' not in r for r in self.rows() if r['lecture_id'] == self.lid))

    def test_reversed_source_order_is_withheld_even_without_overlap(self):
        self.change_raw(lambda r: (r['source']['paragraphs'][0].update(audioStartInMillis=4000, audioEndInMillis=5000),
                                   r['source']['paragraphs'][1].update(audioStartInMillis=500, audioEndInMillis=1500)))
        report = self.run_preparation()
        self.assertEqual(report['issue_counts']['unordered_source_timestamps'], 2)
        self.assertEqual(report['accepted_paragraphs'], 4)

    def test_out_of_bounds_not_silently_clamped(self):
        self.change_raw(lambda r: r['source']['paragraphs'][1].update(audioEndInMillis=6001))
        self.assertEqual(self.run_preparation()['accepted_paragraphs'], 5)
        row = next(r for r in self.rows() if r['lecture_id'] == self.lid and r['paragraph_index'] == 1)
        self.assertEqual(row['source_end_ms'], 6001)
        self.assertEqual(row['issues'], ['source_interval_out_of_bounds'])
        self.assertNotIn('audio', row)

    def test_long_paragraph_is_withheld_whole_and_gap_is_not_used_to_split(self):
        self.change_raw(lambda r: (r['source']['paragraphs'][0].update(audioStartInMillis=0, audioEndInMillis=4000),
                                   r['source']['paragraphs'][1].update(audioStartInMillis=4500, audioEndInMillis=5500)))
        report = self.run_preparation(max_duration=3)
        self.assertEqual(report['source_paragraphs'], 6)
        self.assertEqual(report['accepted_paragraphs'], 5)
        self.assertEqual(report['issue_counts'], {'paragraph_exceeds_max_duration': 1})

    def test_stale_note_duration_warns_but_does_not_replace_exact_intervals(self):
        self.change_raw(lambda r: r['source'].update(totalRecordingDurationInMillis=1000))
        report = self.run_preparation()
        lecture = next(r for r in report['lectures'] if r['lecture_id'] == self.lid)
        self.assertEqual(lecture['warnings'], ['stale_note_duration_metadata'])
        self.assertEqual(lecture['note_metadata_duration_delta_seconds'], -5)
        self.assertEqual(report['accepted_paragraphs'], 6)

    def test_complete_original_text_must_match(self):
        self.change_raw(lambda r: r['source']['paragraphs'][0]['diarizedTranscript']['segments'][0].update(content='different label'))
        with self.assertRaisesRegex(ValueError, 'entire original transcript'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_source_title_must_match_original(self):
        self.change_raw(lambda r: r['source'].update(title='unrelated title'))
        with self.assertRaisesRegex(ValueError, 'Share title'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_user_override_cannot_silently_replace_original_labels(self):
        self.change_raw(lambda r: r['source']['paragraphs'][0].update(userTranscript='edited words'))
        with self.assertRaisesRegex(ValueError, 'entire original transcript'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_frozen_split_cannot_change(self):
        self.lectures[0]['split'] = 'train'
        tiro.write_jsonl(self.base / 'lectures.jsonl', self.lectures)
        with self.assertRaisesRegex(ValueError, 'Split/source provenance changed'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_duplicate_recordings_cannot_cross_splits(self):
        self.lectures[1]['audio_sha256'] = self.lectures[0]['audio_sha256']
        tiro.write_jsonl(self.base / 'lectures.jsonl', self.lectures)
        with self.assertRaisesRegex(ValueError, 'Duplicate recording crosses splits'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_repeated_lecturer_title_cannot_cross_splits(self):
        title = '2_' + self.lectures[0]['title'].split('_', 1)[1]
        self.lectures[1]['title'] = title
        tiro.write_jsonl(self.base / 'lectures.jsonl', self.lectures)
        path = self.base / 'split_lock.json'
        lock = json.loads(path.read_text())
        lock['lectures'][1]['title'] = title
        tiro.write_json(path, lock)
        with self.assertRaisesRegex(ValueError, 'Repeated lecturer title crosses splits'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_source_mutation_is_rejected(self):
        path = Path(self.lectures[0]['source_transcript'])
        path.write_text(path.read_text() + '\nchanged')
        with self.assertRaisesRegex(ValueError, 'Changed source transcript'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_sanitized_timestamps_must_match_cached_page(self):
        path = self.output / 'raw' / (self.lid + '.json')
        raw = json.loads(path.read_text())
        raw['source']['paragraphs'][0]['audioStartInMillis'] = 300
        tiro.write_json(path, raw)
        with self.assertRaisesRegex(ValueError, 'differs from cached source page'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_page_hash_is_verified(self):
        path = self.output / 'raw' / (self.lid + '.html')
        path.write_text(path.read_text() + 'altered')
        with self.assertRaisesRegex(ValueError, 'source page hash mismatch'):
            self.run_preparation()
        self.assert_no_manifest()

    def test_unknown_output_is_not_overwritten(self):
        marker = self.output / 'unrelated.txt'
        marker.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'Unknown output artifacts'):
            self.run_preparation()
        self.assertEqual(marker.read_text(), 'preserve')
        self.assert_no_manifest()

    def test_rerun_rejects_changed_clip_bytes(self):
        self.run_preparation()
        clip = Path(self.rows()[0]['audio'])
        with clip.open('ab') as handle:
            handle.write(b'altered')
        with self.assertRaisesRegex(ValueError, 'Existing timestamp clip changed'):
            self.run_preparation()

    def test_same_content_with_different_whitespace_and_unicode_composition_matches(self):
        import unicodedata
        self.change_raw(lambda r: r['source']['paragraphs'][0]['diarizedTranscript']['segments'][0].update(
            content=unicodedata.normalize('NFD', '첫   문장\nalpha.')))
        self.assertEqual(self.run_preparation()['accepted_paragraphs'], 6)


if __name__ == '__main__':
    unittest.main()
