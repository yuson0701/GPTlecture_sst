import importlib.util
import json
from pathlib import Path
import tempfile
import unicodedata
import unittest
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_qwen_dataset', ROOT / 'scripts/prepare_qwen_dataset.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    # Fictional titles only; real lecture metadata belongs in ignored data files.
    config = {'splits': {'1_샘플나 교수님': 'test', '1_샘플다 교수님': 'train', '2_샘플다 교수님': 'train'}}

    def source_pair(self, source, title='1_샘플나 교수님', transcript=None):
        # WAV content with M4A extension tests probing by actual container, not filename.
        audio = source / (unicodedata.normalize('NFD', title) + '.m4a')
        t = np.arange(48000) / 48000
        samples = (np.sin(2 * np.pi * 440 * t) * 9000).astype('<i2')
        with wave.open(str(audio), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(samples.tobytes())
        name = title.replace('_', '_ ', 1) + ' 스크립트.txt'
        script = source / name
        script.write_text(transcript or '---\ncreated: 2020-01-01T00:00:00Z\nsource: Tiro\n---\n\n# Title\n\nA: 연골 cartilage입니다. [기침]\n\nB: [박수]\n', encoding='utf-8')
        return audio, script

    def test_normalized_filename_pairing_excludes_summaries_and_rejects_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.source_pair(source)
            (source / '1_샘플나 교수님 요약.txt').write_text('Not a label')
            pairs, ignored = prep.discover_pairs(source, self.config)
            self.assertEqual(pairs[0]['lecture_id'], prep.lecture_id('1_샘플나 교수님'))
            self.assertEqual(pairs[0]['split'], 'test')
            self.assertEqual(len(ignored), 1)
            (source / '1_샘플나 교수님 스크립트.txt').write_text('A: duplicate')
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                prep.discover_pairs(source, self.config)

    def test_unpaired_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audio, _ = self.source_pair(Path(directory))
            audio.unlink()
            with self.assertRaisesRegex(ValueError, 'Unpaired'):
                prep.discover_pairs(directory, self.config)

    def test_cleaning_preserves_bilingual_spelling_and_offsets_without_fake_timestamps(self):
        text, paragraphs, events, metadata, unknown = prep.clean_transcript(
            '---\ncreated: 2020-01-01T00:00:00Z\n---\n# 제목\nA: 어, splanchnic mesoderm입니다. [기침]\n'
            '\nB: [박수]\n\nA: myoepithelium [Ca++] 그대로.\n')
        self.assertEqual(text, '어, splanchnic mesoderm입니다.\n\nmyoepithelium [Ca++] 그대로.')
        self.assertEqual(len(events), 2)
        self.assertEqual(len(unknown), 1)
        self.assertEqual(metadata['created'], '2020-01-01T00:00:00Z')
        for paragraph in paragraphs:
            self.assertEqual(text[paragraph['char_start']:paragraph['char_end']], paragraph['text'])
            self.assertNotIn('start_seconds', paragraph)

    def test_full_decode_split_lock_and_no_training_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            audio, script = self.source_pair(source)
            output = root / 'prepared'
            (output / 'model_cache').mkdir(parents=True)
            (output / 'model_cache' / 'fixture').write_text('preserved model')
            config_path = output / 'source_config.json'
            config_path.write_text(json.dumps(self.config))
            report = prep.prepare(source, output, config_path)
            row = json.loads((output / 'lectures.jsonl').read_text())
            self.assertEqual(report['training_example_count'], 0)
            self.assertEqual(row['alignment_status'], 'pending')
            self.assertFalse(row['eligible_for_training'])
            self.assertEqual(row['sample_rate'], 16000)
            self.assertAlmostEqual(row['duration_seconds'], 1, places=3)
            self.assertFalse((output / 'train.jsonl').exists())
            self.assertEqual((output / 'model_cache' / 'fixture').read_text(), 'preserved model')
            self.assertEqual(json.loads(config_path.read_text()), self.config)
            self.assertEqual(row['source_audio_sha256'], prep.sha256(audio))
            self.assertEqual(prep.prepare(source, output), report)
            script.write_text('A: changed label')
            with self.assertRaisesRegex(ValueError, 'split lock'):
                prep.prepare(source, output)

    def test_unknown_output_and_overlap_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            self.source_pair(source)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prep.prepare(source, source / 'output', self.config)
            output = Path(directory) / 'output'
            output.mkdir()
            marker = output / 'user-data.txt'
            marker.write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'split lock'):
                prep.prepare(source, output, self.config)
            self.assertEqual(marker.read_text(), 'preserve')

    def test_silent_audio_rejected_and_staging_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            audio, _ = self.source_pair(source)
            with wave.open(str(audio), 'wb') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(bytes(32000))
            output = Path(directory) / 'output'
            with self.assertRaisesRegex(ValueError, 'silent'):
                prep.prepare(source, output, self.config)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).glob('.output.preparing-*')), [])

    def test_same_named_lecturer_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.source_pair(source, '1_샘플다 교수님')
            self.source_pair(source, '2_샘플다 교수님')
            config = {'splits': {**self.config['splits'], '2_샘플다 교수님': 'validation'}}
            with self.assertRaisesRegex(ValueError, 'crosses splits'):
                prep.discover_pairs(source, config)

    def test_initial_config_required_and_existing_split_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            self.source_pair(source)
            output = Path(directory) / 'output'
            with self.assertRaisesRegex(ValueError, 'explicit private --split-config'):
                prep.prepare(source, output)
            prep.prepare(source, output, self.config)
            frozen = (output / 'split_lock.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'split lock'):
                prep.prepare(source, output, {'splits': {'1_샘플나 교수님': 'train'}})
            self.assertEqual((output / 'split_lock.json').read_bytes(), frozen)

    def test_rerun_rejects_tampered_text_and_paragraph_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            self.source_pair(source)
            output = Path(directory) / 'output'
            prep.prepare(source, output, self.config)
            manifest = output / 'lectures.jsonl'
            row = json.loads(manifest.read_text())
            row['paragraphs'][0]['char_end'] += 1
            manifest.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'text or paragraph boundaries'):
                prep.prepare(source, output)


if __name__ == '__main__':
    unittest.main()
