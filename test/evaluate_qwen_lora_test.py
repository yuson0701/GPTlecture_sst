"""Synthetic CPU-only tests; no held-out source text or models are loaded."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_qwen_lora as evaluation


class HeldOutEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / 'lecture_0123456789ab_paragraph_1.wav'
        with wave.open(str(self.audio), 'wb') as handle:
            handle.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            handle.writeframes(b'\x01\x00' * 1600)
        self.test_file = self.root / 'test.jsonl'
        self.original = ' 한글 GDP, 원문! '
        self.source = {'audio': self.audio.name, 'text': evaluation.training.PREFIX + self.original}
        self.write_source(self.source)
        self.provenance = {split: [{'lecture_id': 'lecture_' + letter * 12,
                                    'audio': str(self.root / (split + '.wav')),
                                    'audio_sha256': letter * 64, 'pcm_sha256': letter * 64}]
                           for split, letter in [('train', 'a'), ('validation', 'b')]}

    def write_source(self, source):
        self.test_file.write_text(json.dumps(source, ensure_ascii=False) + '\n', encoding='utf-8')

    def test_test_loader_preserves_original_label_and_resolves_audio(self):
        rows = evaluation.read_test_manifest(self.test_file)
        self.assertEqual(rows[0]['transcript'], self.original)
        self.assertEqual(rows[0]['audio'], str(self.audio.resolve()))
        self.assertEqual(rows[0]['lecture_id'], 'lecture_0123456789ab')
        evaluation.validate_held_out(rows, self.provenance)

    def test_each_leakage_key_is_checked_against_both_partitions(self):
        row = evaluation.read_test_manifest(self.test_file)[0]
        for split in ('train', 'validation'):
            for field, other in [('lecture_id', 'lecture_id'), ('audio', 'audio'),
                                 ('file_sha256', 'audio_sha256'), ('pcm_sha256', 'pcm_sha256')]:
                provenance = json.loads(json.dumps(self.provenance))
                provenance[split][0][other] = row[field]
                with self.subTest(split=split, field=field), self.assertRaisesRegex(ValueError, 'overlap'):
                    evaluation.validate_held_out([row], provenance)

    def test_inconsistent_or_non_test_metadata_is_rejected(self):
        for extra in ({'split': 'train'}, {'lecture_id': 'different_lecture'}, {'text': 'unprefixed'},
                      {'text': evaluation.training.PREFIX + '...'},
                      {'text': evaluation.training.PREFIX + '원문<|im_end|>'}):
            self.write_source({**self.source, **extra})
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                evaluation.read_test_manifest(self.test_file)

    def test_changed_audio_fails_snapshot_guard(self):
        files = {str(self.audio): evaluation.sha256(self.audio)}
        evaluation.assert_files_unchanged(files)
        self.audio.write_bytes(self.audio.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            evaluation.assert_files_unchanged(files)

    def test_truncated_prediction_is_scored_and_never_censored(self):
        row = evaluation.read_test_manifest(self.test_file)[0]
        result = evaluation.score_prediction(row, '한글', evaluation.MAX_TOKENS)
        self.assertTrue(result['truncated'])
        self.assertGreater(result['edits'], 0)
        self.assertGreater(result['normalized_cer'], 0)
        self.assertEqual(result['transcript'], self.original)
        same = evaluation.score_prediction(row, '한글 gdp 원문', 10)
        self.assertEqual(same['normalized_cer'], 0)

    def test_existing_output_is_never_reused_even_if_empty(self):
        output = self.root / 'existing'; output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            evaluation.require_fresh_output(output, self.root / 'run', self.root / 'model')

    def make_completed_run(self):
        run, model = self.root / 'run', self.root / 'model'
        run.mkdir(); model.mkdir()
        (model / 'config.json').write_text('{}')
        (run / 'selected.safetensors').write_bytes(b'synthetic adapter, never loaded')
        (run / 'input_provenance.json').write_text(json.dumps(self.provenance))
        config = {'base_model': str(model.resolve()), 'base_config_sha256': evaluation.sha256(model / 'config.json'),
                  'input_provenance_sha256': evaluation.sha256(run / 'input_provenance.json')}
        report = {'training_completed': True, 'base_unchanged': True, 'input_audio_unchanged': True,
                  'train_examples': 1, 'validation_examples': 1,
                  'input_provenance_sha256': config['input_provenance_sha256'],
                  'selected_adapter_sha256': evaluation.sha256(run / 'selected.safetensors')}
        (run / 'adapter_config.json').write_text(json.dumps(config))
        (run / 'training_report.json').write_text(json.dumps(report))
        return run, model, report

    def test_completed_run_requires_report_bound_provenance(self):
        run, model, _ = self.make_completed_run()
        evaluation.read_completed_run(run, model)
        (run / 'input_provenance.json').write_text(json.dumps({**self.provenance, 'extra': []}))
        with self.assertRaisesRegex(ValueError, 'provenance hash mismatch'):
            evaluation.read_completed_run(run, model)

    def test_incomplete_training_cannot_be_evaluated(self):
        run, model, report = self.make_completed_run()
        report['training_completed'] = False
        (run / 'training_report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'has not completed'):
            evaluation.read_completed_run(run, model)

    def test_changed_selected_checkpoint_cannot_be_evaluated(self):
        run, model, _ = self.make_completed_run()
        (run / 'selected.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Selected adapter changed'):
            evaluation.read_completed_run(run, model)


if __name__ == '__main__':
    unittest.main()
