"""Synthetic decoding-policy tests; no model, references, or GPU are accessed."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qwen_decode import DEFAULT_POLICY, decode_audio, policy_proof, repeated_ngram
from evaluate_qwen_lora import score_decoded


class MockModel:
    def __init__(self, generate):
        self.respond = generate
        self.calls = []

    def generate(self, audio, **kwargs):
        self.calls.append((audio.copy(), kwargs))
        return self.respond(len(self.calls), audio, kwargs)


def output(text='정상 발화입니다.', tokens=12):
    return SimpleNamespace(text=text, generation_tokens=tokens)


class BoundedDecodeTests(unittest.TestCase):
    def test_unicode_word_and_phrase_repetitions_are_detector_only(self):
        for text in ['어, ' * 10, 'YES! yes, ' * 5, '가 나 다! ' * 10, 'ＣＡＦÉ, café! ' * 5]:
            with self.subTest(text=text):
                self.assertIsNotNone(repeated_ngram(text))
        self.assertIsNone(repeated_ngram('어, ' * 9))
        self.assertIsNone(repeated_ngram('여기에서 이 단어를 다시 말하지만 뒤에는 다른 내용이 있습니다.'))

    def test_successful_full_prediction_is_unchanged(self):
        original = '  정상, Korean & English.\n'
        model = MockModel(lambda *_: output(original))
        audio = np.arange(16000, dtype=np.float32)
        result = decode_audio(model, audio)
        self.assertEqual(result['text'], original)
        self.assertEqual(len(model.calls), 1)
        np.testing.assert_array_equal(model.calls[0][0], audio)
        self.assertEqual(result['retries'], 0)
        self.assertFalse(result['unresolved'])
        self.assertEqual(result['policy'], policy_proof())

    def test_capped_full_prediction_is_discarded_and_all_audio_is_covered(self):
        model = MockModel(lambda call, audio, kw: output('FAILED FULL', 2048) if call == 1 else output(f'leaf {call}', 21))
        audio = np.arange(31 * 16000 + 7, dtype=np.float32)
        result = decode_audio(model, audio)
        self.assertNotIn('FAILED FULL', result['text'])
        self.assertEqual(result['initial_text'], 'FAILED FULL')
        self.assertEqual(result['initial_problem'], ['generation_token_limit'])
        self.assertEqual(result['generation_tokens'], 63)
        self.assertEqual(result['retries'], 3)
        self.assertFalse(result['unresolved'])
        np.testing.assert_array_equal(np.concatenate([call[0] for call in model.calls[1:]]), audio)
        self.assertTrue(all(len(call[0]) <= 15 * 16000 for call in model.calls[1:]))
        self.assertTrue(all(call[1]['max_tokens'] == 768 for call in model.calls[1:]))

    def test_repetition_alone_triggers_retry_without_cleaning_leaf_text(self):
        model = MockModel(lambda call, *_: output('어, ' * 10, 30) if call == 1 else output('  원래 띄어쓰기!  ', 15))
        result = decode_audio(model, np.zeros(10 * 16000, dtype=np.float32))
        self.assertEqual(result['initial_problem'], ['consecutive_repetition'])
        self.assertEqual(result['text'], '  원래 띄어쓰기!  ')
        self.assertEqual(result['retries'], 1)

    def test_recursive_caps_are_finite_and_unresolved_predictions_are_retained(self):
        model = MockModel(lambda call, audio, kw: output(f'prediction {call}', kw['max_tokens']))
        audio = np.zeros(90 * 16000, dtype=np.float32)
        result = decode_audio(model, audio)
        self.assertEqual(len(model.calls), 43)
        self.assertEqual(len(result['leaf_intervals']), 24)
        self.assertEqual(result['generation_tokens'], 24 * 768)
        self.assertEqual(result['attempt_generation_tokens'], 2048 + 42 * 768)
        self.assertTrue(result['unresolved'])
        self.assertEqual(len(result['unresolved_leaf_attempt_ids']), 24)
        cursor = 0
        for leaf in result['leaf_intervals']:
            self.assertEqual(leaf['start_sample'], cursor)
            self.assertGreaterEqual(leaf['end_sample'] - cursor, 3 * 16000)
            self.assertLessEqual(leaf['depth'], DEFAULT_POLICY.max_depth)
            self.assertIn(leaf['text'], result['text'])
            cursor = leaf['end_sample']
        self.assertEqual(cursor, len(audio))

    def test_depth_bound_is_enforced_independently(self):
        model = MockModel(lambda call, audio, kw: output('capped', kw['max_tokens']))
        result = decode_audio(model, np.zeros(12 * 16000, dtype=np.float32), policy=replace(DEFAULT_POLICY, max_depth=0))
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(result['unresolved'])

    def test_final_aggregate_token_count_is_not_a_truncation_flag(self):
        model = MockModel(lambda call, audio, kw: output('capped', 2048) if call == 1 else output('normal chunk', 700))
        result = decode_audio(model, np.zeros(60 * 16000, dtype=np.float32))
        self.assertGreater(result['generation_tokens'], 2048)
        self.assertFalse(result['unresolved'])
        row = {'transcript': ' '.join(['normal chunk'] * 4)}
        scored = score_decoded(row, result)
        self.assertFalse(scored['truncated'])
        self.assertTrue(scored['initial_token_limit'])
        self.assertEqual(scored['normalized_cer'], 0)
        self.assertGreater(scored['initial_normalized_cer'], 0)
        self.assertEqual(scored['transcript'], row['transcript'])

    def test_evaluator_keeps_unresolved_leaf_even_below_full_token_limit(self):
        model = MockModel(lambda call, audio, kw: output('unresolved prediction', kw['max_tokens']))
        decoded = decode_audio(model, np.zeros(16000, dtype=np.float32))
        scored = score_decoded({'transcript': 'original reference'}, decoded)
        self.assertEqual(scored['generation_tokens'], 768)
        self.assertTrue(scored['truncated'])
        self.assertTrue(scored['unresolved'])
        self.assertEqual(scored['prediction'], 'unresolved prediction')
        self.assertGreater(scored['edits'], 0)

    def test_evaluator_rejects_unproven_policy_changes(self):
        decoded = decode_audio(MockModel(lambda *_: output()), np.zeros(16000, dtype=np.float32),
                               policy=replace(DEFAULT_POLICY, full_max_tokens=512))
        with self.assertRaisesRegex(ValueError, 'policy'):
            score_decoded({'transcript': 'reference'}, decoded)

    def test_short_audio_is_never_dropped(self):
        model = MockModel(lambda call, audio, kw: output('capped', kw['max_tokens']))
        result = decode_audio(model, np.zeros(17, dtype=np.float32))
        self.assertEqual(result['leaf_intervals'][0]['end_sample'], 17)
        self.assertEqual(len(model.calls), 2)

    def test_policy_is_identical_for_two_independent_models(self):
        waveform = np.zeros(30 * 16000, dtype=np.float32)
        outputs = []
        for _ in range(2):
            model = MockModel(lambda call, audio, kw: output('cap', 2048) if call == 1 else output('good', 20))
            outputs.append(decode_audio(model, waveform))
        self.assertEqual(outputs[0], outputs[1])

    def test_invalid_waveforms_and_model_outputs_fail_explicitly(self):
        model = MockModel(lambda *_: output())
        for audio in [np.array([]), np.array([float('nan')]), np.zeros((2, 3))]:
            with self.assertRaises(ValueError):
                decode_audio(model, audio)
        with self.assertRaises(ValueError):
            decode_audio(MockModel(lambda *_: output(tokens=-1)), np.zeros(16000))


if __name__ == '__main__':
    unittest.main()
