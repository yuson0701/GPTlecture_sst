"""CPU-only checks for the LoRA trainer's data and causal-label boundaries."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_qwen_lora as trainer


def write_wav(path, frames=1600, sample=1, rate=16000, channels=1):
    with wave.open(str(path), 'wb') as handle:
        handle.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
        handle.writeframes(struct.pack('<h', sample) * frames * channels)


class CausalTargetTests(unittest.TestCase):
    def test_first_word_and_eos_are_predicted_from_their_predecessors(self):
        prompt = ['system', 'audio', 'assistant', '<asr_text>']
        targets = ['첫', '단어', '.', '<|im_end|>']
        sequence = prompt + targets[:-1]
        length, start, stop = trainer.target_positions(len(prompt), len(targets))
        self.assertEqual(len(sequence), length)
        self.assertNotIn('<|im_end|>', sequence)
        predicted_from = sequence[start:stop]
        self.assertEqual(list(zip(predicted_from, targets)), [
            ('<asr_text>', '첫'), ('첫', '단어'), ('단어', '.'), ('.', '<|im_end|>')])

    def test_single_eos_target_uses_last_prompt_position(self):
        length, start, stop = trainer.target_positions(5, 1)
        self.assertEqual(list(range(length))[start:stop], [4])

    def test_invalid_lengths_fail_instead_of_creating_an_empty_loss(self):
        for prompt, target in [(0, 2), (3, 0), (-1, 1), (1, -1), (True, 2), (2, 1.5)]:
            with self.subTest(prompt=prompt, target=target), self.assertRaises(ValueError):
                trainer.target_positions(prompt, target)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.train_audio = self.root / 'lecture_0123456789ab_paragraph_0001.wav'
        self.val_audio = self.root / 'lecture_fedcba987654_paragraph_0001.wav'
        write_wav(self.train_audio, sample=10)
        write_wav(self.val_audio, sample=20)
        self.original = ' 원문과 GDP, 한글 punctuation! '

    def manifest(self, name='train.jsonl', audio=None, **extra):
        row = {'audio': (audio or self.train_audio).name,
               'text': trainer.PREFIX + self.original, **extra}
        path = self.root / name
        path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
        return path

    def split_rows(self):
        train = trainer.read_manifest(self.manifest())
        validation = trainer.read_manifest(self.manifest('validation.jsonl', self.val_audio))
        return train, validation

    def test_relative_audio_and_original_text_are_preserved(self):
        row = trainer.read_manifest(self.manifest())[0]
        self.assertEqual(row['audio'], str(self.train_audio.resolve()))
        self.assertEqual(row['transcript'], self.original)
        self.assertEqual(row['lecture_id'], 'lecture_0123456789ab')
        self.assertEqual(row['duration_seconds'], 0.1)

    def test_separate_lectures_and_audio_pass(self):
        trainer.validate_splits(*self.split_rows())

    def test_missing_lecture_identity_fails(self):
        train, validation = self.split_rows()
        for missing in (None, '', 123):
            with self.subTest(identity=missing), self.assertRaisesRegex(ValueError, 'Lecture identity'):
                trainer.validate_splits([{**train[0], 'lecture_id': missing}], validation)

    def test_different_clips_from_same_lecture_cannot_cross_splits(self):
        train, validation = self.split_rows()
        validation[0]['lecture_id'] = train[0]['lecture_id']
        with self.assertRaisesRegex(ValueError, 'lecture_id overlap'):
            trainer.validate_splits(train, validation)

    def test_identical_pcm_in_different_wav_containers_cannot_cross_splits(self):
        write_wav(self.val_audio, sample=10)
        contents = self.val_audio.read_bytes()
        contents += b'JUNK' + struct.pack('<I', 4) + b'test'
        contents = contents[:4] + struct.pack('<I', len(contents) - 8) + contents[8:]
        self.val_audio.write_bytes(contents)
        train, validation = self.split_rows()
        self.assertNotEqual(train[0]['file_sha256'], validation[0]['file_sha256'])
        self.assertEqual(train[0]['pcm_sha256'], validation[0]['pcm_sha256'])
        with self.assertRaisesRegex(ValueError, 'pcm_sha256 overlap'):
            trainer.validate_splits(train, validation)

    def test_same_file_cannot_cross_splits_under_different_identity(self):
        train, validation = self.split_rows()
        validation[0]['audio'] = train[0]['audio']
        with self.assertRaisesRegex(ValueError, 'audio overlap'):
            trainer.validate_splits(train, validation)

    def test_test_manifest_names_are_rejected_before_reading_data(self):
        for name in ('test.jsonl', 'TEST.JSONL', 'held_out_test.jsonl'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Test data'):
                trainer.read_manifest(self.root / name)

    def test_test_record_is_rejected_in_renamed_manifest(self):
        with self.assertRaisesRegex(ValueError, 'Test record'):
            trainer.read_manifest(self.manifest(split='test'))

    def test_duplicate_pcm_within_manifest_is_rejected(self):
        write_wav(self.val_audio, sample=10)
        path = self.manifest()
        row = json.loads(path.read_text())
        path.write_text(json.dumps(row) + '\n' + json.dumps({**row, 'audio': self.val_audio.name}) + '\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            trainer.read_manifest(path)


class AudioLimitTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audio = Path(directory.name) / 'synthetic.wav'

    def test_exactly_ninety_seconds_is_supported(self):
        self.assertEqual(trainer.MAX_AUDIO_SECONDS, 90)
        write_wav(self.audio, frames=90 * 16000)
        self.assertEqual(len(trainer.read_pcm(self.audio)), 90 * 32000)

    def test_one_frame_over_ninety_seconds_is_rejected(self):
        write_wav(self.audio, frames=90 * 16000 + 1)
        with self.assertRaisesRegex(ValueError, '90 seconds'):
            trainer.read_pcm(self.audio)

    def test_empty_audio_is_rejected(self):
        write_wav(self.audio, frames=0)
        with self.assertRaisesRegex(ValueError, 'between 0'):
            trainer.read_pcm(self.audio)

    def test_stereo_and_wrong_sample_rate_are_rejected(self):
        for arguments in ({'channels': 2}, {'rate': 8000}):
            write_wav(self.audio, **arguments)
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, 'mono 16 kHz'):
                trainer.read_pcm(self.audio)

    def test_truncated_pcm_is_rejected(self):
        write_wav(self.audio)
        self.audio.write_bytes(self.audio.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, 'Truncated WAV'):
            trainer.read_pcm(self.audio)


class TrainingArgumentTests(unittest.TestCase):
    def invoke(self, *extra):
        argv = ['train_qwen_lora.py', '--model', '/synthetic/model', '--output', '/synthetic/run',
                '--train-file', '/synthetic/train.jsonl', '--validation-file', '/synthetic/validation.jsonl', *extra]
        with mock.patch.object(sys, 'argv', argv):
            trainer.main()

    def test_finite_step_and_token_limits_reach_training_unchanged(self):
        with mock.patch.object(trainer, 'train') as train:
            self.invoke('--epochs', '1', '--max-steps', '2')
        args = train.call_args.args[0]
        self.assertEqual((args.epochs, args.max_steps), (1, 2))
        self.assertEqual((args.max_target_tokens, args.generation_max_tokens), (2048, 2048))

    def test_invalid_training_limits_fail_before_runtime(self):
        invalid = [('--epochs', '0'), ('--rank', '0'), ('--layers', '0'), ('--max-steps', '-1'),
                   ('--max-target-tokens', '0'), ('--generation-max-tokens', '0'),
                   ('--learning-rate', '0'), ('--learning-rate', 'nan'), ('--learning-rate', 'inf')]
        with mock.patch.object(trainer, 'train') as train:
            for option, value in invalid:
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, 'Invalid training limits'):
                    self.invoke(option, value)
        train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
