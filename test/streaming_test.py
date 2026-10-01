import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('streaming', root / 'local' / 'streaming.py')
streaming = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streaming)


class StreamingTests(unittest.TestCase):
    def make_recognizer(self):
        stream = types.SimpleNamespace(accept_waveform=lambda *args: None, input_finished=lambda: None)
        recognizer = object.__new__(streaming.StreamingRecognizer)
        recognizer.recognizer = types.SimpleNamespace(create_stream=lambda: stream, get_result=lambda s: '한국어', is_ready=lambda s: False, is_endpoint=lambda s: False, reset=lambda s: None)
        recognizer.start()
        return recognizer

    def test_retry_does_not_decode_duplicate_audio(self):
        import numpy as np
        recognizer = self.make_recognizer()
        first = recognizer.accept(np.zeros(3200), recognizer.session, 0)
        second = recognizer.accept(np.zeros(3200), recognizer.session, 0)
        self.assertIs(first, second)
        self.assertEqual(recognizer.samples, 3200)
        with self.assertRaises(RuntimeError):
            recognizer.accept(np.zeros(3200), recognizer.session, 3)
        with self.assertRaises(RuntimeError):
            recognizer.accept(np.zeros(3200), 'other-session', 1)

    def test_endpoint_resets_decoder_without_throwing_away_lookahead(self):
        import numpy as np
        recognizer = self.make_recognizer()
        stream = recognizer.stream
        recognizer.recognizer.is_endpoint = lambda s: True
        first = recognizer.accept(np.zeros(3200), recognizer.session, 0)
        self.assertTrue(first['events'][0]['final'])
        self.assertIs(recognizer.stream, stream)
        second = recognizer.accept(np.zeros(3200), recognizer.session, 1)
        self.assertNotEqual(first['events'][0]['id'], second['events'][0]['id'])

    def test_final_is_idempotent_and_rejects_late_packets(self):
        import numpy as np
        recognizer = self.make_recognizer()
        result = recognizer.accept(np.zeros(0), recognizer.session, 0, True)
        self.assertTrue(result['events'][0]['final'])
        self.assertIs(result, recognizer.accept(np.zeros(0), recognizer.session, 0, True))
        with self.assertRaises(RuntimeError):
            recognizer.accept(np.zeros(3200), recognizer.session, 1)

    def test_decoder_failure_cannot_replay_samples_into_mutated_state(self):
        import numpy as np
        recognizer = self.make_recognizer()
        recognizer.recognizer.is_ready = lambda s: True
        def fail(s):
            raise RuntimeError('decode failed')
        recognizer.recognizer.decode_stream = fail
        with self.assertRaisesRegex(RuntimeError, 'decode failed'):
            recognizer.accept(np.zeros(3200), recognizer.session, 0)
        self.assertEqual(recognizer.samples, 3200)
        with self.assertRaisesRegex(RuntimeError, '복구'):
            recognizer.accept(np.zeros(3200), recognizer.session, 0)
        self.assertEqual(recognizer.samples, 3200)

    def test_silence_has_no_transcript_events(self):
        import numpy as np
        recognizer = self.make_recognizer()
        recognizer.recognizer.get_result = lambda s: ''
        self.assertEqual(recognizer.accept(np.zeros(3200), recognizer.session, 0)['events'], [])

    def test_setup_preserves_settings_and_creates_private_backup(self):
        spec = importlib.util.spec_from_file_location('setup_realtime', root / 'scripts' / 'setup_realtime.py')
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / '.env'
            original = 'WHISPER_BACKEND=faster-whisper\nPORT=3001\nSTT_BACKEND=mlx\n'
            env.write_text(original)
            setup.update_env(env)
            setup.update_env(env)
            self.assertEqual(env.read_text().count('STT_BACKEND=sherpa'), 1)
            self.assertIn('PORT=3001', env.read_text())
            self.assertEqual(env.with_name('.env.before-realtime').read_text(), original)


if __name__ == '__main__':
    unittest.main()
