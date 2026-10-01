"""Validate the local worker contract without downloading model weights."""
import base64
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('worker', Path(__file__).resolve().parents[1] / 'local' / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def tearDown(self):
        worker.model = None

    def test_pcm_and_korean_settings(self):
        class Model:
            def transcribe(self, samples, **kwargs):
                self.samples = samples
                self.settings = kwargs
                return iter([types.SimpleNamespace(text=' 기회비용입니다. ')]), None
        model = Model()
        worker.model = model
        result = worker.dispatch({'action': 'transcribe', 'audio': base64.b64encode(b'\x00\x40' * 16000).decode(), 'glossary': '기회비용'})
        self.assertEqual(result['text'], '기회비용입니다.')
        self.assertEqual(len(model.samples), 16000)
        self.assertEqual(model.samples[0], 0.5)
        self.assertEqual(model.settings['language'], 'ko')
        self.assertEqual(model.settings['initial_prompt'], '기회비용')
        self.assertTrue(model.settings['vad_filter'])

    def test_model_is_loaded_offline_and_reused(self):
        calls = []
        def constructor(name, **kwargs):
            calls.append((name, kwargs))
            return object()
        with patch.dict(sys.modules, {'faster_whisper': types.SimpleNamespace(WhisperModel=constructor)}), patch.object(worker, 'model_path', return_value='/cached/model'):
            worker.dispatch({'action': 'load'})
            worker.dispatch({'action': 'load'})
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]['local_files_only'])

    def test_bad_audio_is_rejected(self):
        worker.model = object()
        with self.assertRaises(ValueError):
            worker.dispatch({'action': 'transcribe', 'audio': base64.b64encode(b'\x00').decode()})


if __name__ == '__main__':
    unittest.main()
