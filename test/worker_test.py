"""Validate the local worker contract without downloading model weights."""
import base64
import importlib.util
from pathlib import Path
import sys
import os
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('worker', Path(__file__).resolve().parents[1] / 'local' / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'WHISPER_BACKEND': 'faster-whisper'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
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
        self.assertEqual(model.settings['beam_size'], 1)
        self.assertEqual(model.settings['temperature'], 0.0)

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

    def test_mlx_rejects_unsupported_platform(self):
        with patch.dict(os.environ, {'WHISPER_BACKEND': 'mlx'}), patch.object(worker.platform, 'system', return_value='Linux'):
            self.assertEqual(worker.dispatch({'action': 'status'})['reason'], 'platform')
            with self.assertRaisesRegex(RuntimeError, 'Apple Silicon'):
                worker.dispatch({'action': 'load'})

    def test_mlx_uses_cached_path_and_rejects_silence(self):
        calls = []
        def transcribe(samples, **options):
            calls.append(options)
            return {'text': ' 한국어 초안 '}
        vad = types.SimpleNamespace(get_speech_timestamps=lambda *args: [], VadOptions=lambda **kwargs: kwargs)
        with patch.dict(os.environ, {'WHISPER_BACKEND': 'mlx'}), patch.dict(sys.modules, {'mlx_whisper': types.SimpleNamespace(transcribe=transcribe), 'faster_whisper.vad': vad}):
            worker.model = '/local/cached/mlx-model'
            request = {'action': 'transcribe', 'audio': base64.b64encode(b'\x00\x40' * 16000).decode(), 'glossary': '한국어'}
            self.assertEqual(worker.dispatch(request)['text'], '')
            self.assertEqual(calls, [])
            vad.get_speech_timestamps = lambda *args: [{'start': 0, 'end': 16000}]
            self.assertEqual(worker.dispatch(request)['text'], '한국어 초안')
            self.assertEqual(calls[0]['path_or_hf_repo'], '/local/cached/mlx-model')
            self.assertEqual(calls[0]['language'], 'ko')
            self.assertEqual(calls[0]['temperature'], 0.0)


if __name__ == '__main__':
    unittest.main()
