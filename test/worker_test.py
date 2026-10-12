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

    def test_mixed_language_whisper_is_transcription_not_translation(self):
        for language, expected in [('auto', None), ('ko', 'ko'), ('en', 'en')]:
            calls = []
            def transcribe(samples, **options):
                calls.append(options)
                return iter([types.SimpleNamespace(text='ECG 활동전위')]), None
            worker.model = types.SimpleNamespace(transcribe=transcribe)
            result = worker.dispatch({'action': 'transcribe', 'language': language,
                                     'audio': base64.b64encode(b'\x00\x40' * 16000).decode(), 'glossary': 'ECG'})
            self.assertEqual(result['text'], 'ECG 활동전위')
            self.assertEqual(calls[0]['language'], expected)
            self.assertEqual(calls[0]['task'], 'transcribe')
            self.assertEqual(calls[0]['initial_prompt'], 'ECG')

    def test_bad_audio_is_rejected(self):
        worker.model = object()
        with self.assertRaises(ValueError):
            worker.dispatch({'action': 'transcribe', 'audio': base64.b64encode(b'\x00').decode()})

    def test_medical_phrase_loops_but_not_normal_repeated_terms(self):
        for phrase in ['칼슘이 높은 ', '칼슘, 센스, 트로포닌, ', '수축조절, 심근, cardiomyocyte, ']:
            self.assertTrue(worker.repetition_loop('앞 문장입니다. ' + phrase * 15 + '다음 설명입니다.'))
            self.assertFalse(worker.repetition_loop(phrase * 2))
        self.assertFalse(worker.repetition_loop('칼슘이 증가합니다. 칼슘 채널이 열립니다. 칼슘을 SR에 저장합니다. 칼슘이 감소합니다.'))
        self.assertTrue(worker.repetition_loop('calcium, HIGH calcium high Calcium HIGH calcium high'))

    def test_whisper_final_retries_without_prompt_and_recovers(self):
        calls = []
        def transcribe(samples, **options):
            calls.append(options)
            text = '칼슘이 높은 ' * 30 if len(calls) == 1 else '이완도 영향을 받습니다.'
            return iter([types.SimpleNamespace(text=text, tokens=[1, 2])]), None
        worker.model = types.SimpleNamespace(transcribe=transcribe)
        result = worker.dispatch({'action': 'transcribe', 'language': 'auto', 'final': True,
                                 'audio': base64.b64encode(b'\x00\x40' * 16000).decode(), 'glossary': '칼슘, SR'})
        self.assertEqual(result['text'], '이완도 영향을 받습니다.')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['initial_prompt'], '칼슘, SR')
        self.assertIsNone(calls[1]['initial_prompt'])
        self.assertEqual(calls[1]['beam_size'], 3)
        self.assertTrue(all(c['max_new_tokens'] == 216 and c['condition_on_previous_text'] is False for c in calls))

    def test_failed_repetition_is_bounded_and_drafts_do_not_retry(self):
        for final, count in [(False, 1), (True, 2)]:
            calls = []
            def decode(retry):
                calls.append(retry)
                return '수축조절, 심근, cardiomyocyte, ' * 16, []
            with self.assertRaisesRegex(RuntimeError, 'Whisper 반복'):
                worker.checked_whisper(decode, final)
            self.assertEqual(len(calls), count)

    def test_token_limit_does_not_silently_publish_truncated_text(self):
        calls = []
        def decode(retry):
            calls.append(retry)
            return '겉으로는 반복이 없는 미완성 문장', [{'tokens': [1] * 216}]
        with self.assertRaises(RuntimeError):
            worker.checked_whisper(decode)
        self.assertEqual(calls, [False, True])

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

    def test_mlx_retries_loop_with_bounded_sampling_and_preserves_language(self):
        calls = []
        def transcribe(samples, **options):
            calls.append(options)
            return {'text': '칼슘, 센스, 트로포닌, ' * 15 if len(calls) == 1 else 'Troponin C에 칼슘이 결합합니다.'}
        vad = types.SimpleNamespace(get_speech_timestamps=lambda *args: [{'start': 0, 'end': 16000}], VadOptions=lambda **kw: kw)
        with patch.dict(os.environ, {'WHISPER_BACKEND': 'mlx'}), patch.dict(sys.modules, {'mlx_whisper': types.SimpleNamespace(transcribe=transcribe), 'faster_whisper.vad': vad}):
            worker.model = '/local/cached/mlx-model'
            result = worker.dispatch({'action': 'transcribe', 'language': 'auto', 'final': True,
                                      'audio': base64.b64encode(b'\x00\x40' * 16000).decode(), 'glossary': 'Troponin C'})
        self.assertEqual(result['text'], 'Troponin C에 칼슘이 결합합니다.')
        self.assertEqual([c['temperature'] for c in calls], [0.0, 0.2])
        self.assertIsNone(calls[-1]['initial_prompt'])
        self.assertTrue(all(c['language'] is None and c['sample_len'] == 216 for c in calls))


if __name__ == '__main__':
    unittest.main()
