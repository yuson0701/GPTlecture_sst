import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local'))
from qwen_asr import QwenRecognizer, validate_platform, model_path, status
spec = importlib.util.spec_from_file_location('setup_qwen', ROOT / 'scripts/setup_qwen.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class QwenTests(unittest.TestCase):
    def recognizer(self):
        model = QwenRecognizer.__new__(QwenRecognizer)
        model.mx = Mock()
        model.vad = Mock(return_value=[{'start': 0, 'end': 16000}])
        model.vad_options = object()
        model.model = Mock()
        model.model.generate.return_value = types.SimpleNamespace(text=' 기회비용입니다. ', generation_tokens=8)
        return model

    def test_korean_hotwords_and_bounded_decode(self):
        model = self.recognizer()
        result = model.transcribe(np.ones(16000, dtype=np.float32), '기회비용, 한계효용\nGDP')
        args = model.model.generate.call_args.kwargs
        self.assertEqual(args['language'], 'Korean')
        self.assertEqual(args['hotwords'], ['기회비용', '한계효용', 'GDP'])
        self.assertEqual(args['max_tokens'], 256)
        self.assertEqual(result['text'], '기회비용입니다.')
        model.mx.clear_cache.assert_called_once()

    def test_silence_never_generates_text(self):
        model = self.recognizer()
        model.vad.return_value = []
        self.assertEqual(model.transcribe(np.zeros(16000))['text'], '')
        model.model.generate.assert_not_called()

    def test_failed_decode_releases_transient_memory(self):
        model = self.recognizer()
        model.model.generate.side_effect = RuntimeError('decode error')
        with self.assertRaises(RuntimeError):
            model.transcribe(np.ones(16000))
        model.mx.clear_cache.assert_called_once()

    def test_no_silent_truncated_transcript(self):
        model = self.recognizer()
        model.model.generate.return_value.generation_tokens = 256
        with self.assertRaises(RuntimeError):
            model.transcribe(np.ones(16000))
        with self.assertRaises(ValueError):
            model.transcribe(np.ones(16000 * 9))

    def test_non_mac_does_not_fall_back_to_cpu(self):
        with patch('qwen_asr.platform.system', return_value='Linux'):
            with self.assertRaises(RuntimeError):
                validate_platform()
            self.assertEqual(status()['reason'], 'platform')

    def test_model_checks_all_shards_locally(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'QWEN_ASR_MODEL': temp}):
            root = Path(temp)
            for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json']:
                (root / name).write_text('{}')
            (root / 'model-1.safetensors').write_bytes(b'fixture')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'model-1.safetensors', 'b': 'model-2.safetensors'}}))
            with self.assertRaises(RuntimeError):
                model_path()
            (root / 'model-2.safetensors').write_bytes(b'fixture')
            self.assertEqual(model_path(), root.resolve())

    def test_setup_preserves_prior_settings_and_backup(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '.env'
            original = 'STT_BACKEND=sherpa\nPYTHON_BIN=old\nOLLAMA_MODEL=my-model\n'
            path.write_text(original)
            settings = {'STT_BACKEND': 'qwen-mlx', 'PYTHON_BIN': '/path with spaces/python'}
            setup.update_env(path, settings)
            setup.update_env(path, settings)
            self.assertEqual((path.parent / '.env.before-qwen').read_text(), original)
            self.assertIn('OLLAMA_MODEL=my-model', path.read_text())
            self.assertEqual(path.read_text().count('STT_BACKEND='), 1)


if __name__ == '__main__':
    unittest.main()
