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
from qwen_asr import QwenRecognizer, validate_platform, model_path, adapter_path, status, DEFAULT_ADAPTER, ADAPTER_REVISION, BASE_REVISION
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
        model.fine_tuned = False
        model.model.generate.return_value = types.SimpleNamespace(text=' 기회비용입니다. ', generation_tokens=8)
        return model

    def test_fine_tuned_decode_uses_retry_policy_and_vocabulary(self):
        model = self.recognizer()
        model.fine_tuned = True
        model.model.generate.side_effect = [
            types.SimpleNamespace(text='반복 ' * 10, generation_tokens=30),
            types.SimpleNamespace(text='회복된 원문', generation_tokens=8)]
        result = model.transcribe(np.ones(16000), 'GDP')
        self.assertEqual(result['text'], '회복된 원문')
        self.assertTrue(result['fineTuned'])
        self.assertEqual(result['decodeRetries'], 1)
        self.assertEqual([c.kwargs['max_tokens'] for c in model.model.generate.call_args_list], [2048, 768])
        self.assertTrue(all(c.kwargs['hotwords'] == ['GDP'] for c in model.model.generate.call_args_list))

    def test_fine_tuned_unresolved_output_is_not_accepted(self):
        model = self.recognizer()
        model.fine_tuned = True
        model.model.generate.return_value = types.SimpleNamespace(text='반복 ' * 10, generation_tokens=30)
        with self.assertRaisesRegex(RuntimeError, '반복'):
            model.transcribe(np.ones(16000))

    def test_fine_tuned_no_glossary_preserves_evaluated_prompt(self):
        model = self.recognizer()
        model.fine_tuned = True
        self.assertTrue(model.transcribe(np.ones(16000))['fineTuned'])
        self.assertNotIn('hotwords', model.model.generate.call_args.kwargs)

    def test_adapter_cache_uses_pinned_revision_offline(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'QWEN_ASR_ADAPTER': DEFAULT_ADAPTER}, clear=True):
            root = Path(temp)
            for name in ['adapter_config.json', 'selected.safetensors']:
                (root / name).write_text('{}')
            with patch('huggingface_hub.snapshot_download', return_value=temp) as download:
                self.assertEqual(adapter_path(), root.resolve())
                download.assert_called_once_with(DEFAULT_ADAPTER, revision=ADAPTER_REVISION, local_files_only=True)
            (root / 'selected.safetensors').unlink()
            with patch('huggingface_hub.snapshot_download', return_value=temp):
                with self.assertRaisesRegex(RuntimeError, '미세조정'):
                    adapter_path()

    def test_base_model_requires_explicit_adapter_opt_out(self):
        with patch.dict(os.environ, {'QWEN_ASR_ADAPTER': 'none'}):
            self.assertIsNone(adapter_path())

    def test_invalid_adapter_status_does_not_claim_base_available(self):
        with patch('qwen_asr.validate_platform'), patch('qwen_asr.importlib.util.find_spec', return_value=True), \
                patch('qwen_asr.model_path'), patch('qwen_asr.adapter_path', side_effect=RuntimeError('missing')):
            self.assertEqual(status(), {'available': False, 'reason': 'model'})

    def test_default_setup_downloads_adapter_and_pins_base_before_setting_env(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(setup, 'ROOT', Path(temp)), \
                patch.object(setup, 'validate_platform'), \
                patch.object(setup.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run, \
                patch('huggingface_hub.snapshot_download', side_effect=['/cached/base', '/cached/adapter']) as download, \
                patch.object(sys, 'argv', ['setup_qwen.py', '--skip-install']):
            setup.main()
            self.assertEqual(download.call_args_list[0].kwargs['revision'], BASE_REVISION)
            self.assertEqual(download.call_args_list[1].args[0], DEFAULT_ADAPTER)
            self.assertEqual(download.call_args_list[1].kwargs['revision'], ADAPTER_REVISION)
            self.assertEqual(run.call_args.kwargs['env']['QWEN_ASR_ADAPTER'], '/cached/adapter')
            self.assertIn('QWEN_ASR_ADAPTER=/cached/adapter', (Path(temp) / '.env').read_text())

    def test_setup_load_failure_preserves_existing_env(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(setup, 'ROOT', Path(temp)), \
                patch.object(setup, 'validate_platform'), \
                patch.object(setup.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1)), \
                patch('huggingface_hub.snapshot_download', side_effect=['/cached/base', '/cached/adapter']), \
                patch.object(sys, 'argv', ['setup_qwen.py', '--skip-install']):
            env = Path(temp) / '.env'
            env.write_text('STT_BACKEND=sherpa\n')
            with self.assertRaises(SystemExit):
                setup.main()
            self.assertEqual(env.read_text(), 'STT_BACKEND=sherpa\n')

    def test_setup_public_download_failure_preserves_existing_env(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(setup, 'ROOT', Path(temp)), \
                patch.object(setup, 'validate_platform'), \
                patch('huggingface_hub.snapshot_download', side_effect=['/cached/base', RuntimeError('network unavailable')]), \
                patch.object(sys, 'argv', ['setup_qwen.py', '--skip-install']):
            env = Path(temp) / '.env'
            env.write_text('STT_BACKEND=sherpa\n')
            with self.assertRaisesRegex(SystemExit, 'No login is required'):
                setup.main()
            self.assertEqual(env.read_text(), 'STT_BACKEND=sherpa\n')

    def test_base_only_setup_removes_old_adapter_setting(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(setup, 'ROOT', Path(temp)), \
                patch.object(setup, 'validate_platform'), \
                patch.object(setup.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)), \
                patch('huggingface_hub.snapshot_download', return_value='/cached/base') as download, \
                patch.object(sys, 'argv', ['setup_qwen.py', '--skip-install', '--base-only', '--size', '0.6B']):
            (Path(temp) / '.env').write_text('QWEN_ASR_ADAPTER=old\n')
            setup.main()
            download.assert_called_once()
            self.assertIn('QWEN_ASR_ADAPTER=none', (Path(temp) / '.env').read_text())

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

    def test_split_qwen_tokenizer_without_tokenizer_json(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'QWEN_ASR_MODEL': temp}):
            root = Path(temp)
            for name in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'vocab.json', 'merges.txt', 'model.safetensors']:
                (root / name).write_text('{}')
            self.assertEqual(model_path(), root.resolve())
            (root / 'merges.txt').unlink()
            with self.assertRaisesRegex(RuntimeError, r'vocab.json \+ merges.txt'):
                model_path()
            (root / 'tokenizer.json').write_text('{}')
            (root / 'preprocessor_config.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'preprocessor_config.json'):
                model_path()

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
