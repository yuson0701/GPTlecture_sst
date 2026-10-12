import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('setup_whisper', Path(__file__).resolve().parents[1] / 'scripts/setup_whisper.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class WhisperSetupTests(unittest.TestCase):
    def test_install_preserves_qwen_and_rolls_back_on_gpu_failure(self):
        for fails in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original = 'STT_BACKEND=qwen-mlx\nQWEN_ASR_ADAPTER=/my/adapter\nPYTHON_BIN=/my/qwen/python\n'
                (root / '.env').write_text(original)
                with patch.object(setup, 'ROOT', root), patch.object(setup.platform, 'system', return_value='Darwin'), \
                     patch.object(setup.platform, 'machine', return_value='arm64'), \
                     patch.object(sys, 'argv', ['setup_whisper.py', '--skip-install']), \
                     patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(snapshot_download=lambda *a, **k: '/cached/whisper')}), \
                     patch.object(setup.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'load') if fails else None):
                    if fails:
                        with self.assertRaises(subprocess.CalledProcessError):
                            setup.main()
                        self.assertEqual((root / '.env').read_text(), original)
                    else:
                        setup.main()
                        saved = (root / '.env').read_text()
                        self.assertTrue(saved.startswith(original))
                        self.assertIn('WHISPER_CHOICE_BACKEND=mlx', saved)
                        self.assertIn('MLX_WHISPER_MODEL=/cached/whisper', saved)
                        self.assertEqual((root / '.env.before-whisper').read_text(), original)


if __name__ == '__main__':
    unittest.main()
