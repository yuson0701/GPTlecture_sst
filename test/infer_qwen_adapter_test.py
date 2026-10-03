import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import infer_qwen_adapter as inference


class PortableAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'relocated-base'
        self.base.mkdir()
        (self.base / 'config.json').write_text('{}')
        self.adapter = self.root / 'public-adapter'
        self.adapter.mkdir()
        (self.adapter / 'selected.safetensors').write_bytes(b'test adapter')
        self.config = {'format': 'qwen3-asr-mlx-lora-v1',
                       'generation_policy': inference.policy_proof(),
                       'base_config_sha256': inference.training.sha256(self.base / 'config.json'),
                       'adapter_sha256': inference.training.sha256(self.adapter / 'selected.safetensors'),
                       'base_frozen_sha256': 'verified-base', 'rank': 8, 'layers': 8, 'alpha': 16}
        self.save()

    def save(self):
        (self.adapter / 'adapter_config.json').write_text(json.dumps(self.config))

    def test_relocated_base_needs_no_private_training_artifacts(self):
        model = Mock()
        with patch.object(inference.training, 'runtime'), \
                patch.object(inference.training, 'load_base', return_value=model), \
                patch.object(inference.training, 'inject_lora'), \
                patch.object(inference.training, 'frozen_fingerprint', return_value='verified-base'), \
                patch.object(inference.training, 'restore_adapter') as restore:
            self.assertIs(inference.load_portable_adapter(self.base, self.adapter), model)
            restore.assert_called_once_with(model, self.adapter / 'selected.safetensors')

    def test_corrupt_adapter_rejected_before_runtime(self):
        (self.adapter / 'selected.safetensors').write_bytes(b'changed')
        with patch.object(inference.training, 'runtime') as runtime:
            with self.assertRaisesRegex(ValueError, 'Adapter checksum'):
                inference.load_portable_adapter(self.base, self.adapter)
            runtime.assert_not_called()

    def test_different_policy_rejected(self):
        self.config['generation_policy']['config']['chunk_seconds'] = 10
        self.save()
        with self.assertRaisesRegex(ValueError, 'Decoding policy'):
            inference.load_portable_adapter(self.base, self.adapter)

    def test_different_base_weights_rejected(self):
        with patch.object(inference.training, 'runtime'), \
                patch.object(inference.training, 'load_base'), \
                patch.object(inference.training, 'inject_lora'), \
                patch.object(inference.training, 'frozen_fingerprint', return_value='different'), \
                patch.object(inference.training, 'restore_adapter') as restore:
            with self.assertRaisesRegex(ValueError, 'Base weights differ'):
                inference.load_portable_adapter(self.base, self.adapter)
            restore.assert_not_called()


if __name__ == '__main__':
    unittest.main()
