"""Load a portable MLX Qwen3-ASR adapter without private training files."""
import argparse
import json
from pathlib import Path

import train_qwen_lora as training
from qwen_decode import decode_audio, policy_proof


def load_portable_adapter(model_path, adapter_directory):
    directory = Path(adapter_directory)
    config = json.loads((directory / 'adapter_config.json').read_text())
    training.require(config.get('format') == 'qwen3-asr-mlx-lora-v1', 'Unsupported adapter format')
    training.require(config.get('generation_policy') == policy_proof(), 'Decoding policy differs from release')
    model_path = Path(model_path).expanduser().resolve()
    training.require(training.sha256(model_path / 'config.json') == config['base_config_sha256'],
                     'Base model configuration differs from the training snapshot')
    adapter = directory / 'selected.safetensors'
    training.require(training.sha256(adapter) == config['adapter_sha256'], 'Adapter checksum mismatch')
    training.runtime()
    model = training.load_base(model_path)
    training.inject_lora(model, config['rank'], config['layers'], config['alpha'])
    training.require(training.frozen_fingerprint(model) == config['base_frozen_sha256'],
                     'Base weights differ from the training snapshot')
    training.restore_adapter(model, adapter)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Local copy of the pinned base snapshot')
    parser.add_argument('--adapter', type=Path, required=True, help='Downloaded adapter directory')
    parser.add_argument('--audio', type=Path, required=True, help='Mono 16 kHz PCM16 WAV, at most 90 seconds')
    args = parser.parse_args()
    pcm = training.read_pcm(args.audio)
    model = load_portable_adapter(args.model, args.adapter)
    audio = training.np.frombuffer(pcm, dtype='<i2').astype(training.np.float32) / 32768
    result = decode_audio(model, audio, after_attempt=training.mx.clear_cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
