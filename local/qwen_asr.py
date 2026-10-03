"""Qwen3-ASR via MLX Audio: Apple GPU, local weights, bounded audio windows."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import time

MODELS = {size: f'mlx-community/Qwen3-ASR-{size}-8bit' for size in ('1.7B', '0.6B')}
DEFAULT_MODEL = MODELS['1.7B']


def validate_platform():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Qwen3-ASR MLX는 Apple Silicon Mac의 ARM Python이 필요합니다. npm run setup:qwen을 실행하세요.')


def model_path():
    name = os.environ.get('QWEN_ASR_MODEL', DEFAULT_MODEL)
    path = Path(name).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(name, local_files_only=True))
    required = ['config.json', 'tokenizer_config.json', 'preprocessor_config.json']
    missing = [file for file in required if not (path / file).is_file()]
    # Qwen tokenizers may ship as vocabulary + merges instead of tokenizer.json.
    # Both layouts are accepted by Transformers' tokenizer loader.
    if not (path / 'tokenizer.json').is_file() and not all((path / file).is_file() for file in ('vocab.json', 'merges.txt')):
        missing.append('tokenizer.json 또는 vocab.json + merges.txt')
    if not list(path.glob('*.safetensors')):
        missing.append('*.safetensors')
    if missing:
        raise RuntimeError(f'Qwen3-ASR 파일 확인 실패: {", ".join(missing)}. 경로: {path}. npm run setup:qwen을 다시 실행하세요. 기존 다운로드는 재사용됩니다.')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        for shard in set(json.loads(index.read_text())['weight_map'].values()):
            if not (path / shard).is_file():
                raise RuntimeError(f'Qwen3-ASR 모델 조각이 없습니다: {shard}. 경로: {path}. 설치를 다시 실행하세요.')
    return path.resolve()


def status():
    try:
        validate_platform()
    except RuntimeError:
        return {'available': False, 'reason': 'platform'}
    if any(importlib.util.find_spec(name) is None for name in ('mlx_audio', 'faster_whisper')):
        return {'available': False, 'reason': 'dependencies'}
    try:
        model_path()
        return {'available': True}
    except Exception:
        return {'available': False, 'reason': 'model'}


class QwenRecognizer:
    def __init__(self):
        validate_platform()
        import mlx.core as mx
        import numpy as np
        from mlx_audio.stt import load
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        self.mx = mx
        self.vad = get_speech_timestamps
        self.vad_options = VadOptions(min_silence_duration_ms=300)
        self.model = load(model_path(), strict=True)
        # Compile/warm the Apple GPU before the browser opens its microphone.
        self.model.generate(np.zeros(16000, dtype=np.float32), language='Korean', max_tokens=8, verbose=False)
        mx.clear_cache()

    def transcribe(self, samples, glossary=''):
        if not 1600 <= len(samples) <= 16000 * 8:
            raise ValueError('Qwen audio window must be 0.1–8 seconds')
        started = time.perf_counter()
        if not self.vad(samples, self.vad_options):
            return {'text': '', 'inferenceMs': round((time.perf_counter() - started) * 1000, 1)}
        terms = [word.strip() for word in re.split(r'[,\n]', glossary) if word.strip()][:50]
        try:
            result = self.model.generate(samples, language='Korean', max_tokens=256,
                                         temperature=0.0, verbose=False, hotwords=terms)
            if result.generation_tokens >= 256:
                raise RuntimeError('음성 인식이 길이 제한에 도달했습니다. 이 구간을 다시 처리해 주세요.')
            return {'text': result.text.strip(), 'inferenceMs': round((time.perf_counter() - started) * 1000, 1)}
        finally:
            # Release transient allocations between windows during long lectures.
            self.mx.clear_cache()
