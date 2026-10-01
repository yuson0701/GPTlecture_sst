"""Persistent offline Whisper worker, with optional Apple Silicon GPU support."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
model = None


def backend():
    name = os.environ.get('WHISPER_BACKEND', 'faster-whisper')
    if name not in ('faster-whisper', 'mlx'):
        raise RuntimeError('WHISPER_BACKEND는 faster-whisper 또는 mlx로 설정하세요.')
    return name


def model_path():
    if backend() == 'mlx':
        from huggingface_hub import snapshot_download
        name = os.environ.get('MLX_WHISPER_MODEL', 'mlx-community/whisper-turbo')
        path = Path(name) if Path(name).is_dir() else Path(snapshot_download(name, local_files_only=True))
        weights = any((path / file).is_file() for file in ('model.safetensors', 'weights.safetensors', 'weights.npz'))
        if not (path / 'config.json').is_file() or not weights:
            raise RuntimeError('MLX 모델 파일이 없습니다. Mac GPU 설치 단계를 완료하세요.')
    else:
        from faster_whisper.utils import download_model
        name = os.environ.get('WHISPER_MODEL', 'large-v3-turbo')
        path = Path(name) if Path(name).is_dir() else Path(download_model(name, local_files_only=True))
        if not all((path / file).is_file() for file in ('model.bin', 'config.json', 'tokenizer.json')):
            raise RuntimeError('음성 모델 파일이 없습니다. 모델 다운로드 단계를 완료하세요.')
    return str(path)


def validate_platform():
    if backend() == 'mlx' and (platform.system() != 'Darwin' or platform.machine() != 'arm64'):
        raise RuntimeError('MLX는 Apple Silicon Mac에서만 사용할 수 있습니다. ARM 버전 Python을 사용하세요.')


def dispatch(request):
    global model
    engine = backend()
    if request.get('action') == 'status':
        try:
            validate_platform()
        except RuntimeError:
            return {'available': False, 'reason': 'platform'}
        dependencies = ['faster_whisper'] + (['mlx_whisper'] if engine == 'mlx' else [])
        if any(importlib.util.find_spec(name) is None for name in dependencies):
            return {'available': False, 'reason': 'dependencies'}
        try:
            model_path()
            return {'available': True, 'loaded': model is not None}
        except Exception:
            return {'available': False, 'reason': 'model'}
    if request.get('action') not in ('load', 'transcribe'):
        raise ValueError('Unknown action')
    if model is None:
        validate_platform()
        try:
            if engine == 'mlx':
                import mlx_whisper
                import numpy as np
                path = model_path()
                # Warm and cache the model before microphone capture starts.
                mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=path,
                                       language='ko', temperature=0.0, verbose=None)
                model = path
            else:
                from faster_whisper import WhisperModel
                model = WhisperModel(model_path(), device=os.environ.get('WHISPER_DEVICE', 'cpu'),
                                     compute_type=os.environ.get('WHISPER_COMPUTE_TYPE', 'int8'), local_files_only=True)
        except ImportError as exc:
            raise RuntimeError('음성 엔진 설치가 필요합니다. README의 로컬 / Mac GPU 설치 단계를 확인하세요.') from exc
        except Exception as exc:
            raise RuntimeError('음성 모델을 불러올 수 없습니다. 모델 다운로드 및 장치 설정을 확인하세요.') from exc
    if request['action'] == 'load':
        return {'loaded': True}
    import numpy as np
    audio = base64.b64decode(request['audio'], validate=True)
    if not 3200 <= len(audio) <= 480000 or len(audio) % 2:
        raise ValueError('Invalid PCM audio length')
    samples = np.frombuffer(audio, dtype='<i2').astype(np.float32) / 32768.0
    if engine == 'mlx':
        import mlx_whisper
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        # MLX Whisper has no standalone VAD; reuse local Silero to reject non-speech.
        if not get_speech_timestamps(samples, VadOptions(min_silence_duration_ms=300)):
            return {'text': ''}
        result = mlx_whisper.transcribe(samples, path_or_hf_repo=model, language='ko', temperature=0.0,
                                        initial_prompt=request.get('glossary') or None,
                                        condition_on_previous_text=False, without_timestamps=True, verbose=None)
        return {'text': result['text'].strip()}
    segments, _ = model.transcribe(
        samples, language='ko', beam_size=1, best_of=1, temperature=0.0,
        without_timestamps=True, vad_filter=True,
        vad_parameters={'min_silence_duration_ms': 300},
        initial_prompt=request.get('glossary') or None,
        condition_on_previous_text=False,
    )
    return {'text': ' '.join(s.text.strip() for s in segments).strip()}


if __name__ == '__main__':
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            print(json.dumps({'id': request['id'], 'result': dispatch(request)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            message = str(exc) if isinstance(exc, RuntimeError) else '로컬 음성 처리에 실패했습니다.'
            print(json.dumps({'id': request.get('id'), 'error': message}, ensure_ascii=False), flush=True)
