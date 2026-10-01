"""Persistent, offline-only faster-whisper worker. JSON-lines over stdin/stdout."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys

# Model downloads happen only via scripts/download_model.py, never during a lecture.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
model = None


def model_path():
    from faster_whisper.utils import download_model
    name = os.environ.get('WHISPER_MODEL', 'large-v3-turbo')
    path = Path(name) if Path(name).is_dir() else Path(download_model(name, local_files_only=True))
    # Requiring a local tokenizer also prevents library fallback downloads.
    if not all((path / file).is_file() for file in ('model.bin', 'config.json', 'tokenizer.json')):
        raise RuntimeError('음성 모델 파일이 없습니다. README의 모델 다운로드 단계를 완료하세요.')
    return str(path)


def dispatch(request):
    global model
    if request.get('action') == 'status':
        if importlib.util.find_spec('faster_whisper') is None:
            return {'available': False, 'reason': 'dependencies'}
        try:
            model_path()
            return {'available': True, 'loaded': model is not None}
        except Exception:
            return {'available': False, 'reason': 'model'}
    if request.get('action') not in ('load', 'transcribe'):
        raise ValueError('Unknown action')
    if model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('faster-whisper 설치가 필요합니다. README의 로컬 설치 단계를 확인하세요.') from exc
        try:
            model = WhisperModel(
                model_path(),
                device=os.environ.get('WHISPER_DEVICE', 'cpu'),
                compute_type=os.environ.get('WHISPER_COMPUTE_TYPE', 'int8'),
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError('음성 모델을 불러올 수 없습니다. 모델을 먼저 다운로드하고 WHISPER_MODEL / 장치 설정을 확인하세요.') from exc
    if request['action'] == 'load':
        return {'loaded': True}
    import numpy as np
    audio = base64.b64decode(request['audio'], validate=True)
    if not 3200 <= len(audio) <= 480000 or len(audio) % 2:
        raise ValueError('Invalid PCM audio length')
    samples = np.frombuffer(audio, dtype='<i2').astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        samples, language='ko', beam_size=3, vad_filter=True,
        vad_parameters={'min_silence_duration_ms': 400},
        initial_prompt=request.get('glossary') or None,
        condition_on_previous_text=False,
    )
    return {'text': ' '.join(s.text.strip() for s in segments).strip()}


if __name__ == '__main__':
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            result = dispatch(request)
            print(json.dumps({'id': request['id'], 'result': result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            # No raw audio, transcript, or internal exception details in logs.
            message = str(exc) if isinstance(exc, RuntimeError) else '로컬 음성 처리에 실패했습니다.'
            print(json.dumps({'id': request.get('id'), 'error': message}, ensure_ascii=False), flush=True)
