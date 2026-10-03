"""Desktop first-run model download. No pip, shell, or writes inside the app bundle."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local'))
from qwen_asr import DEFAULT_MODEL, DEFAULT_ADAPTER, BASE_REVISION, ADAPTER_REVISION, QwenRecognizer, validate_platform


def save_config(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8', delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    validate_platform()
    from huggingface_hub import snapshot_download
    print('1/3 · 한국어 음성 모델 다운로드 (기존 캐시 재사용)', flush=True)
    base = snapshot_download(DEFAULT_MODEL, revision=BASE_REVISION,
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken'])
    print('2/3 · 한국어 강의 미세조정 모델 다운로드', flush=True)
    adapter = snapshot_download(DEFAULT_ADAPTER, revision=ADAPTER_REVISION,
                                allow_patterns=['adapter_config.json', 'selected.safetensors'])
    os.environ.update(QWEN_ASR_MODEL=base, QWEN_ASR_ADAPTER=adapter, HF_HUB_OFFLINE='1')
    print('3/3 · 모델 검증 및 Apple GPU 준비', flush=True)
    QwenRecognizer()
    save_config(args.config, {'schema': 1, 'base': base, 'adapter': adapter,
                             'baseRevision': BASE_REVISION, 'adapterRevision': ADAPTER_REVISION})
    print('설치 완료 · 강의 노트를 엽니다.', flush=True)


if __name__ == '__main__':
    main()
