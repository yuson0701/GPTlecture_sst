"""One-time Mac installation. Select Qwen only after download and GPU load succeed."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local'))
from qwen_asr import MODELS, DEFAULT_ADAPTER, BASE_REVISION, ADAPTER_REVISION, validate_platform


def update_env(path, settings):
    lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    backup = path.with_name('.env.before-qwen')
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    kept = [line for line in lines if line.split('=', 1)[0].strip() not in settings]
    path.write_text('\n'.join(kept + [f'{key}={value}' for key, value in settings.items()]) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, default='1.7B')
    parser.add_argument('--base-only', action='store_true', help='Explicitly use the original model without the lecture adapter')
    parser.add_argument('--skip-install', action='store_true', help='Reuse installed dependencies; GPU verification still runs')
    args = parser.parse_args()
    if not args.base_only and args.size != '1.7B':
        parser.error('The lecture adapter requires 1.7B. Use --base-only --size 0.6B for the original smaller model.')
    validate_platform()
    if not args.skip_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-qwen-mac.txt')], check=True)
    from huggingface_hub import snapshot_download
    repo = MODELS[args.size]
    print(f'Downloading {repo}. Fine-tuned setup also needs Hugging Face access to {DEFAULT_ADAPTER}.', flush=True)
    # Runtime uses this exact cached snapshot, not a moving remote branch.
    destination = snapshot_download(repo, revision=BASE_REVISION if args.size == '1.7B' else None,
                                    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken'])
    adapter = 'none'
    if not args.base_only:
        try:
            adapter = snapshot_download(DEFAULT_ADAPTER, revision=ADAPTER_REVISION,
                                        allow_patterns=['adapter_config.json', 'selected.safetensors'])
        except Exception as exc:
            raise SystemExit(f'Cannot download the private lecture adapter. Run .venv-qwen/bin/hf auth login with an account that has access to {DEFAULT_ADAPTER}, then retry. Existing .env is unchanged.') from exc
    settings = {'STT_BACKEND': 'qwen-mlx', 'QWEN_ASR_MODEL': destination, 'QWEN_ASR_ADAPTER': adapter,
                'QWEN_ASR_ADAPTER_REVISION': ADAPTER_REVISION if adapter != 'none' else '',
                'PYTHON_BIN': str(ROOT / '.venv-qwen/bin/python')}
    env = {**os.environ, **settings, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    print('Checking complete files and warming the Apple GPU before changing settings…', flush=True)
    verification = subprocess.run([sys.executable, '-c', 'from qwen_asr import QwenRecognizer; QwenRecognizer(); print("Qwen GPU load OK")'], cwd=ROOT / 'local', env=env, check=False)
    if verification.returncode:
        raise SystemExit('Qwen 모델 확인에 실패했습니다. 위 오류를 확인하세요. 기존 .env는 변경하지 않았으며 다운로드 파일은 보관되어 있습니다.')
    update_env(ROOT / '.env', settings)
    print(f'Qwen3-ASR {"base model" if args.base_only else "fine-tuned lecture model"} selected. Restart npm start and refresh the browser. Prior settings: .env.before-qwen', flush=True)


if __name__ == '__main__':
    main()
