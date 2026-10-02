"""One-time Mac installation. Select Qwen only after download and GPU load succeed."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local'))
from qwen_asr import MODELS, validate_platform


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
    args = parser.parse_args()
    validate_platform()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-qwen-mac.txt')], check=True)
    from huggingface_hub import snapshot_download
    repo = MODELS[args.size]
    print(f'Downloading {repo}. Weights are hosted on Hugging Face; no API key is required.', flush=True)
    # Runtime uses this exact cached snapshot, not a moving remote branch.
    destination = snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken'])
    env = {**os.environ, 'QWEN_ASR_MODEL': destination, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    print('Checking complete files and warming the Apple GPU before changing settings…', flush=True)
    subprocess.run([sys.executable, '-c', 'from qwen_asr import QwenRecognizer; QwenRecognizer(); print("Qwen GPU load OK")'], cwd=ROOT / 'local', env=env, check=True)
    update_env(ROOT / '.env', {'STT_BACKEND': 'qwen-mlx', 'QWEN_ASR_MODEL': destination,
                               'PYTHON_BIN': str(ROOT / '.venv-qwen/bin/python')})
    print('Qwen3-ASR selected. Restart npm start and refresh the browser. Prior settings: .env.before-qwen', flush=True)


if __name__ == '__main__':
    main()
