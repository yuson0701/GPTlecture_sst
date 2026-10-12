"""Install Apple GPU Whisper alongside Qwen; only select paths after a load check."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def update_env(path, settings):
    original = path.read_text() if path.exists() else ''
    backup = path.with_name('.env.before-whisper')
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    lines = [line for line in original.splitlines() if line.split('=', 1)[0].strip() not in settings]
    path.write_text('\n'.join(lines + [f'{key}={value}' for key, value in settings.items()]) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('Apple Silicon Mac and ARM Python are required.')
    if not args.skip_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-mac.txt')], check=True)
    from huggingface_hub import snapshot_download
    path = snapshot_download('mlx-community/whisper-turbo', allow_patterns=['config.json', '*.safetensors', '*.npz'])
    settings = {'WHISPER_PYTHON_BIN': str(ROOT / '.venv-whisper/bin/python'),
                'WHISPER_CHOICE_BACKEND': 'mlx', 'MLX_WHISPER_MODEL': path}
    env = {**os.environ, **settings, 'STT_BACKEND': 'mlx', 'HF_HUB_OFFLINE': '1'}
    subprocess.run([sys.executable, '-c', 'from worker import dispatch; dispatch({"action":"load"}); print("Whisper GPU load OK")'],
                   cwd=ROOT / 'local', env=env, check=True)
    update_env(ROOT / '.env', settings)
    print('Whisper is installed alongside Qwen. Restart npm start, refresh, and select Whisper before recording.')


if __name__ == '__main__':
    main()
