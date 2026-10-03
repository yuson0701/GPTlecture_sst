"""Install and select the CPU-efficient Korean streaming model. No API key."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local'))
from streaming import MODEL_NAME, MODEL_FILES

URL = f'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2'


def update_env(path):
    settings = {'STT_BACKEND': 'sherpa', 'LECTURE_SUMMARY_MODE': 'extractive'}
    lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    if path.exists():
        backup = path.with_name('.env.before-realtime')
        if not backup.exists():
            shutil.copy2(path, backup)
    updated = []
    for line in lines:
        key = line.split('=', 1)[0].strip()
        if key in settings:
            continue
        updated.append(line)
    updated.extend(f'{key}={value}' for key, value in settings.items())
    path.write_text('\n'.join(updated) + '\n', encoding='utf-8')


def download(destination):
    destination.mkdir(parents=True, exist_ok=True)
    if all((destination / name).is_file() for name in MODEL_FILES):
        print('Korean streaming model is already installed.', flush=True)
        return
    print('Downloading official Korean streaming model (several hundred MB, once).', flush=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        archive = Path(temporary) / 'model.tar.bz2'
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(archive, 'r:bz2') as tar:
            # Copy only named files. Never extract paths, links, or executable archive content.
            wanted = set(MODEL_FILES) | {'test_wavs/0.wav'}
            for name in wanted:
                member = tar.getmember(f'{MODEL_NAME}/{name}')
                if not member.isfile():
                    raise RuntimeError('Unexpected model archive content')
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                staging = target.with_suffix(target.suffix + '.partial')
                with tar.extractfile(member) as source, staging.open('wb') as output:
                    shutil.copyfileobj(source, output)
                staging.replace(target)
    print('Model installed. Runtime recognition is offline.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if not args.download_only:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-realtime.txt')], check=True)
    download(ROOT / 'models' / MODEL_NAME)
    if not args.download_only:
        update_env(ROOT / '.env')
        print('Selected Korean streaming mode and lightweight live summaries. Run npm start.', flush=True)


if __name__ == '__main__':
    main()
