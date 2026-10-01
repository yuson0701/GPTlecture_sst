"""Explicit one-time download. No weights are downloaded during a lecture."""
import argparse

parser = argparse.ArgumentParser(description='Download a local Korean-capable Whisper model.')
parser.add_argument('--backend', choices=['faster-whisper', 'mlx'], default='faster-whisper')
parser.add_argument('--model', help='Must match WHISPER_MODEL or MLX_WHISPER_MODEL in .env')
args = parser.parse_args()
name = args.model or ('mlx-community/whisper-turbo' if args.backend == 'mlx' else 'large-v3-turbo')
print(f'Downloading {name}. Internet and disk space are required for this step.')
if args.backend == 'mlx':
    from huggingface_hub import snapshot_download
    print(snapshot_download(name, allow_patterns=['config.json', '*.safetensors', '*.npz']))
else:
    from faster_whisper.utils import download_model
    print(download_model(name))
print('Download complete. Lecture-time recognition uses cached files only.')
