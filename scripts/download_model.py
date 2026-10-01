"""Explicit one-time model download. Run in the project's virtual environment."""
import argparse
from faster_whisper.utils import download_model

parser = argparse.ArgumentParser(description='Download a local Korean-capable Whisper model.')
parser.add_argument('--model', default='large-v3-turbo', help='Must match WHISPER_MODEL in .env')
args = parser.parse_args()
print(f'Downloading {args.model}. Internet and disk space are required for this step.')
print(download_model(args.model))
print('Download complete. Lecture-time recognition uses cached files only.')
