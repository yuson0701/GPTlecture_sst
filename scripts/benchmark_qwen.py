"""Run on the Mac: real window timings and optional reference CER, no API calls."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unicodedata
import wave

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local'))
from qwen_asr import QwenRecognizer


def normalized(text):
    return ''.join(c.lower() for c in unicodedata.normalize('NFC', text) if c.isalnum())


def cer(reference, hypothesis):
    left, right = normalized(reference), normalized(hypothesis)
    if not left:
        raise ValueError('Reference must contain letters or numbers')
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        new = [i]
        for j, b in enumerate(right, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1] / len(left)


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', help='16 kHz mono PCM16 Korean WAV')
    parser.add_argument('--reference', help='UTF-8 exact transcript for optional normalized character error rate')
    parser.add_argument('--model', help='Cached model path or repo; defaults to QWEN_ASR_MODEL')
    args = parser.parse_args()
    if args.model:
        os.environ['QWEN_ASR_MODEL'] = args.model
    with wave.open(args.audio, 'rb') as file:
        if (file.getframerate(), file.getnchannels(), file.getsampwidth()) != (16000, 1, 2):
            raise SystemExit('Use 16 kHz mono PCM16 WAV.')
        audio = np.frombuffer(file.readframes(file.getnframes()), dtype='<i2').astype(np.float32) / 32768
    if len(audio) < 1600:
        raise SystemExit('Audio must be at least 0.1 seconds.')
    model = QwenRecognizer()
    timings, texts = [], []
    for start in range(0, len(audio), 96000):
        part = audio[start:start + 96000]
        if len(part) < 1600:
            part = np.pad(part, (0, 1600 - len(part)))
        began = time.perf_counter()
        result = model.transcribe(part)
        timings.append(time.perf_counter() - began)
        texts.append(result['text'])
    text = ' '.join(texts)
    report = {'text': text, 'audioSeconds': len(audio) / 16000,
              'decodeSeconds': sum(timings), 'realTimeFactor': sum(timings) / (len(audio) / 16000),
              'windowDecodeP95Seconds': float(np.percentile(timings, 95)),
              'note': 'Final 6-second windows only; excludes rolling-preview work, capture latency and summary contention.'}
    if args.reference:
        report['normalizedCER'] = cer(Path(args.reference).read_text(encoding='utf-8'), text)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
