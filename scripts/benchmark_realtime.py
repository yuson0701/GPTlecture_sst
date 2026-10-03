"""Decode real Korean audio incrementally, optionally repeated for an hour of audio.
This is a throughput/continuity test, not a one-hour wall-clock thermal test.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local'))
from streaming import StreamingRecognizer, model_directory

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=60)
parser.add_argument('--audio', default=str(model_directory() / 'test_wavs' / '0.wav'))
args = parser.parse_args()
with wave.open(args.audio, 'rb') as wav:
    if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
        raise SystemExit('Benchmark input must be 16 kHz mono PCM16 WAV.')
    audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
recognizer = StreamingRecognizer()
session = recognizer.start()['session']
start = time.perf_counter()
timings, first = [], None
finals = 0
peak_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for sequence, offset in enumerate(range(0, int(args.seconds * 16000), 3200)):
    indices = (np.arange(min(3200, int(args.seconds * 16000) - offset)) + offset) % len(audio)
    result = recognizer.accept(audio[indices], session, sequence)
    timings.append(result['inferenceMs'])
    if result['events'] and first is None:
        first = {'audioSeconds': round((offset + len(indices)) / 16000, 2), 'text': result['events'][0]['text']}
    finals += sum(event['final'] for event in result['events'])
    if sequence and sequence % 3000 == 0:
        print(json.dumps({'audioMinutes': sequence / 300, 'wallSeconds': round(time.perf_counter() - start, 1)}), flush=True)
result = recognizer.accept(np.zeros(0, dtype=np.float32), session, sequence + 1, True)
finals += sum(event['final'] for event in result['events'])
wall = time.perf_counter() - start
# macOS reports bytes; Linux reports KiB.
scale = 1024 * 1024 if sys.platform == 'darwin' else 1024
print(json.dumps({'audioSeconds': args.seconds, 'wallSeconds': round(wall, 2), 'realTimeFactor': round(wall / args.seconds, 4),
                  'packetMsP50': round(float(np.percentile(timings, 50)), 1), 'packetMsP95': round(float(np.percentile(timings, 95)), 1),
                  'packetMsMax': max(timings), 'firstResult': first, 'finalizedSegments': finals,
                  'peakRssMB': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1),
                  'peakGrowthMB': round((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak_start) / scale, 1)}, ensure_ascii=False), flush=True)
