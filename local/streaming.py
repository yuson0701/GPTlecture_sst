"""Stateful Korean streaming ASR. Each PCM sample is decoded exactly once."""
import os
from pathlib import Path
import time
import uuid

MODEL_NAME = 'sherpa-onnx-streaming-zipformer-korean-2024-06-16'
MODEL_FILES = ('tokens.txt', 'encoder-epoch-99-avg-1.int8.onnx',
               'decoder-epoch-99-avg-1.onnx', 'joiner-epoch-99-avg-1.int8.onnx')


def model_directory():
    return Path(os.environ.get('SHERPA_MODEL_DIR', str(Path(__file__).resolve().parents[1] / 'models' / MODEL_NAME)))


def available():
    return all((model_directory() / name).is_file() for name in MODEL_FILES)


class StreamingRecognizer:
    def __init__(self):
        import sherpa_onnx
        root = model_directory()
        if not available():
            raise RuntimeError('스트리밍 모델을 설치하세요: npm run setup:realtime')
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(root / MODEL_FILES[0]), encoder=str(root / MODEL_FILES[1]),
            decoder=str(root / MODEL_FILES[2]), joiner=str(root / MODEL_FILES[3]),
            num_threads=2, sample_rate=16000, feature_dim=80,
            decoding_method='greedy_search', provider='cpu', enable_endpoint_detection=True,
            rule1_min_trailing_silence=1.2, rule2_min_trailing_silence=0.6,
            rule3_min_utterance_length=20,
        )
        self.session = None

    def start(self):
        self.session = str(uuid.uuid4())
        self.stream = self.recognizer.create_stream()
        self.sequence = -1
        self.reply = None
        self.segment = 0
        self.samples = 0
        self.segment_start = 0.0
        self.last_text = ''
        self.ended = False
        self.faulted = False
        return {'loaded': True, 'streaming': True, 'session': self.session}

    def accept(self, samples, session, sequence, final=False):
        import numpy as np
        if session != self.session:
            raise RuntimeError('강의 세션이 변경되었습니다. 현재 노트를 저장한 후 새 강의를 시작하세요.')
        # A lost HTTP reply can be retried without replaying samples into the model.
        if sequence == self.sequence and self.reply is not None:
            return self.reply
        if self.faulted:
            raise RuntimeError('음성 처리 상태를 복구할 수 없습니다. 노트를 저장한 후 새 강의를 시작하세요.')
        if self.ended or sequence != self.sequence + 1:
            raise RuntimeError('음성 순서가 맞지 않습니다. 현재 노트를 저장한 후 새 강의를 시작하세요.')
        started = time.perf_counter()
        # Never replay samples into partially mutated state after a decoder exception.
        self.faulted = True
        self.stream.accept_waveform(16000, samples)
        self.samples += len(samples)
        if final:
            self.stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
            self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream).strip()
        endpoint = final or self.recognizer.is_endpoint(self.stream)
        events = []
        if text and (text != self.last_text or endpoint):
            events.append({'id': f'stream-{self.segment}', 'seconds': self.segment_start,
                           'text': text, 'final': endpoint})
        self.last_text = text
        if endpoint:
            # Keep unread lookahead and encoder state; sherpa discards consumed feature frames.
            # Replacing the stream here would lose audio at 20-second boundaries.
            if not final:
                self.recognizer.reset(self.stream)
            self.segment += 1
            self.segment_start = self.samples / 16000
            self.last_text = ''
        self.faulted = False
        self.sequence = sequence
        self.ended = final
        self.reply = {'events': events, 'processedSeconds': self.samples / 16000,
                      'inferenceMs': round((time.perf_counter() - started) * 1000, 1)}
        return self.reply
