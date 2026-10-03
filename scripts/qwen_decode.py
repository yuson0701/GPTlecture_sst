"""Reference-free, bounded retries for Qwen3-ASR decoding; never edits transcripts.

A failed full prediction is discarded only from the final concatenated result,
but remains in the evidence. Sample intervals describe audio coverage, not word
alignment. Unresolved leaves retain their unmodified predictions and warnings.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import unicodedata


@dataclass(frozen=True)
class DecodePolicy:
    version: str = 'qwen-bounded-audio-retry-v1'
    sample_rate: int = 16000
    language: str = 'Korean'
    temperature: float = 0.0
    full_max_tokens: int = 2048
    chunk_max_tokens: int = 768
    chunk_seconds: float = 15.0
    minimum_split_seconds: float = 3.0
    max_depth: int = 3
    repetition_copies: int = 10
    max_ngram_words: int = 8


DEFAULT_POLICY = DecodePolicy()


def policy_proof(policy=DEFAULT_POLICY):
    values = asdict(policy)
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'config': values, 'sha256': digest}


def repeated_ngram(text, *, minimum_copies=10, maximum_words=8):
    """Detect consecutive repeated lexical n-grams; normalization is detector-only."""
    if minimum_copies < 2 or maximum_words < 1:
        raise ValueError('Invalid repetition detector bounds')
    words = re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold(), flags=re.UNICODE)
    best = None
    for size in range(1, min(maximum_words, len(words) // minimum_copies) + 1):
        for offset in range(size):
            previous, copies, start = None, 0, offset
            for index in range(offset, len(words) - size + 1, size):
                phrase = tuple(words[index:index + size])
                if phrase == previous:
                    copies += 1
                else:
                    previous, copies, start = phrase, 1, index
                if copies >= minimum_copies and (best is None or copies * size > best['repeated_words']):
                    best = {'ngram': list(phrase), 'copies': copies, 'ngram_words': size,
                            'start_word': start, 'repeated_words': copies * size}
    return best


def decode_audio(model, audio, *, policy=DEFAULT_POLICY, after_attempt=None):
    """Return text, initial evidence, attempts and exhaustive final audio intervals.

    No reference/target parameter exists. ``after_attempt`` may clear a model
    allocator cache; it cannot change policy or any text. Full-audio success is
    returned byte-for-byte. Retry text is joined with a single separating space.
    """
    import numpy as np
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio must be a nonempty finite mono NumPy waveform')
    if (policy.sample_rate != 16000 or policy.language != 'Korean' or policy.temperature != 0
            or policy.full_max_tokens < 1 or policy.chunk_max_tokens < 1
            or not 0 < policy.minimum_split_seconds <= policy.chunk_seconds
            or not math.isfinite(policy.chunk_seconds) or policy.max_depth < 0
            or policy.repetition_copies < 2 or policy.max_ngram_words < 1):
        raise ValueError('Invalid bounded decoding policy')
    attempts, leaves = [], []

    def attempt(start, end, budget, stage, depth):
        result = model.generate(audio[start:end], language=policy.language, temperature=policy.temperature,
                                max_tokens=budget, verbose=False)
        if not isinstance(result.text, str) or type(result.generation_tokens) is not int or result.generation_tokens < 0:
            raise ValueError('Model returned invalid text or token count')
        repetition = repeated_ngram(result.text, minimum_copies=policy.repetition_copies,
                                     maximum_words=policy.max_ngram_words)
        issues = (['generation_token_limit'] if result.generation_tokens >= budget else [])
        if repetition is not None:
            issues.append('consecutive_repetition')
        record = {'attempt_id': len(attempts), 'stage': stage, 'depth': depth,
                  'start_sample': start, 'end_sample': end,
                  'start_seconds': start / policy.sample_rate, 'end_seconds': end / policy.sample_rate,
                  'max_tokens': budget, 'generation_tokens': result.generation_tokens,
                  'text_sha256': hashlib.sha256(result.text.encode('utf-8')).hexdigest(),
                  'text_characters': len(result.text), 'issues': issues, 'repetition': repetition}
        attempts.append(record)
        if after_attempt is not None:
            after_attempt()
        return result.text, record

    def visit(start, end, depth):
        text, record = attempt(start, end, policy.chunk_max_tokens, 'chunk' if depth == 0 else 'bisect', depth)
        midpoint = (start + end) // 2
        minimum = math.ceil(policy.minimum_split_seconds * policy.sample_rate)
        if record['issues'] and depth < policy.max_depth and midpoint - start >= minimum and end - midpoint >= minimum:
            visit(start, midpoint, depth + 1)
            visit(midpoint, end, depth + 1)
        else:
            leaves.append({**record, 'text': text, 'unresolved': bool(record['issues'])})

    initial_text, initial = attempt(0, len(audio), policy.full_max_tokens, 'full', 0)
    if not initial['issues']:
        leaves.append({**initial, 'text': initial_text, 'unresolved': False})
        final_text = initial_text
    else:
        maximum = max(1, math.floor(policy.chunk_seconds * policy.sample_rate))
        count = math.ceil(len(audio) / maximum)
        # Evenly distribute the tail, avoiding an artificial sub-3-second last chunk.
        boundaries = [(index * len(audio)) // count for index in range(count + 1)]
        for start, end in zip(boundaries, boundaries[1:]):
            visit(start, end, 0)
        final_text = ' '.join(leaf['text'] for leaf in leaves)
    cursor = 0
    for leaf in leaves:
        if leaf['start_sample'] != cursor or not leaf['end_sample'] > cursor:
            raise ValueError('Retry intervals lost or overlapped audio')
        cursor = leaf['end_sample']
    if cursor != len(audio):
        raise ValueError('Retry intervals do not cover all input audio')
    unresolved = [leaf['attempt_id'] for leaf in leaves if leaf['unresolved']]
    return {'text': final_text, 'generation_tokens': sum(leaf['generation_tokens'] for leaf in leaves),
            'initial_text': initial_text, 'initial_generation_tokens': initial['generation_tokens'],
            'initial_problem': initial['issues'], 'initial_repetition': initial['repetition'],
            'retries': len(attempts) - 1, 'fallback_used': bool(initial['issues']),
            'unresolved': bool(unresolved), 'unresolved_leaf_attempt_ids': unresolved,
            'leaf_intervals': leaves, 'attempts': attempts,
            'attempt_generation_tokens': sum(attempt['generation_tokens'] for attempt in attempts),
            'audio_samples': len(audio), 'all_audio_covered': True, 'policy': policy_proof(policy)}
