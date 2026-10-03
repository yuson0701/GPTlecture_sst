"""Conservative, dependency-free evidence checks for ASR dataset candidates.

The ASR hypothesis is independent evidence, not ground truth. These functions
never grant human approval. Offsets always index the original reference Python
string (end exclusive), and returned reference text is an exact source slice.
"""

from difflib import SequenceMatcher
import math
import unicodedata


def _fold(text):
    return unicodedata.normalize("NFKC", text).casefold()


def normalize_with_offsets(text):
    """Return Unicode alphanumeric text and its original character spans.

    Normalization is per lexical run, so decomposed Hangul/combining accents are
    equivalent to their composed forms. Compatibility expansions retain their
    source span. Rare composition replacements conservatively retain the whole
    replaced source span; an ambiguous cut will subsequently fail fidelity/CER.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized, offsets = [], []
    cursor = 0
    while cursor < len(text):
        if not (text[cursor].isalnum() or unicodedata.category(text[cursor]).startswith("M")):
            cursor += 1
            continue
        stop = cursor + 1
        while stop < len(text) and (
            text[stop].isalnum() or unicodedata.category(text[stop]).startswith("M")
        ):
            stop += 1
        run = text[cursor:stop]
        individual, origins = [], []
        for index, char in enumerate(run, cursor):
            folded = _fold(char)
            individual.extend(folded)
            origins.extend([(index, index + 1)] * len(folded))
        individual = "".join(individual)
        folded = _fold(run)
        if individual == folded:
            folded_origins = origins
        else:
            folded_origins = []
            for tag, a, b, c, d in SequenceMatcher(
                None, individual, folded, autojunk=False
            ).get_opcodes():
                if tag == "equal":
                    folded_origins.extend(origins[a:b])
                elif tag != "delete":
                    span = (origins[a][0], origins[b - 1][1]) if b > a else (cursor, stop)
                    folded_origins.extend([span] * (d - c))
        for char, span in zip(folded, folded_origins):
            if char.isalnum():
                normalized.append(char)
                offsets.append(span)
        cursor = stop
    return "".join(normalized), offsets


def normalize_text(text):
    return "".join(char for char in _fold(text) if char.isalnum())


def _edit_distance(left, right):
    """Exact Levenshtein distance, using a bit-vector dynamic program."""
    if not left:
        return len(right)
    if not right:
        return len(left)
    # Use the shorter string as the bit vector. This remains fast even when a
    # bad global mapping spans thousands of reference characters.
    if len(left) > len(right):
        left, right = right, left
    width = len(left)
    mask = (1 << width) - 1
    high = 1 << (width - 1)
    char_masks = {}
    for index, char in enumerate(left):
        char_masks[char] = char_masks.get(char, 0) | (1 << index)
    positive, negative, distance = mask, 0, width
    for char in right:
        equal = char_masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        if plus & high:
            distance += 1
        elif minus & high:
            distance -= 1
        plus = (plus << 1) | 1
        minus <<= 1
        positive = (minus | ~(vertical | plus)) & mask
        negative = plus & vertical
    return distance


def character_error_rate(reference, hypothesis):
    """CER after NFKC, case folding, and removal of non-alphanumeric chars."""
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else None
    return _edit_distance(reference, hypothesis) / len(reference)


def _issue(row, code):
    if code not in row["issues"]:
        row["issues"].append(code)


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def map_windows_to_reference(
    reference_text, windows, *, max_cer=0.25, min_match_coverage=0.75, min_anchor_chars=4
):
    """Map consecutive ASR windows to a complete lecture reference.

    Each input row has ``start``, ``end``, ``text``, ``truncated``, and
    ``window_index``. Input order must be chronological. Output copies each row
    and adds ``reference_text``, ``reference_start``, ``reference_end``, ``cer``,
    ``match_coverage``, and ``issues``. Offsets/CER are None when unmapped.

    Coverage is exact matched normalized characters divided by the larger of
    hypothesis length and the mapped reference length. Global matching is
    monotonic, not an independent substring search for each window. Unassigned
    reference characters at boundaries flag adjacent candidates, so omissions
    cannot disappear silently. Windows with issues must not enter training.
    """
    if not (0 <= max_cer <= 1 and 0 <= min_match_coverage <= 1):
        raise ValueError("CER/coverage thresholds must lie between zero and one")
    if min_anchor_chars < 1:
        raise ValueError("min_anchor_chars must be positive")
    ref, offsets = normalize_with_offsets(reference_text)
    results, hypotheses, boundaries = [], [], []
    total = 0
    previous_end = None
    index_rows = {}
    for position, source in enumerate(windows):
        row = dict(source)
        row.update(reference_text="", reference_start=None, reference_end=None,
                   cer=None, match_coverage=0.0, issues=[])
        text = source.get("text", "")
        if not isinstance(text, str):
            text = ""
            _issue(row, "invalid_hypothesis_text")
        hyp = normalize_text(text)
        if not hyp:
            _issue(row, "empty_hypothesis")
        elif len(hyp) < min_anchor_chars:
            _issue(row, "insufficient_anchor_text")
        if source.get("truncated"):
            _issue(row, "truncated_hypothesis")
        start, end = source.get("start"), source.get("end")
        if not (_finite_number(start) and _finite_number(end) and 0 <= start < end):
            _issue(row, "invalid_window_time")
        else:
            if previous_end is not None and start < previous_end - 1e-6:
                _issue(row, "overlapping_or_unordered_windows")
                if results:
                    _issue(results[-1], "overlapping_or_unordered_windows")
            previous_end = end
        window_index = source.get("window_index", position)
        if window_index in index_rows:
            _issue(row, "duplicate_window_index")
            _issue(results[index_rows[window_index]], "duplicate_window_index")
        else:
            index_rows[window_index] = position
        if not ref:
            _issue(row, "empty_reference")
        results.append(row)
        hypotheses.append(hyp)
        boundaries.append((total, total + len(hyp)))
        total += len(hyp)
    if not ref or not total:
        for row in results:
            _issue(row, "unmapped_reference")
        return results

    # Ref coordinates for exact matched hypothesis characters; -1 means no
    # anchor. SequenceMatcher's ordered blocks guarantee monotonic ownership.
    anchors = [-1] * total
    for ref_start, hyp_start, size in SequenceMatcher(
        None, ref, "".join(hypotheses), autojunk=False
    ).get_matching_blocks():
        anchors[hyp_start:hyp_start + size] = range(ref_start, ref_start + size)

    spans = []
    for row, hyp, (start, end) in zip(results, hypotheses, boundaries):
        matches = [anchor for anchor in anchors[start:end] if anchor >= 0]
        if not matches:
            _issue(row, "unmapped_reference")
            spans.append(None)
            continue
        first, last = matches[0], matches[-1] + 1
        raw_start, raw_end = offsets[first][0], offsets[last - 1][1]
        # Keep source punctuation after the final matched character, but no
        # following letters. Do not include trailing whitespace in the slice.
        next_start = offsets[last][0] if last < len(offsets) else len(reference_text)
        if next_start >= raw_end:
            raw_end = len(reference_text[:next_start].rstrip())
        candidate = reference_text[raw_start:raw_end]
        row.update(reference_text=candidate, reference_start=raw_start,
                   reference_end=raw_end,
                   cer=character_error_rate(candidate, row.get("text", "")),
                   match_coverage=len(matches) / max(len(hyp), last - first))
        if row["cer"] is None or row["cer"] > max_cer:
            _issue(row, "high_cer")
        if row["match_coverage"] < min_match_coverage:
            _issue(row, "weak_reference_match")
        if len(matches) < min_anchor_chars:
            _issue(row, "insufficient_matching_anchors")
        spans.append((first, last))

    mapped = [index for index, span in enumerate(spans) if span is not None]
    if mapped:
        if spans[mapped[0]][0] > 0:
            _issue(results[mapped[0]], "unmapped_reference_prefix")
        if spans[mapped[-1]][1] < len(ref):
            _issue(results[mapped[-1]], "unmapped_reference_suffix")
    for previous, current in zip(mapped, mapped[1:]):
        before, after = results[previous], results[current]
        if spans[current][0] > spans[previous][1]:
            _issue(before, "reference_gap_after")
            _issue(after, "reference_gap_before")
        if after["reference_start"] < before["reference_end"]:
            _issue(before, "overlapping_reference_span")
            _issue(after, "overlapping_reference_span")
        if (before["reference_start"], before["reference_end"]) == (
            after["reference_start"], after["reference_end"]
        ):
            _issue(before, "duplicate_reference_span")
            _issue(after, "duplicate_reference_span")
    return results


def prepare_crop_candidate(
    reference_text, row, *, max_boundary_chars=4, max_cer=0.25, min_match_coverage=0.75
):
    """Refine one mapped row for a later acoustic crop, without approving it.

    Gaps outside a retained span are lecture coverage warnings when cropping
    to the aligned words. A span may not begin/end inside a lexical run: expand
    to its full Unicode alphanumeric/combining-mark boundaries, or reject if
    either expansion exceeds ``max_boundary_chars`` normalized characters.
    Recheck reference/hypothesis agreement after expansion. The caller must
    still force-align in sufficient audio context, check context-edge margins,
    and reject overlapping resulting clips; neighboring rows are not available
    to this pure function. Original rows and issue lists are never mutated.
    """
    if not (0 <= max_cer <= 1 and 0 <= min_match_coverage <= 1):
        raise ValueError("CER/coverage thresholds must lie between zero and one")
    if not isinstance(max_boundary_chars, int) or max_boundary_chars < 0:
        raise ValueError("max_boundary_chars must be a nonnegative integer")
    if not isinstance(reference_text, str):
        raise TypeError("reference_text must be a string")
    result = dict(row)
    result["issues"] = list(row.get("issues", []))
    result["coverage_warnings"] = list(row.get("coverage_warnings", []))
    start, end = row.get("reference_start"), row.get("reference_end")
    if not (
        isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and 0 <= start < end <= len(reference_text)
    ):
        _issue(result, "unmapped_reference")
        return result
    if row.get("reference_text") != reference_text[start:end]:
        _issue(result, "reference_source_mismatch")
        return result

    coverage_codes = {
        "reference_gap_before", "reference_gap_after",
        "unmapped_reference_prefix", "unmapped_reference_suffix",
    }
    for code in result["issues"]:
        if code in coverage_codes and code not in result["coverage_warnings"]:
            result["coverage_warnings"].append(code)
    result["issues"] = [code for code in result["issues"] if code not in coverage_codes]

    def lexical(char):
        return char.isalnum() or unicodedata.category(char).startswith("M")

    new_start, new_end = start, end
    if lexical(reference_text[start]):
        while new_start > 0 and lexical(reference_text[new_start - 1]):
            new_start -= 1
    if lexical(reference_text[end - 1]):
        while new_end < len(reference_text) and lexical(reference_text[new_end]):
            new_end += 1
    left_extra = len(normalize_text(reference_text[new_start:start]))
    right_extra = len(normalize_text(reference_text[end:new_end]))
    if max(left_extra, right_extra) > max_boundary_chars:
        _issue(result, "partial_word_boundary")
    else:
        result["reference_start"], result["reference_end"] = new_start, new_end
        result["reference_text"] = reference_text[new_start:new_end]
    reference = normalize_text(result["reference_text"])
    hypothesis_text = row.get("text", "")
    if not isinstance(hypothesis_text, str):
        _issue(result, "invalid_hypothesis_text")
        hypothesis_text = ""
    hypothesis = normalize_text(hypothesis_text)
    result["issues"] = [code for code in result["issues"] if code not in {"high_cer", "weak_reference_match"}]
    result["cer"] = character_error_rate(result["reference_text"], hypothesis_text)
    matched = sum(block.size for block in SequenceMatcher(None, reference, hypothesis, autojunk=False).get_matching_blocks())
    result["match_coverage"] = matched / max(len(reference), len(hypothesis), 1)
    if result["cer"] is None or result["cer"] > max_cer:
        _issue(result, "high_cer")
    if result["match_coverage"] < min_match_coverage:
        _issue(result, "weak_reference_match")
    if not hypothesis:
        _issue(result, "empty_hypothesis")
    return result


def audit_alignment(items, text, duration):
    """Return conservative issue codes for a forced-alignment result.

    ``items`` accepts dicts or Qwen objects exposing text/start_time/end_time.
    Even an empty issue list is automated evidence, not approval or proof that
    the reference is correct: forced aligners can align erroneous transcripts.
    """
    issues = []

    def add(code):
        if code not in issues:
            issues.append(code)

    if not (_finite_number(duration) and duration > 0):
        add("invalid_audio_duration")
    reference = normalize_text(text) if isinstance(text, str) else ""
    if not reference:
        add("empty_alignment_reference")
    items = list(items)
    if not items:
        add("empty_alignment")
        return issues
    aligned_text, total_speech, previous_start, previous_end = [], 0.0, None, None
    for item in items:
        def get(key):
            return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

        token, start, end = get("text"), get("start_time"), get("end_time")
        if not isinstance(token, str) or not normalize_text(token):
            add("empty_alignment_token")
        else:
            aligned_text.append(token)
        if not (_finite_number(start) and _finite_number(end)):
            add("nonfinite_alignment_timestamp")
            continue
        if start < 0 or (_finite_number(duration) and end > duration + 1e-6):
            add("alignment_out_of_bounds")
        if end < start:
            add("reversed_alignment_span")
        elif end == start:
            add("zero_duration_alignment_token")
        else:
            total_speech += end - start
        if previous_start is not None and start < previous_start:
            add("nonmonotonic_alignment")
        if previous_end is not None:
            if start < previous_end - 1e-6:
                add("overlapping_alignment_spans")
            elif _finite_number(duration) and start - previous_end > max(5.0, duration * 0.25):
                add("long_internal_alignment_gap")
        previous_start, previous_end = start, end
    if normalize_text("".join(aligned_text)) != reference:
        add("alignment_text_mismatch")
    if total_speech > 0 and len(reference) / total_speech > 40:
        add("implausible_alignment_speech_rate")
    return issues


def boundary_review_eligible(items, text, duration, cer, match_coverage):
    """Whether an otherwise sound clip with isolated zero spans merits review.

    This is a REVIEW-QUEUE predicate, never a training pass or approval. Keep
    the zero-duration issue and every reference token. The caller must retain
    ``human_review='pending'`` and require human approval before export.
    Strict ``audit_alignment`` behavior intentionally remains unchanged.
    """
    items = list(items)
    if audit_alignment(items, text, duration) != ["zero_duration_alignment_token"]:
        return False
    if len(items) < 10:
        return False
    if not (
        _finite_number(cer) and 0 <= cer <= 0.15
        and _finite_number(match_coverage) and 0.85 <= match_coverage <= 1
    ):
        return False

    def get(item, key):
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

    spans = [(get(item, "start_time"), get(item, "end_time")) for item in items]
    # The strict audit above has already verified finite, ordered timestamps.
    if spans[0][1] <= spans[0][0] or spans[-1][1] <= spans[-1][0]:
        return False
    if spans[0][0] < 0.08 - 1e-9 or duration - spans[-1][1] < 0.08 - 1e-9:
        return False
    zeros = [end == start for start, end in spans]
    if sum(zeros) / len(items) > 0.10:
        return False
    if any(before and after for before, after in zip(zeros, zeros[1:])):
        return False
    if any(end - start > 3 for start, end in spans):
        return False
    return True
