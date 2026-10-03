import random
import sys
from pathlib import Path
from types import SimpleNamespace
import unicodedata
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen_alignment_checks import (
    _edit_distance, audit_alignment, boundary_review_eligible, character_error_rate,
    map_windows_to_reference, normalize_text, normalize_with_offsets, prepare_crop_candidate,
)


def windows(*texts):
    return [dict(start=i * 30, end=(i + 1) * 30, text=text,
                 truncated=False, window_index=i) for i, text in enumerate(texts)]


class ReferenceMatchingTests(unittest.TestCase):
    def test_korean_english_punctuation_and_offsets(self):
        reference = "오늘은 경제학입니다! 다음은 GDP growth입니다."
        rows = map_windows_to_reference(reference, windows("오늘은 경제학입니다", "다음은 gdp GROWTH입니다"))
        for row in rows:
            self.assertEqual(row["issues"], [])
            self.assertEqual(row["cer"], 0)
            self.assertEqual(row["match_coverage"], 1)
            self.assertEqual(row["reference_text"], reference[row["reference_start"]:row["reference_end"]])
        self.assertLessEqual(rows[0]["reference_end"], rows[1]["reference_start"])
        self.assertTrue(rows[0]["reference_text"].endswith("!"))

    def test_shifted_reference_is_not_silently_accepted(self):
        rows = map_windows_to_reference("이전 내용입니다. 오늘은 경제학입니다. 다음 내용입니다.", windows("오늘은 경제학입니다"))
        self.assertIn("unmapped_reference_prefix", rows[0]["issues"])
        self.assertIn("unmapped_reference_suffix", rows[0]["issues"])

    def test_repeated_words_have_distinct_monotonic_ownership(self):
        reference = "수요 수요 공급 공급. 수요 수요 공급 공급. 마지막 강의입니다."
        rows = map_windows_to_reference(reference, windows("수요 수요 공급 공급", "수요 수요 공급 공급", "마지막 강의입니다"))
        self.assertTrue(all(not row["issues"] for row in rows))
        for before, after in zip(rows, rows[1:]):
            self.assertLessEqual(before["reference_end"], after["reference_start"])

    def test_missing_repeated_phrase_is_flagged(self):
        rows = map_windows_to_reference("수요 공급입니다. 수요 공급입니다.", windows("수요 공급입니다"))
        self.assertTrue(rows[0]["issues"])

    def test_between_window_omission_flags_both_neighbors(self):
        rows = map_windows_to_reference("가나다라마바. 사아자차카타. 파하거너더러.", windows("가나다라마바", "파하거너더러"))
        self.assertIn("reference_gap_after", rows[0]["issues"])
        self.assertIn("reference_gap_before", rows[1]["issues"])

    def test_omission_with_repeated_suffix_still_fails(self):
        rows = map_windows_to_reference("첫 번째 내용입니다. 빠진 내용입니다. 마지막 내용입니다.", windows("첫 번째 내용입니다", "마지막 내용입니다"))
        self.assertTrue(any("high_cer" in row["issues"] or "reference_gap_after" in row["issues"] for row in rows))

    def test_internal_omission_raises_cer(self):
        rows = map_windows_to_reference("abcdef omitted omitted ghijkl", windows("abcdef ghijkl"))
        self.assertGreater(rows[0]["cer"], 0.25)
        self.assertIn("high_cer", rows[0]["issues"])
        self.assertIn("weak_reference_match", rows[0]["issues"])

    def test_unicode_composition_and_compatibility(self):
        reference = unicodedata.normalize("NFD", "한글입니다") + " ＧＤＰ ① ﬃ café"
        hypothesis = "한글입니다 GDP 1 FFI " + unicodedata.normalize("NFD", "café")
        normalized, offsets = normalize_with_offsets(reference)
        self.assertEqual(normalized, normalize_text(hypothesis))
        self.assertEqual(len(normalized), len(offsets))
        self.assertTrue(all(0 <= start < end <= len(reference) for start, end in offsets))
        row = map_windows_to_reference(reference, windows(hypothesis))[0]
        self.assertEqual(row["issues"], [])
        self.assertEqual(row["reference_text"], reference)

    def test_empty_and_unrelated_inputs(self):
        self.assertEqual(map_windows_to_reference("reference", []), [])
        self.assertIn("empty_reference", map_windows_to_reference("", windows("words"))[0]["issues"])
        self.assertIn("empty_hypothesis", map_windows_to_reference("words", windows("..."))[0]["issues"])
        self.assertIn("unmapped_reference", map_windows_to_reference("abcd", windows("가나다라"))[0]["issues"])
        self.assertIsNone(character_error_rate("", "abc"))

    def test_truncation_duplicate_index_and_bad_times_are_rejected(self):
        source = windows("첫번째 내용입니다", "두번째 내용입니다")
        source[0]["truncated"] = True
        source[1].update(window_index=0, start=29)
        rows = map_windows_to_reference("첫번째 내용입니다 두번째 내용입니다", source)
        self.assertIn("truncated_hypothesis", rows[0]["issues"])
        self.assertTrue(all("duplicate_window_index" in row["issues"] for row in rows))
        self.assertTrue(all("overlapping_or_unordered_windows" in row["issues"] for row in rows))
        source[0]["start"] = float("nan")
        self.assertIn("invalid_window_time", map_windows_to_reference("첫번째 내용입니다", source[:1])[0]["issues"])
        self.assertNotIn("issues", source[0])

    def test_small_error_cer_is_exact(self):
        self.assertAlmostEqual(character_error_rate("abcdef", "abcxef"), 1 / 6)
        row = map_windows_to_reference("abcdef ghijkl", windows("abcxef ghijkl"))[0]
        self.assertEqual(row["issues"], [])
        self.assertAlmostEqual(row["cer"], 1 / 12)

    def test_bit_vector_distance_against_independent_dynamic_program(self):
        def reference_distance(a, b):
            row = list(range(len(b) + 1))
            for i, char in enumerate(a, 1):
                next_row = [i]
                for j, other in enumerate(b, 1):
                    next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (char != other)))
                row = next_row
            return row[-1]
        rng = random.Random(7)
        for _ in range(250):
            a = "".join(rng.choices("가나다abc", k=rng.randrange(35)))
            b = "".join(rng.choices("가나다abc", k=rng.randrange(35)))
            self.assertEqual(_edit_distance(a, b), reference_distance(a, b))


class CropCandidateTests(unittest.TestCase):
    def row(self, reference, start, end, hypothesis, issues=()):
        return dict(text=hypothesis, reference_text=reference[start:end],
                    reference_start=start, reference_end=end, cer=0.9,
                    match_coverage=0.1, issues=list(issues))

    def test_partial_korean_eojeol_is_restored_and_rechecked(self):
        reference = "다음은 경제학입니다."
        original = self.row(reference, reference.index("제"), len(reference),
                            "경제학입니다", ["high_cer", "weak_reference_match", "reference_gap_before"])
        result = prepare_crop_candidate(reference, original)
        self.assertEqual(result["reference_text"], "경제학입니다.")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["coverage_warnings"], ["reference_gap_before"])
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["match_coverage"], 1)
        self.assertEqual(original["issues"], ["high_cer", "weak_reference_match", "reference_gap_before"])
        self.assertEqual(original["reference_text"], "제학입니다.")

    def test_english_both_partial_boundaries_stop_at_punctuation(self):
        reference = "lesson: interesting economics! next"
        original = self.row(reference, reference.index("nteresting"),
                            reference.index("economics") + len("economic"), "INTERESTING economics")
        result = prepare_crop_candidate(reference, original)
        self.assertEqual(result["reference_text"], "interesting economics")
        self.assertEqual(result["issues"], [])
        self.assertEqual(reference[result["reference_start"]:result["reference_end"]], result["reference_text"])

    def test_expansion_over_limit_is_refused_without_partial_change(self):
        reference = "prefixabcdef suffix"
        original = self.row(reference, 6, 12, "abcdef")
        result = prepare_crop_candidate(reference, original, max_boundary_chars=4)
        self.assertIn("partial_word_boundary", result["issues"])
        self.assertEqual(result["reference_start"], 6)
        self.assertEqual(result["reference_end"], 12)
        self.assertEqual(result["reference_text"], "abcdef")

    def test_source_gaps_become_coverage_warnings_only_for_mapped_rows(self):
        reference = "Earlier material. Selected segment. Later material."
        mapped = map_windows_to_reference(reference, windows("selected segment"))[0]
        result = prepare_crop_candidate(reference, mapped)
        self.assertEqual(result["issues"], [])
        self.assertEqual(set(result["coverage_warnings"]), {"unmapped_reference_prefix", "unmapped_reference_suffix"})
        missing = dict(text="", reference_text="", reference_start=None,
                       reference_end=None, issues=["reference_gap_before"])
        result = prepare_crop_candidate(reference, missing)
        self.assertIn("reference_gap_before", result["issues"])
        self.assertIn("unmapped_reference", result["issues"])
        self.assertEqual(result["coverage_warnings"], [])

    def test_internal_omission_remains_fatal(self):
        reference = "abcdef omitted omitted ghijkl"
        mapped = map_windows_to_reference(reference, windows("abcdef ghijkl"))[0]
        result = prepare_crop_candidate(reference, mapped)
        self.assertIn("high_cer", result["issues"])
        self.assertIn("weak_reference_match", result["issues"])

    def test_combining_hangul_boundary_restoration(self):
        reference = unicodedata.normalize("NFD", "한글입니다")
        original = self.row(reference, 1, len(reference), "한글입니다")
        result = prepare_crop_candidate(reference, original)
        self.assertEqual(result["reference_text"], reference)
        self.assertEqual(result["issues"], [])

    def test_unrelated_fatal_codes_and_stale_offsets_are_preserved(self):
        reference = "complete word"
        original = self.row(reference, 0, len(reference), reference,
                            ["truncated_hypothesis", "overlapping_reference_span"])
        result = prepare_crop_candidate(reference, original)
        self.assertEqual(result["issues"], original["issues"])
        original["reference_text"] = "different text"
        original["issues"].append("reference_gap_after")
        result = prepare_crop_candidate(reference, original)
        self.assertIn("reference_source_mismatch", result["issues"])
        self.assertIn("reference_gap_after", result["issues"])
        self.assertEqual(result["coverage_warnings"], [])


class AlignmentAuditTests(unittest.TestCase):
    def test_valid_objects_and_punctuation(self):
        items = [SimpleNamespace(text="한글", start_time=0.3, end_time=1.0),
                 SimpleNamespace(text="GDP", start_time=1.1, end_time=2.0)]
        self.assertEqual(audit_alignment(items, "한글, gdp!", 3), [])

    def test_nonfinite_zero_reversed_and_bounds(self):
        items = [dict(text="a", start_time=-0.1, end_time=0.2),
                 dict(text="b", start_time=0.3, end_time=0.3),
                 dict(text="c", start_time=0.8, end_time=0.4),
                 dict(text="d", start_time=float("nan"), end_time=1.0),
                 dict(text="e", start_time=2, end_time=5)]
        issues = audit_alignment(items, "abcde", 3)
        for code in ["alignment_out_of_bounds", "zero_duration_alignment_token",
                     "reversed_alignment_span", "nonfinite_alignment_timestamp"]:
            self.assertIn(code, issues)

    def test_overlap_order_text_mismatch(self):
        items = [dict(text="first", start_time=1, end_time=2),
                 dict(text="second", start_time=0.5, end_time=1.5)]
        issues = audit_alignment(items, "first missing second", 3)
        self.assertIn("nonmonotonic_alignment", issues)
        self.assertIn("overlapping_alignment_spans", issues)
        self.assertIn("alignment_text_mismatch", issues)

    def test_empty_alignment_and_bad_duration(self):
        self.assertIn("empty_alignment", audit_alignment([], "text", 3))
        self.assertIn("empty_alignment_reference", audit_alignment([], "", 3))
        self.assertIn("invalid_audio_duration", audit_alignment([], "text", float("inf")))

    def test_extreme_speech_rate_and_large_internal_gap(self):
        self.assertIn("implausible_alignment_speech_rate", audit_alignment(
            [dict(text="abcdefghij", start_time=0.0, end_time=0.1)], "abcdefghij", 1))
        self.assertIn("long_internal_alignment_gap", audit_alignment(
            [dict(text="a", start_time=0, end_time=1), dict(text="b", start_time=20, end_time=21)], "ab", 30))


class BoundaryReviewTests(unittest.TestCase):
    def fixture(self, count=10, zero_indices=(4,)):
        items = [dict(text=f"word{i}", start_time=0.5+i*0.5,
                      end_time=0.5+i*0.5+(0 if i in zero_indices else 0.3))
                 for i in range(count)]
        return items, " ".join(item["text"] for item in items), count*0.5+1

    def test_isolated_zero_is_review_eligible_and_still_strict_failure(self):
        items, text, duration = self.fixture()
        self.assertTrue(boundary_review_eligible(items, text, duration, 0.15, 0.85))
        self.assertEqual(audit_alignment(items, text, duration), ["zero_duration_alignment_token"])
        objects = [SimpleNamespace(**item) for item in items]
        self.assertTrue(boundary_review_eligible(objects, text, duration, 0.1, 0.9))

    def test_clean_alignment_is_not_this_special_review_queue(self):
        items, text, duration = self.fixture(zero_indices=())
        self.assertFalse(boundary_review_eligible(items, text, duration, 0.1, 0.9))

    def test_rejects_zero_runs_even_under_fraction_limit(self):
        items, text, duration = self.fixture(count=20, zero_indices=(4, 5))
        self.assertFalse(boundary_review_eligible(items, text, duration, 0.1, 0.9))

    def test_rejects_too_many_zero_tokens_and_short_outputs(self):
        for count, zero_indices in ((10, (3, 6)), (9, (4,))):
            items, text, duration = self.fixture(count=count, zero_indices=zero_indices)
            self.assertFalse(boundary_review_eligible(items, text, duration, 0.1, 0.9))

    def test_rejects_zero_boundary_tokens(self):
        for zero_indices in ((0,), (9,)):
            items, text, duration = self.fixture(zero_indices=zero_indices)
            self.assertFalse(boundary_review_eligible(items, text, duration, 0.1, 0.9))

    def test_rejects_weak_or_nonfinite_agreement(self):
        items, text, duration = self.fixture()
        for cer, coverage in ((0.151, 0.9), (0.1, 0.849), (-0.01, 0.9),
                              (float("nan"), 0.9), (0.1, float("inf")), (False, 0.9)):
            self.assertFalse(boundary_review_eligible(items, text, duration, cer, coverage))

    def test_rejects_unreliable_context_boundaries(self):
        items, text, duration = self.fixture()
        self.assertFalse(boundary_review_eligible(items, text, items[-1]["end_time"]+0.04, 0.1, 0.9))
        items[0].update(start_time=0.04, end_time=0.34)
        self.assertFalse(boundary_review_eligible(items, text, duration, 0.1, 0.9))
        items[0].update(start_time=0.08, end_time=0.38)
        self.assertTrue(boundary_review_eligible(items, text, items[-1]["end_time"]+0.08, 0.1, 0.9))

    def test_rejects_long_positive_span_and_other_audit_failures(self):
        items, text, duration = self.fixture()
        self.assertFalse(boundary_review_eligible(items, text+" omitted", duration, 0.1, 0.9))
        for index, item in enumerate(items):
            item["start_time"] = 0.5+index*4
            item["end_time"] = item["start_time"]+(0 if index == 4 else 0.3)
        items[0]["end_time"] = items[0]["start_time"]+3.1
        duration = items[-1]["end_time"]+1
        self.assertEqual(audit_alignment(items, text, duration), ["zero_duration_alignment_token"])
        self.assertFalse(boundary_review_eligible(items, text, duration, 0.1, 0.9))


if __name__ == "__main__":
    unittest.main()
