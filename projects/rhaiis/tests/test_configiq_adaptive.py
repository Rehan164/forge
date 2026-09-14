import math
import unittest

from projects.rhaiis.orchestration.configiq_adaptive import (
    GuideLLMSaturationResult,
    KneeResult,
    analyze_guidellm_report,
    assess_saturation,
    extract_guidellm_saturation,
    find_throughput_knee,
)

CONCURRENCIES = [1, 10, 20, 30, 40, 50, 60]


def _benchmark(
    concurrency: int,
    is_over_saturated: bool | None,
    *,
    constraint_group: str = "scheduler_constraints",
) -> dict:
    scheduler_state = {}
    if is_over_saturated is not None:
        scheduler_state[constraint_group] = {
            "over_saturation": {
                "metadata": {
                    "is_over_saturated": is_over_saturated,
                    "concurrent_slope": 0.25,
                    "concurrent_slope_moe": 0.1,
                    "concurrent_n": 20,
                    "ttft_slope": 0.02,
                    "ttft_slope_moe": 0.01,
                    "ttft_n": 20,
                    "ttft_violations": 2,
                }
            }
        }
    return {
        "config": {"strategy": {"max_concurrency": concurrency}},
        "scheduler_state": scheduler_state,
    }


class ThroughputKneeTests(unittest.TestCase):
    def test_finds_clear_throughput_plateau(self) -> None:
        result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 30, 30, 30])

        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.knee, 30)
        self.assertEqual(result.breakpoint_concurrency, 30)
        self.assertEqual(result.saturation_concurrency, 30)
        self.assertAlmostEqual(result.slope_ratio, 0)

    def test_allows_small_tail_gain(self) -> None:
        result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 31, 32, 33])

        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.knee, 30)
        self.assertAlmostEqual(result.slope_ratio, 0.1, delta=0.02)

    def test_finds_plateau_despite_tail_noise(self) -> None:
        result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 31, 29, 30])

        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.knee, 30.29, places=2)
        self.assertEqual(result.breakpoint_concurrency, 30)
        self.assertEqual(result.saturation_concurrency, 40)

    def test_rejects_linear_curve(self) -> None:
        result = find_throughput_knee(CONCURRENCIES, [1, 10, 20, 30, 40, 50, 60])

        self.assertEqual(result.status, "no_knee")
        self.assertIn("linear", result.reason)

    def test_requires_five_distinct_valid_points(self) -> None:
        result = find_throughput_knee([1, 2, 3, 4], [10, 20, 20, 20])

        self.assertEqual(result.status, "no_knee")
        self.assertIn("at least 5", result.reason)

    def test_averages_duplicate_concurrency_measurements(self) -> None:
        result = find_throughput_knee(
            [1, 10, 20, 30, 30, 40, 50, 60],
            [1, 10, 20, 29, 31, 30, 30, 30],
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.breakpoint_concurrency, 30)

    def test_rejects_mismatched_input_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            find_throughput_knee([1, 2, 3], [10, 20])

    def test_rejects_invalid_measurements(self) -> None:
        cases = [
            ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], "Concurrency"),
            ([1, 2, 3, 4, 5], [1, 2, -1, 4, 5], "Throughput"),
            ([1, 2, 3, 4, 5], [1, 2, math.nan, 4, 5], "Throughput"),
        ]
        for concurrencies, throughputs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    find_throughput_knee(concurrencies, throughputs)

    def test_gemma_short_run_does_not_claim_saturation(self) -> None:
        result = find_throughput_knee(
            [1, 2, 5, 10, 25, 50, 75, 100, 200, 300],
            [
                209.63,
                419.93,
                843.22,
                1411.77,
                2911.77,
                4488.20,
                5400.58,
                6190.11,
                7829.66,
                8328.99,
            ],
        )

        self.assertEqual(result.status, "no_knee")
        self.assertIn("thresholds", result.reason)


class GuideLLMSaturationTests(unittest.TestCase):
    def test_extracts_first_oversaturated_and_previous_safe_concurrency(self) -> None:
        result = extract_guidellm_saturation(
            {
                "benchmarks": [
                    _benchmark(100, True),
                    _benchmark(50, False),
                    _benchmark(200, True),
                    _benchmark(75, False),
                ]
            }
        )

        self.assertEqual(result.status, "detected")
        self.assertEqual(result.previous_safe_concurrency, 75)
        self.assertEqual(result.first_oversaturated_concurrency, 100)
        self.assertEqual([point.concurrency for point in result.points], [50, 75, 100, 200])
        self.assertEqual(result.points[0].ttft_slope, 0.02)

    def test_reports_when_saturation_is_not_detected(self) -> None:
        result = extract_guidellm_saturation(
            {"benchmarks": [_benchmark(1, False), _benchmark(2, False)]}
        )

        self.assertEqual(result.status, "not_detected")
        self.assertIsNone(result.first_oversaturated_concurrency)

    def test_reports_when_detector_metadata_is_unavailable(self) -> None:
        result = extract_guidellm_saturation(
            {"benchmarks": [_benchmark(1, None), _benchmark(2, None)]}
        )

        self.assertEqual(result.status, "unavailable")
        self.assertTrue(all(point.is_over_saturated is None for point in result.points))

    def test_reads_end_processing_constraint_snapshot(self) -> None:
        result = extract_guidellm_saturation(
            {"benchmarks": [_benchmark(100, True, constraint_group="end_processing_constraints")]}
        )

        self.assertEqual(result.status, "detected")
        self.assertEqual(result.first_oversaturated_concurrency, 100)

    def test_rejects_duplicate_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate GuideLLM concurrency"):
            extract_guidellm_saturation(
                {"benchmarks": [_benchmark(50, False), _benchmark(50, True)]}
            )

    def test_rejects_malformed_detector_metadata(self) -> None:
        benchmark = _benchmark(50, False)
        metadata = benchmark["scheduler_state"]["scheduler_constraints"]["over_saturation"][
            "metadata"
        ]
        metadata["is_over_saturated"] = "false"

        with self.assertRaisesRegex(ValueError, "boolean"):
            extract_guidellm_saturation({"benchmarks": [benchmark]})

    def test_rejects_non_monotonic_saturation_states(self) -> None:
        result = extract_guidellm_saturation(
            {
                "benchmarks": [
                    _benchmark(50, False),
                    _benchmark(75, True),
                    _benchmark(100, False),
                ]
            }
        )

        self.assertEqual(result.status, "inconsistent")


class SaturationAssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.knee = KneeResult(status="ok", reason="test", knee=90)

    def test_corroborates_knee_inside_guidellm_boundary(self) -> None:
        result = assess_saturation(
            self.knee,
            GuideLLMSaturationResult(
                status="detected",
                reason="test",
                points=(),
                previous_safe_concurrency=75,
                first_oversaturated_concurrency=100,
            ),
        )

        self.assertEqual(result.status, "corroborated")
        self.assertEqual(result.selection_center, 90)
        self.assertEqual(result.refinement_lower, 75)
        self.assertEqual(result.refinement_upper, 100)

    def test_keeps_throughput_knee_without_guidellm_boundary(self) -> None:
        result = assess_saturation(
            self.knee,
            GuideLLMSaturationResult(status="not_detected", reason="test", points=()),
        )

        self.assertEqual(result.status, "throughput_only")
        self.assertEqual(result.selection_center, 90)

    def test_uses_guidellm_boundary_when_throughput_has_no_knee(self) -> None:
        result = assess_saturation(
            KneeResult(status="no_knee", reason="test"),
            GuideLLMSaturationResult(
                status="detected",
                reason="test",
                points=(),
                previous_safe_concurrency=75,
                first_oversaturated_concurrency=100,
            ),
        )

        self.assertEqual(result.status, "oversaturation_boundary")
        self.assertEqual(result.refinement_lower, 75)
        self.assertEqual(result.refinement_upper, 100)

    def test_rejects_guidellm_boundary_below_throughput_knee(self) -> None:
        result = assess_saturation(
            self.knee,
            GuideLLMSaturationResult(
                status="detected",
                reason="test",
                points=(),
                first_oversaturated_concurrency=75,
            ),
        )

        self.assertEqual(result.status, "disagreement")
        self.assertIsNone(result.selection_center)

    def test_rejects_inconsistent_guidellm_states(self) -> None:
        result = assess_saturation(
            self.knee,
            GuideLLMSaturationResult(
                status="inconsistent",
                reason="test",
                points=(),
                first_oversaturated_concurrency=75,
            ),
        )

        self.assertEqual(result.status, "disagreement")


class ConfigIQReportAnalysisTests(unittest.TestCase):
    def test_combines_throughput_and_guidellm_evidence(self) -> None:
        report = {
            "benchmarks": [
                self._benchmark_with_throughput(1, 1, False),
                self._benchmark_with_throughput(10, 10, False),
                self._benchmark_with_throughput(20, 20, False),
                self._benchmark_with_throughput(30, 30, False),
                self._benchmark_with_throughput(40, 30, True),
                self._benchmark_with_throughput(50, 30, True),
                self._benchmark_with_throughput(60, 30, True),
            ]
        }

        result = analyze_guidellm_report(report)

        self.assertEqual(result.throughput.status, "ok")
        self.assertEqual(result.guidellm.status, "detected")
        self.assertEqual(result.assessment.status, "corroborated")

    def test_requires_output_throughput_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "output throughput"):
            analyze_guidellm_report({"benchmarks": [_benchmark(1, False)]})

    @staticmethod
    def _benchmark_with_throughput(
        concurrency: int,
        throughput: float,
        is_over_saturated: bool,
    ) -> dict:
        benchmark = _benchmark(concurrency, is_over_saturated)
        benchmark["metrics"] = {"output_tokens_per_second": {"successful": {"mean": throughput}}}
        return benchmark


if __name__ == "__main__":
    unittest.main()
