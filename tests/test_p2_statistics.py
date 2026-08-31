"""Tests for the seed-paired statistics that gate every dendrite claim.

The August 30 effect audit's finding was that single-seed differences were being
read as wins.  These tests pin the arithmetic (against published Student-t
values), the pairing rules, and — most importantly — that a comparison with
fewer than three paired seeds can never come back claimable, whatever the
difference between the two arms looks like.
"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from dendritic_benchmark.artifacts import (
    finalize_artifact_manifest,
    write_artifact_manifest,
)
from dendritic_benchmark.results import (
    _write_effect_statistics,
    dendrite_effect_estimates,
    write_comparison_reports,
)
from dendritic_benchmark.statistics import (
    MINIMUM_PAIRED_SEEDS,
    SIGNIFICANCE_LEVEL,
    VERDICT_CONTRADICTED,
    VERDICT_INSUFFICIENT_SEEDS,
    VERDICT_SUPPORTED,
    VERDICT_WITHIN_NOISE,
    SeedObservation,
    dense_control_key,
    estimate_all_effects,
    estimate_effect,
    group_seed_observations,
    paired_seed_values,
    two_sided_t_p_value,
)
from dendritic_benchmark.specs import MetricDirection
from dendritic_benchmark.training import QUANTIZATION_EVALUATION_REVISION


def _observations(values: dict[int, float]) -> list[SeedObservation]:
    return [SeedObservation(seed=seed, metric_value=value) for seed, value in values.items()]


def _estimate(
    baseline: dict[int, float],
    treatment: dict[int, float],
    direction: MetricDirection = "maximize",
):
    return estimate_effect(
        model_key="lenet5",
        condition_key="dendrites_fp32",
        baseline_condition_key="base_fp32",
        metric_name="Accuracy",
        metric_direction=direction,
        baseline=_observations(baseline),
        treatment=_observations(treatment),
    )


def _seal_record(
    condition_dir: Path,
    *,
    model_key: str,
    condition_key: str,
    seed: int,
    metric_value: float,
) -> dict[str, object]:
    artifact_id = f"{model_key}-{condition_key}-{seed}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.nn.Linear(2, 1).state_dict(), condition_dir / "model.pt")
    (condition_dir / "metrics.json").write_text(json.dumps({"seed": seed}))
    (condition_dir / "history.csv").write_text("epoch\n1\n")
    record: dict[str, object] = {
        "artifact_id": artifact_id,
        "artifact_dir": str(condition_dir),
        "model_key": model_key,
        "condition_key": condition_key,
        "metric_name": "Accuracy",
        "metric_value": metric_value,
        "metric_direction": "maximize",
        "best_metric_value": metric_value,
        "best_epoch": 1,
        "param_count": 1000,
        "nonzero_params": 1000,
        "file_size_mb": 0.1,
        "train_seconds": 1.0,
        "dendrite_audit_status": (
            "verified_retained" if condition_key.startswith("dendrites_") else "not_applicable"
        ),
    }
    (condition_dir / "record.json").write_text(json.dumps(record))
    with (condition_dir / "record.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    (condition_dir / "best_model_stats.csv").write_text("metric_value\n0.5\n")
    write_artifact_manifest(
        condition_dir,
        artifact_id=artifact_id,
        identity={
            "model_key": model_key,
            "condition_key": condition_key,
            "seed": seed,
            "quantization_evaluation_revision": QUANTIZATION_EVALUATION_REVISION,
        },
        pai_save_name=f"{model_key}_{condition_key}",
        validity={
            "dendrite_status": record["dendrite_audit_status"],
            "quantization_status": "not_applicable",
        },
    )
    finalize_artifact_manifest(condition_dir, artifact_id=artifact_id)
    return record


class StudentTTests(unittest.TestCase):
    def test_p_values_match_published_t_table_entries(self) -> None:
        # Two-sided critical values: t(4)=2.776445 at 0.05, t(9)=2.262157 at
        # 0.05, t(2)=4.302653 at 0.05.
        for t_statistic, df in ((2.776445, 4), (2.262157, 9), (4.302653, 2)):
            with self.subTest(t=t_statistic, df=df):
                self.assertAlmostEqual(
                    two_sided_t_p_value(t_statistic, df), 0.05, places=5
                )
        self.assertAlmostEqual(two_sided_t_p_value(2.0, 10), 0.073388, places=5)
        self.assertEqual(two_sided_t_p_value(0.0, 3), 1.0)
        self.assertEqual(two_sided_t_p_value(math.inf, 3), 0.0)

    def test_p_value_shrinks_as_evidence_grows(self) -> None:
        previous = two_sided_t_p_value(2.5, 2)
        for df in range(3, 12):
            current = two_sided_t_p_value(2.5, df)
            self.assertLess(current, previous)
            previous = current

    def test_a_t_test_needs_a_degree_of_freedom(self) -> None:
        with self.assertRaises(ValueError):
            two_sided_t_p_value(1.0, 0)


class EffectEstimateTests(unittest.TestCase):
    def test_fewer_than_three_paired_seeds_is_never_claimable(self) -> None:
        # A single seed showing a landslide is still not evidence.
        for seeds in ({0: 0.10}, {0: 0.10, 1: 0.11}):
            with self.subTest(seed_count=len(seeds)):
                baseline = {seed: 0.50 for seed in seeds}
                treatment = {seed: 0.50 + value for seed, value in seeds.items()}
                estimate = _estimate(baseline, treatment)
                self.assertEqual(estimate.verdict, VERDICT_INSUFFICIENT_SEEDS)
                self.assertFalse(estimate.claimable)
                self.assertIsNone(estimate.p_value)
                self.assertIn(str(MINIMUM_PAIRED_SEEDS), estimate.reason)

    def test_three_consistent_seeds_support_a_maximized_metric(self) -> None:
        estimate = _estimate(
            {0: 0.900, 1: 0.902, 2: 0.898},
            {0: 0.930, 1: 0.933, 2: 0.928},
        )
        self.assertEqual(estimate.seed_count, 3)
        self.assertEqual(estimate.verdict, VERDICT_SUPPORTED)
        self.assertTrue(estimate.claimable)
        self.assertAlmostEqual(estimate.mean_improvement, 0.0303333, places=6)
        self.assertLess(estimate.p_value or 1.0, SIGNIFICANCE_LEVEL)
        self.assertTrue(estimate.exceeds_noise_floor)

    def test_improvement_is_signed_by_the_metric_direction(self) -> None:
        # For MAE, lower is better, so the same raw numbers flip their meaning.
        lower_is_better = _estimate(
            {0: 0.400, 1: 0.402, 2: 0.398},
            {0: 0.370, 1: 0.373, 2: 0.368},
            direction="minimize",
        )
        self.assertGreater(lower_is_better.mean_improvement, 0)
        self.assertEqual(lower_is_better.verdict, VERDICT_SUPPORTED)

        higher_is_better = _estimate(
            {0: 0.400, 1: 0.402, 2: 0.398},
            {0: 0.370, 1: 0.373, 2: 0.368},
            direction="maximize",
        )
        self.assertLess(higher_is_better.mean_improvement, 0)
        self.assertEqual(higher_is_better.verdict, VERDICT_CONTRADICTED)

    def test_a_difference_inside_the_seed_spread_is_within_noise(self) -> None:
        estimate = _estimate(
            {0: 0.900, 1: 0.870, 2: 0.930},
            {0: 0.905, 1: 0.845, 2: 0.945},
        )
        self.assertEqual(estimate.verdict, VERDICT_WITHIN_NOISE)
        self.assertFalse(estimate.claimable)
        self.assertGreater(estimate.noise_floor, 0.0)
        self.assertFalse(estimate.exceeds_noise_floor)

    def test_identical_arms_are_not_significant_and_never_divide_by_zero(self) -> None:
        estimate = _estimate({0: 0.5, 1: 0.5, 2: 0.5}, {0: 0.5, 1: 0.5, 2: 0.5})
        self.assertEqual(estimate.t_statistic, 0.0)
        self.assertEqual(estimate.p_value, 1.0)
        self.assertEqual(estimate.verdict, VERDICT_WITHIN_NOISE)
        self.assertEqual(estimate.noise_floor, 0.0)

    def test_pairing_drops_seeds_only_one_arm_completed(self) -> None:
        seeds, baseline_values, treatment_values = paired_seed_values(
            _observations({0: 0.1, 1: 0.2, 3: 0.4}),
            _observations({1: 0.9, 2: 0.8, 3: 0.7}),
        )
        self.assertEqual(seeds, (1, 3))
        self.assertEqual(baseline_values, [0.2, 0.4])
        self.assertEqual(treatment_values, [0.9, 0.7])

        # Three arms each, but only two seeds overlap: still not claimable.
        estimate = _estimate({0: 0.5, 1: 0.5, 2: 0.5}, {1: 0.9, 2: 0.9, 5: 0.9})
        self.assertEqual(estimate.paired_seeds, (1, 2))
        self.assertEqual(estimate.verdict, VERDICT_INSUFFICIENT_SEEDS)

    def test_every_dendritic_condition_pairs_with_its_own_dense_control(self) -> None:
        self.assertEqual(dense_control_key("dendrites_q1_58"), "base_q1_58")
        self.assertEqual(dense_control_key("dendrites_fp32"), "base_fp32")
        self.assertIsNone(dense_control_key("base_q8"))

        grouped = group_seed_observations(
            [
                ("lenet5", "base_q8", SeedObservation(seed, 0.80))
                for seed in range(3)
            ]
            + [
                ("lenet5", "dendrites_q8", SeedObservation(seed, 0.85))
                for seed in range(3)
            ]
            + [("lenet5", "base_fp32", SeedObservation(0, 0.9))]
        )
        estimates = estimate_all_effects(grouped)
        self.assertEqual(len(estimates), 1)
        self.assertEqual(estimates[0].condition_key, "dendrites_q8")
        self.assertEqual(estimates[0].baseline_condition_key, "base_q8")
        self.assertEqual(estimates[0].verdict, VERDICT_SUPPORTED)

    def test_an_arm_without_a_control_produces_no_estimate(self) -> None:
        grouped = group_seed_observations(
            [
                ("lenet5", "dendrites_fp32", SeedObservation(seed, 0.9))
                for seed in range(3)
            ]
        )
        self.assertEqual(estimate_all_effects(grouped), [])


class EffectReportingTests(unittest.TestCase):
    def test_records_without_a_manifest_seed_cannot_enter_a_claim(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            records = []
            for seed in range(3):
                for condition_key, value in (
                    ("base_fp32", 0.90),
                    ("dendrites_fp32", 0.93),
                ):
                    records.append(
                        _seal_record(
                            root_path / f"seed{seed}" / condition_key,
                            model_key="lenet5",
                            condition_key=condition_key,
                            seed=seed,
                            metric_value=value + seed * 0.001,
                        )
                    )
            estimates = dendrite_effect_estimates(records)
            self.assertEqual(len(estimates), 1)
            self.assertEqual(estimates[0].seed_count, 3)
            self.assertEqual(estimates[0].verdict, VERDICT_SUPPORTED)

            # An arm whose artifact no longer validates drops out of the
            # statistics entirely rather than contributing a stale number.
            tampered = root_path / "seed2" / "dendrites_fp32" / "history.csv"
            tampered.write_text("epoch\n1\n2\n")
            degraded = dendrite_effect_estimates(records)
            self.assertEqual(degraded[0].seed_count, 2)
            self.assertEqual(degraded[0].verdict, VERDICT_INSUFFICIENT_SEEDS)

    def test_the_statistics_csv_carries_the_evidence_a_claim_needs(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            records = [
                _seal_record(
                    root_path / f"seed{seed}" / condition_key,
                    model_key="lenet5",
                    condition_key=condition_key,
                    seed=seed,
                    metric_value=value,
                )
                for seed in range(3)
                for condition_key, value in (
                    ("base_fp32", 0.90 + seed * 0.002),
                    ("dendrites_fp32", 0.93 + seed * 0.002),
                )
            ]
            output_dir = root_path / "comparison"
            output_dir.mkdir()
            _write_effect_statistics(records, output_dir)
            with (output_dir / "dendrite_effect_statistics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            for column in (
                "paired_seeds",
                "seed_count",
                "noise_floor",
                "mean_improvement",
                "p_value",
                "verdict",
                "reason",
            ):
                self.assertIn(column, row)
            self.assertEqual(row["paired_seeds"], "0,1,2")
            self.assertEqual(row["seed_count"], "3")
            self.assertEqual(row["verdict"], VERDICT_SUPPORTED)

    def test_seed_roots_are_what_let_a_claim_reach_three_paired_seeds(self) -> None:
        """One results root is one seed; the others have to be passed in."""
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            per_seed: list[list[dict[str, object]]] = []
            for seed in range(3):
                per_seed.append(
                    [
                        _seal_record(
                            root_path / f"seed_{seed}" / "results" / "lenet5" / key,
                            model_key="lenet5",
                            condition_key=key,
                            seed=seed,
                            metric_value=value + seed * 0.002,
                        )
                        for key, value in (("base_fp32", 0.90), ("dendrites_fp32", 0.93))
                    ]
                )
            output_dir = root_path / "comparison"
            output_dir.mkdir()

            # This run's root alone cannot reach the seed minimum.
            alone = _write_effect_statistics(per_seed[0], output_dir)
            self.assertEqual(alone[0].verdict, VERDICT_INSUFFICIENT_SEEDS)

            write_comparison_reports(
                per_seed[0],
                output_dir,
                model_keys=["lenet5"],
                seed_records=[*per_seed[1], *per_seed[2]],
            )
            with (output_dir / "dendrite_effect_statistics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["seed_count"], str(MINIMUM_PAIRED_SEEDS))
            self.assertEqual(rows[0]["verdict"], VERDICT_SUPPORTED)

            # The audit CSV of this run carries the pooled verdict too.
            with (output_dir / "dendrite_audit.csv").open() as handle:
                audit_rows = list(csv.DictReader(handle))
            dendritic = [
                row for row in audit_rows if row["condition_key"] == "dendrites_fp32"
            ]
            self.assertEqual(len(dendritic), 1)
            self.assertEqual(dendritic[0]["paired_seed_count"], "3")
            self.assertEqual(dendritic[0]["effect_verdict"], VERDICT_SUPPORTED)

    def test_an_empty_comparison_writes_an_empty_table_not_a_claim(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            output_dir = Path(root)
            self.assertEqual(_write_effect_statistics([], output_dir), [])
            self.assertEqual(
                (output_dir / "dendrite_effect_statistics.csv").read_text(), ""
            )


if __name__ == "__main__":
    unittest.main()
