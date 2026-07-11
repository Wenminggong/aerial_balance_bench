#!/usr/bin/env python3
"""Summarize evaluation summary.csv files with pooled benchmark statistics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


TRACKING_METRIC_SUFFIXES = (
    "mean_absolute_error",
    "root_mean_square_error",
    "maximum_absolute_error",
)

POOLED_METRIC_SUFFIXES = (
    "steady_state_error",
    "convergence_time",
    "climbing_time",
)

WEIGHT_COLUMNS = (
    "benchmark_completed_episodes",
    "completed_episodes",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute aggregate mean/std statistics from an evaluation summary.csv. "
            "Metrics with per-row std columns are combined using pooled variance."
        )
    )
    parser.add_argument("summary_csv", type=str, help="Path to the evaluation summary.csv file.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Defaults to summary_statistics.csv next to the input file.",
    )
    return parser.parse_args()


def _to_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _population_mean_std(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(max(variance, 0.0))


def _weighted_mean_std(values: list[float], weights: list[float]) -> tuple[float, float, float]:
    samples = [
        (value, weight)
        for value, weight in zip(values, weights, strict=True)
        if math.isfinite(value) and math.isfinite(weight) and weight > 0.0
    ]
    total_weight = sum(weight for _, weight in samples)
    if total_weight <= 0.0:
        return math.nan, math.nan, 0.0
    mean = sum(value * weight for value, weight in samples) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in samples) / total_weight
    return mean, math.sqrt(max(variance, 0.0)), total_weight


def _select_weight_column(fieldnames: list[str]) -> str | None:
    for column in WEIGHT_COLUMNS:
        if column in fieldnames:
            return column
    return None


def _get_weights(rows: list[dict[str, str]], fieldnames: list[str]) -> tuple[list[float], str]:
    weight_column = _select_weight_column(fieldnames)
    if weight_column is None:
        print(
            "[WARN] Neither benchmark_completed_episodes nor completed_episodes was found. "
            "Using weight=1 for every row.",
            file=sys.stderr,
        )
        return [1.0] * len(rows), "unit_weight"

    weights = []
    invalid_count = 0
    for row in rows:
        weight = _to_float(row.get(weight_column))
        if not math.isfinite(weight) or weight <= 0.0:
            invalid_count += 1
            weight = math.nan
        weights.append(weight)
    if invalid_count:
        print(
            f"[WARN] Ignoring {invalid_count} row(s) with invalid {weight_column} values "
            "for pooled metrics.",
            file=sys.stderr,
        )
    return weights, weight_column


def _metric_weight_column(metric: str, fieldnames: list[str]) -> str | None:
    if not metric.startswith("benchmark_"):
        return _select_weight_column(fieldnames)
    metric_name = metric.removeprefix("benchmark_")
    for suffix in (*TRACKING_METRIC_SUFFIXES, "success_rate", *POOLED_METRIC_SUFFIXES):
        marker = f"_{suffix}"
        if metric_name.endswith(marker):
            prefix = metric_name[: -len(marker)]
            candidate = f"benchmark_{prefix}_completed_episodes"
            if candidate in fieldnames:
                return candidate
    return _select_weight_column(fieldnames)


def _weights_for_column(rows: list[dict[str, str]], fieldnames: list[str], metric: str) -> tuple[list[float], str]:
    weight_column = _metric_weight_column(metric, fieldnames)
    if weight_column is None:
        return [1.0] * len(rows), "unit_weight"
    return [_to_float(row.get(weight_column)) for row in rows], weight_column


def _pooled_mean_std(
    rows: list[dict[str, str]],
    weights: list[float],
    mean_column: str,
    std_column: str,
) -> tuple[float, float, float]:
    samples: list[tuple[float, float, float]] = []
    for row, weight in zip(rows, weights, strict=True):
        mean = _to_float(row.get(mean_column))
        std = _to_float(row.get(std_column))
        if math.isfinite(weight) and math.isfinite(mean) and math.isfinite(std):
            samples.append((weight, mean, std))

    total_weight = sum(weight for weight, _, _ in samples)
    if total_weight <= 0.0:
        return math.nan, math.nan, 0.0

    pooled_mean = sum(weight * mean for weight, mean, _ in samples) / total_weight
    pooled_variance = (
        sum(weight * (std**2 + (mean - pooled_mean) ** 2) for weight, mean, std in samples) / total_weight
    )
    return pooled_mean, math.sqrt(max(pooled_variance, 0.0)), total_weight


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")
    return rows, list(reader.fieldnames)


def _available_metric_columns(fieldnames: list[str]) -> tuple[list[str], dict[str, str]]:
    scalar_metrics = []
    pooled_metrics = {}
    for column in fieldnames:
        is_benchmark = column.startswith("benchmark_")
        if column == "policy_compute_time_mean" or is_benchmark and (
            column.endswith("_success_rate") or column.endswith(TRACKING_METRIC_SUFFIXES)
        ):
            scalar_metrics.append(column)
        if is_benchmark and column.endswith(POOLED_METRIC_SUFFIXES):
            std_column = f"{column}_std"
            if std_column in fieldnames:
                pooled_metrics[column] = std_column
    return scalar_metrics, pooled_metrics


def _write_summary(output_path: Path, summary: dict[str, float | int | str]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _print_summary(summary: dict[str, float | int | str]):
    print("[INFO] Aggregate statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def summarize(summary_csv: Path, output_path: Path) -> dict[str, float | int | str]:
    rows, fieldnames = _read_rows(summary_csv)
    weights, weight_column = _get_weights(rows, fieldnames)
    scalar_metrics, pooled_metrics = _available_metric_columns(fieldnames)
    if not scalar_metrics and not pooled_metrics:
        raise ValueError("No supported target-position or tracking metric columns were found.")

    total_completed_episodes = sum(weight for weight in weights if math.isfinite(weight) and weight > 0.0)
    summary: dict[str, float | int | str] = {
        "source_csv": str(summary_csv),
        "num_rows": len(rows),
        "weight_column": weight_column,
        "total_completed_episodes": total_completed_episodes,
    }

    for metric in scalar_metrics:
        values = [_to_float(row.get(metric)) for row in rows]
        if metric == "policy_compute_time_mean":
            mean, std = _population_mean_std(values)
            metric_weight = float(len([value for value in values if math.isfinite(value)]))
        else:
            metric_weights, metric_weight_column = _weights_for_column(rows, fieldnames, metric)
            mean, std, metric_weight = _weighted_mean_std(values, metric_weights)
            summary[f"{metric}_weight_column"] = metric_weight_column
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"{metric}_total_weight"] = metric_weight

    for mean_column, std_column in pooled_metrics.items():
        metric_weights, metric_weight_column = _weights_for_column(rows, fieldnames, mean_column)
        mean, std, metric_weight = _pooled_mean_std(rows, metric_weights, mean_column, std_column)
        summary[f"{mean_column}_mean"] = mean
        summary[f"{mean_column}_std"] = std
        summary[f"{mean_column}_total_episodes"] = metric_weight
        summary[f"{mean_column}_weight_column"] = metric_weight_column

    _write_summary(output_path, summary)
    return summary


def main():
    args = _parse_args()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else summary_csv.parent / "summary_statistics.csv"
    )

    try:
        summary = summarize(summary_csv, output_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    _print_summary(summary)
    print(f"[INFO] Saved aggregate statistics to: {output_path}")


if __name__ == "__main__":
    main()
