"""Render reproducible v1.5 results without overwriting the manual report text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig


START_MARKER = "<!-- AUTO_RESULTS_START -->"
END_MARKER = "<!-- AUTO_RESULTS_END -->"


def update_report(config: AppConfig) -> Path:
    summary_path = config.paths.results_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}; run evaluate and test first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_path = config.paths.report_path
    text = report_path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError(f"Report markers are missing from {report_path}")
    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    generated = _render(summary)
    report_path.write_text(
        before + START_MARKER + "\n\n" + generated + "\n\n" + END_MARKER + after,
        encoding="utf-8",
    )
    return report_path


def _render(summary: dict[str, Any]) -> str:
    training = summary["training"]
    threshold = training["deadzone_threshold_mps2"]
    lines = [
        "## 自动生成的实验结果",
        "",
        "> 本区块由 `python -m residual_dynamics_v1_5.cli report` 生成。",
        "",
        f"- Run：`{summary['run_name']}`",
        f"- 配置哈希：`{summary['config_hash']}`",
        f"- fit dataset 哈希：`{summary['fit_dataset_hash']}`",
        f"- retrospective test dataset 哈希：`{summary['test_dataset_hash']}`",
        f"- seed：`{summary['seed']}`；设备：`{training['training_device']}`",
        f"- 设备选择原因：`{training['device_benchmark']['reason']}`",
        f"- 动态 MLP 最佳 epoch：`{training['dynamic']['best_epoch']}`；"
        f"死区最佳 epoch：`{training['deadzone']['best_epoch']}`",
        f"- 拟合静止阈值：`{threshold:.6f} m/s²`",
        f"- 总训练时间：`{training['total_training_time_s']:.1f} s`",
        f"- W&B：`{training['wandb']['actual_mode']}`，run ID："
        f"`{training['wandb']['run_id'] or 'N/A'}`",
        f"- W&B 初始化说明：`{training['wandb']['initialization_error'] or 'none'}`",
        f"- v1.5 checkpoint SHA-256：`{summary['validation']['model_hashes']['v1_5_checkpoint_sha256']}`",
        f"- 固定 v1.0 权重 SHA-256：`{summary['validation']['model_hashes']['v1_weights_sha256']}`",
        f"- 总体验收：`{'PASS' if summary['acceptance']['passed'] else 'FAIL'}`",
        "",
        "### 动态样本逐点残差",
        "",
        "| 数据 | 模型 | RMSE (m/s²) | MAE | Bias | R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope in ("validation", "test", "stress"):
        for model in ("nominal", "v1", "dynamic_mlp", "v1_5"):
            metric = summary[scope]["point"]["dynamic"][model]
            lines.append(
                f"| {scope} | {model} | {_fmt(metric['rmse'])} | {_fmt(metric['mae'])} | "
                f"{_fmt(metric['bias'])} | {_fmt(metric['r2'])} |"
            )
    lines += [
        "",
        "### 多步 rollout",
        "",
        "| 数据 | Horizon | 模型 | 位置 RMSE (m) | 速度 RMSE (m/s) | 位置 MSE 改善 | 速度 MSE 改善 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for scope in ("validation", "test", "stress"):
        rollout = summary[scope]["rollout"]["overall"]
        for horizon in sorted(rollout, key=lambda value: int(value[1:])):
            entry = rollout[horizon]
            for model in ("nominal", "v1", "dynamic_mlp", "v1_5"):
                metric = entry["models"][model]
                lines.append(
                    f"| {scope} | {entry['horizon_s']:.3f}s | {model} | "
                    f"{_fmt(metric['pb_rmse_m'])} | {_fmt(metric['vb_rmse_mps'])} | "
                    f"{_pct(metric.get('pb_mse_improvement'))} | {_pct(metric.get('vb_mse_improvement'))} |"
                )
    lines += [
        "",
        "### 静止诊断",
        "",
        "| 数据 | 模型 | Median | P95 | 虚假运动率 |",
        "|---|---|---:|---:|---:|",
    ]
    for scope in ("validation", "test", "stress"):
        for model in ("nominal", "v1", "dynamic_mlp", "v1_5"):
            metric = summary[scope]["static"][model]
            lines.append(
                f"| {scope} | {model} | {_fmt(metric['median_abs_mps2'])} | "
                f"{_fmt(metric['p95_abs_mps2'])} | {_pct(metric['false_motion_rate'])} |"
            )
    lines += ["", "### 验收条件", ""]
    for name, passed in summary["acceptance"]["checks"].items():
        lines.append(f"- `{'PASS' if passed else 'FAIL'}` — `{name}`")
    lines += [
        "",
        "### 诊断边界",
        "",
        "| 数据 | OOD比例 | MLP饱和率 | 死区激活比例 | 梯度范数P99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope in ("validation", "test", "stress"):
        diag = summary[scope]["diagnostics"]
        lines.append(
            f"| {scope} | {_pct(diag['ood_fraction'])} | {_pct(diag['mlp_saturation_rate'])} | "
            f"{_pct(diag['deadzone_suppression_active_fraction'])} | {_fmt(diag['gradient_norm_p99'])} |"
        )
    lines += [
        "",
        "08-17 数据是回顾性外部测试：它没有参与权重、归一化或死区参数拟合，"
        "但此前已参与方案分析，因此不能视作完全未见的 prospective test。",
    ]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"
