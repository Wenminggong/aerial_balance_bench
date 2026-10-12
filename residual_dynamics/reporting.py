"""Generate the result section of the residual-dynamics Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig


BEGIN_MARKER = "<!-- BEGIN GENERATED RESULTS -->"
END_MARKER = "<!-- END GENERATED RESULTS -->"


def generate_report(config: AppConfig) -> Path:
    summary_path = config.paths.results_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Evaluation summary is missing. Run evaluate first.")
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    screening_path = config.paths.results_dir / "screening_summary.json"
    if screening_path.is_file():
        with screening_path.open("r", encoding="utf-8") as stream:
            summary["screening"] = json.load(stream)
    report_path = config.paths.report_path
    text = report_path.read_text(encoding="utf-8")
    if BEGIN_MARKER not in text or END_MARKER not in text:
        raise ValueError("REPORT.md is missing generated-results markers.")
    generated = _render_results(summary)
    prefix, remainder = text.split(BEGIN_MARKER, maxsplit=1)
    _, suffix = remainder.split(END_MARKER, maxsplit=1)
    report_path.write_text(
        f"{prefix}{BEGIN_MARKER}\n{generated}\n{END_MARKER}{suffix}",
        encoding="utf-8",
    )
    return report_path


def _render_results(summary: dict[str, Any]) -> str:
    training = summary["training"]
    point = summary["point"]["overall"]
    rollout = summary["rollout"]["overall"]
    diagnostics = summary["diagnostics"]
    acceptance = summary["acceptance"]
    screening = summary.get("screening")
    overall_passed = acceptance["passed"] and (
        screening is None or screening["seed_stability_passed"]
    )
    wandb = training["wandb"]
    benchmark = training["device_benchmark"]
    lines = [
        "## 实际训练与验证结果",
        "",
        f"- Run：`{summary['run_name']}`",
        f"- 随机种子：`{summary['seed']}`",
        f"- 最佳 epoch：`{training['best_epoch']}`（停止原因：`{training['stop_reason']}`）",
        f"- 训练设备：`{training['training_device']}`",
        f"- CPU/GPU 基准：{_benchmark_text(benchmark)}",
        f"- 总训练时间：{training['total_training_time_s']:.1f} s",
        f"- W&B 模式：`{wandb['actual_mode']}`；Run ID：`{wandb.get('run_id') or 'N/A'}`",
        f"- W&B 链接：{_wandb_link(wandb)}",
        f"- 当前单模型验收：**{'PASS' if acceptance['passed'] else 'FAIL'}**",
        f"- 总体验收：**{'PASS' if overall_passed else 'FAIL'}**",
    ]
    lines.extend(_render_loss_budget(training))
    lines.extend(
        [
            "",
            "### 逐点加速度残差",
            "",
            "| 模型 | RMSE (m/s²) | MAE (m/s²) | Bias (m/s²) | R² |",
            "| --- | ---: | ---: | ---: | ---: |",
            _point_row("名义模型", point["nominal"]),
            _point_row("五输入 MLP", point["mlp"]),
            "",
            "### 多步自由 rollout",
            "",
            "| Horizon | 模型 | 位置 RMSE (m) | 速度 RMSE (m/s) | 位置 MSE 改善 | 速度 MSE 改善 |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in sorted(rollout, key=lambda item: int(item[1:])):
        value = rollout[key]
        lines.append(
            f"| {value['horizon_s']:.3f} s | 名义模型 | "
            f"{value['nominal']['pb_rmse_m']:.6f} | "
            f"{value['nominal']['vb_rmse_mps']:.6f} | — | — |"
        )
        lines.append(
            f"| {value['horizon_s']:.3f} s | 五输入 MLP | "
            f"{value['mlp']['pb_rmse_m']:.6f} | "
            f"{value['mlp']['vb_rmse_mps']:.6f} | "
            f"{100.0 * value['improvement']['pb_mse_fraction']:.1f}% | "
            f"{100.0 * value['improvement']['vb_mse_fraction']:.1f}% |"
        )
    if summary.get("training_variant") == "causal_rollout":
        lines.extend(
            [
                "",
                "### SG 辅助目标与响应链诊断",
                "",
                "| Horizon | SG目标 MLP位置 RMSE | SG目标 MLP速度 RMSE | theta RMSE / bias | vrz RMSE / bias |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in sorted(rollout, key=lambda item: int(item[1:])):
            value = rollout[key]
            smooth = value["smooth"]["mlp"]
            response = value["response_chain"]
            lines.append(
                f"| {value['horizon_s']:.3f} s | {smooth['pb_rmse_m']:.6f} | "
                f"{smooth['vb_rmse_mps']:.6f} | {response['theta_rmse_rad']:.6f} / "
                f"{response['theta_bias_rad']:.6f} rad | {response['vrz_rmse_mps']:.6f} / "
                f"{response['vrz_bias_mps']:.6f} m/s |"
            )
    if screening is not None:
        lines.extend(_render_screening(screening))
    if summary.get("schema_version") == 3:
        lines.extend(
            [
                "",
                "### 独立 split 验收",
                "",
                "| Split | trial 数 | h33 pb/vb RMSE | 验收 |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for split, split_summary in summary["splits"].items():
            h33 = split_summary["rollout"]["overall"]["h33"]["mlp"]
            lines.append(
                f"| `{split}` | {split_summary['trial_count']} | "
                f"{h33['pb_rmse_m']:.6f} / {h33['vb_rmse_mps']:.6f} | "
                f"{'PASS' if split_summary['acceptance']['passed'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "### 分轨迹的 0.5 s rollout",
            "",
            "| 轨迹 | 名义位置 RMSE | MLP位置 RMSE | 位置MSE改善 | 名义速度 RMSE | MLP速度 RMSE | 速度MSE改善 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, group_values in sorted(summary["rollout"]["by_group"].items()):
        value = group_values["h30"]
        lines.append(
            f"| `{group}` | {value['nominal']['pb_rmse_m']:.6f} | "
            f"{value['mlp']['pb_rmse_m']:.6f} | "
            f"{100.0 * value['improvement']['pb_mse_fraction']:.1f}% | "
            f"{value['nominal']['vb_rmse_mps']:.6f} | "
            f"{value['mlp']['vb_rmse_mps']:.6f} | "
            f"{100.0 * value['improvement']['vb_mse_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "### 分速度状态的逐点残差",
            "",
            "| 速度状态 | 样本数 | 名义RMSE (m/s²) | MLP RMSE (m/s²) | MLP Bias (m/s²) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for regime, value in summary["point"]["by_speed_regime"].items():
        lines.append(
            f"| `{regime}` | {value['count']} | {_number(value['nominal']['rmse'])} | "
            f"{_number(value['mlp']['rmse'])} | {_number(value['mlp']['bias'])} |"
        )
    lines.extend(
        [
            "",
            "### 分位置区间的逐点残差",
            "",
            "| 位置区间 (m) | 样本数 | 名义RMSE (m/s²) | MLP RMSE (m/s²) | MLP R² |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for position_range, value in summary["point"]["by_position"].items():
        lines.append(
            f"| `{position_range}` | {value['count']} | {_number(value['nominal']['rmse'])} | "
            f"{_number(value['mlp']['rmse'])} | {_number(value['mlp']['r2'], 4)} |"
        )
    lines.extend(
        [
            "",
            "### 诊断与验收",
            "",
            f"- 验证样本 OOD 比例：{100.0 * diagnostics['ood_fraction']:.2f}%",
            f"- 残差输出饱和比例：{100.0 * diagnostics['residual_saturation_rate']:.3f}%",
            f"- 静止段总预测加速度中位数：{diagnostics['static_total_acc_median_mps2']:.6f} m/s²",
            f"- {'物理状态无量纲 Jacobian' if summary.get('training_variant') == 'causal_rollout' else '归一化输入梯度'}范数 P99：{diagnostics['gradient_norm_p99']:.4f}",
        ]
    )
    if summary.get("training_variant") == "causal_rollout":
        queue = diagnostics["command_queue_alignment"]
        response_gate = diagnostics["response_chain_gate"]
        lines.extend(
            [
                f"- Rollout 状态 OOD 比例：{100.0 * diagnostics['rollout_state_ood_fraction']:.2f}%",
                f"- 物理 Jacobian 范数 P50/P95/P99/max："
                f"{diagnostics['jacobian_norm_p50_mps2']:.6f} / "
                f"{diagnostics['jacobian_norm_p95_mps2']:.6f} / "
                f"{diagnostics['jacobian_norm_p99_mps2']:.6f} / "
                f"{diagnostics['jacobian_norm_max_mps2']:.6f} m/s²",
                f"- 相邻预测状态残差变化 P95/P99/max："
                f"{diagnostics['adjacent_residual_change_p95_mps2']:.6f} / "
                f"{diagnostics['adjacent_residual_change_p99_mps2']:.6f} / "
                f"{diagnostics['adjacent_residual_change_max_mps2']:.6f} m/s²",
                f"- 命令队列对齐 RMSE/max：{queue['rmse_mps']:.3e} / "
                f"{queue['max_abs_error_mps']:.3e} m/s",
                f"- 响应链系统偏差："
                f"{'检测到，禁止进入实物测试' if response_gate['systematic_bias_detected'] else '未检测到'}",
            ]
        )
        if "channel" in diagnostics:
            channel = diagnostics["channel"]
            lines.extend(
                [
                    f"- theta 总增益比例 min/P1/P99/max："
                    f"{channel['theta_ratio_min']:.4f} / {channel['theta_ratio_p01']:.4f} / "
                    f"{channel['theta_ratio_p99']:.4f} / {channel['theta_ratio_max']:.4f}",
                    f"- anti-damping P99/max：{channel['anti_damping_p99_sinv']:.4f} / "
                    f"{channel['anti_damping_max_sinv']:.4f} s⁻¹",
                ]
            )
    lines.extend(["", "| 验收项 | 结果 |", "| --- | --- |"])
    for name, passed in acceptance["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "### 结果图",
            "",
            f"![Training history]({summary.get('results_dir', 'results_causal' if summary.get('training_variant') == 'causal_rollout' else 'results')}/figures/training_history.png)",
            "",
            f"![Point residual validation]({summary.get('results_dir', 'results_causal' if summary.get('training_variant') == 'causal_rollout' else 'results')}/figures/point_residual_validation.png)",
            "",
            f"![Rollout RMSE validation]({summary.get('results_dir', 'results_causal' if summary.get('training_variant') == 'causal_rollout' else 'results')}/figures/rollout_rmse_validation.png)",
            "",
            f"![Example rollout]({summary.get('results_dir', 'results_causal' if summary.get('training_variant') == 'causal_rollout' else 'results')}/figures/example_rollout_validation.png)",
            "",
            (
                "> 注意：validation 用于 early stopping；test/stress 在 checkpoint 固定后"
                "才运行，未参与模型选择。"
                if summary.get("schema_version") == 3
                else "> 注意：同一验证集用于 early stopping 和最终离线评价，以上结果"
                "不是独立测试集上的无偏泛化估计。"
            ),
        ]
    )
    return "\n".join(lines)


def _render_loss_budget(training: dict[str, Any]) -> list[str]:
    """Render exact weighted validation-loss contributions when available."""

    metrics = training.get("best_validation_metrics", {})
    required = {
        "total_loss",
        "rollout_loss",
        "weighted_point_residual_loss",
        "weighted_jacobian_loss",
    }
    if not required.issubset(metrics):
        return []

    total = float(metrics["total_loss"])
    denominator = max(total, 1.0e-15)
    channel = float(metrics.get("weighted_theta_channel_loss", 0.0)) + float(
        metrics.get("weighted_anti_damping_loss", 0.0)
    )
    components = (
        ("Rollout", float(metrics["rollout_loss"])),
        ("逐点 residual", float(metrics["weighted_point_residual_loss"])),
        ("Jacobian", float(metrics["weighted_jacobian_loss"])),
        ("控制通道", channel),
    )
    lines = [
        "",
        "### 最佳 checkpoint 的 validation loss 构成",
        "",
        "| 加权项 | 数值 | 占总 loss |",
        "| --- | ---: | ---: |",
    ]
    for name, value in components:
        lines.append(f"| {name} | {value:.6f} | {100.0 * value / denominator:.2f}% |")
    lines.append(f"| **总计** | **{total:.6f}** | **100.00%** |")

    horizon_keys = sorted(
        (
            key
            for key in metrics
            if key.startswith("weighted_rollout_loss_h")
        ),
        key=lambda item: int(item.rsplit("h", maxsplit=1)[1]),
    )
    if horizon_keys:
        rollout_total = max(float(metrics["rollout_loss"]), 1.0e-15)
        lines.extend(
            [
                "",
                "| Rollout horizon | 加权贡献 | 占 rollout loss | 占总 loss |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for key in horizon_keys:
            step = int(key.rsplit("h", maxsplit=1)[1])
            value = float(metrics[key])
            lines.append(
                f"| h{step} | {value:.6f} | {100.0 * value / rollout_total:.2f}% | "
                f"{100.0 * value / denominator:.2f}% |"
            )
    return lines


def _render_screening(screening: dict[str, Any]) -> list[str]:
    if screening.get("schema_version") == 2:
        lines = [
            "",
            "### 控制通道约束筛选",
            "",
            f"- 短预算候选：`{screening['screening_selected_candidate'] or '无'}`",
            f"- 三 seed 稳定性：**{'PASS' if screening['seed_stability_passed'] else 'FAIL'}**",
            f"- test/stress：**{'PASS' if screening['test_and_stress_passed'] else 'FAIL/未运行'}**",
            f"- 最终可导出候选：`{screening['selected_candidate'] or '无'}`",
            "",
            "| 候选 | λθ | 残差 RMSE | h33 pb/vb RMSE | ρ P1/P99 | gv P99/max | 硬过滤 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for value in screening["initial_candidates"]:
            lines.append(
                f"| `{value['candidate']}` | {value['theta_channel_weight']:.1f} | "
                f"{value['residual_rmse_mps2']:.6f} | "
                f"{value['h33_causal_pb_rmse_m']:.6f} / "
                f"{value['h33_causal_vb_rmse_mps']:.6f} | "
                f"{value['theta_ratio_p01']:.4f} / {value['theta_ratio_p99']:.4f} | "
                f"{value['anti_damping_p99_sinv']:.4f} / "
                f"{value['anti_damping_max_sinv']:.4f} | "
                f"{'PASS' if value['hard_filters_passed'] else 'FAIL'} |"
            )
        return lines
    policy = screening.get("selection_policy", {})
    repeat_seeds = policy.get("repeat_seeds", [7, 17, 27])
    repeat_label = "单 seed 正式训练硬过滤" if len(repeat_seeds) == 1 else "三 seed 稳定性"
    final_label = (
        "最终选择候选"
        if policy.get("allow_ineligible_fallback", False)
        else "最终可导出候选"
    )
    lines = [
        "",
        "### Jacobian 受控筛选",
        "",
        f"- 短预算候选：`{screening['screening_selected_candidate'] or '无'}`",
        f"- {repeat_label}：**{'PASS' if screening['seed_stability_passed'] else 'FAIL'}**",
        f"- {final_label}：`{screening['selected_candidate'] or '无'}`",
        "",
        "| 初筛候选 | 残差 RMSE | h33 pb/vb RMSE | Jacobian P99/max | 硬过滤 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    if policy.get("allow_ineligible_fallback", False):
        lines[5:5] = [
            f"- 强制择优：**{'YES' if screening.get('forced_selection') else 'NO'}**",
            f"- test/stress：**{'PASS' if screening.get('test_and_stress_passed') else 'FAIL'}**",
        ]
    for value in screening["initial_candidates"]:
        lines.append(
            f"| `{value['candidate']}` | {value['residual_rmse_mps2']:.6f} | "
            f"{value['h33_causal_pb_rmse_m']:.6f} / "
            f"{value['h33_causal_vb_rmse_mps']:.6f} | "
            f"{value['jacobian_p99_mps2']:.6f} / "
            f"{value['jacobian_max_mps2']:.6f} | "
            f"{'PASS' if value['hard_filters_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "| 正式复验 seed | 残差 RMSE | h33 pb/vb RMSE | Jacobian P99/max | 硬过滤 |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for value in screening["seed_repetitions"]:
        lines.append(
            f"| {value['seed']} | {value['residual_rmse_mps2']:.6f} | "
            f"{value['h33_causal_pb_rmse_m']:.6f} / "
            f"{value['h33_causal_vb_rmse_mps']:.6f} | "
            f"{value['jacobian_p99_mps2']:.6f} / "
            f"{value['jacobian_max_mps2']:.6f} | "
            f"{'PASS' if value['hard_filters_passed'] else 'FAIL'} |"
        )
    return lines


def _point_row(name: str, values: dict[str, float]) -> str:
    return (
        f"| {name} | {values['rmse']:.6f} | {values['mae']:.6f} | "
        f"{values['bias']:.6f} | {values['r2']:.4f} |"
    )


def _number(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _benchmark_text(benchmark: dict[str, Any]) -> str:
    cpu = benchmark.get("cpu_median_step_s")
    cuda = benchmark.get("cuda_median_step_s")
    if cpu is None or cuda is None:
        return str(benchmark.get("reason") or "未执行")
    gpu_name = benchmark.get("gpu_name") or "CUDA"
    return (
        f"CPU {1000.0 * cpu:.2f} ms/step，{gpu_name} "
        f"{1000.0 * cuda:.2f} ms/step；{benchmark.get('reason')}"
    )


def _wandb_link(wandb: dict[str, Any]) -> str:
    url = wandb.get("run_url")
    return f"[{url}]({url})" if url else "N/A（本地 offline run）"
