"""Generate the measured result section of the RBF design report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import Config
from .training import resolve_run_dir


BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


def _gate_table(acceptance: Mapping[str, Any]) -> list[str]:
    lines = ["| Gate | Result |", "| --- | --- |"]
    for name, passed in acceptance["gates"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    return lines


def _results(summary: Mapping[str, Any], training: Mapping[str, Any]) -> str:
    lines = [
        "## 实际训练与评估结果",
        "",
        f"- Run：`{training['run_name']}`",
        f"- seed：`{training['seed']}`；训练设备：`{training['training_device']}`",
        f"- 有效 cap：`±{training['effective_cap_mps2']:.3f} m/s²`",
        f"- Stage 1：epoch {training['stage1']['best_epoch']}，耗时 {training['stage1']['training_time_s']:.1f} s，selection score {training['stage1']['selection_score']:.6f}",
        f"- Stage 2：epoch {training['stage2']['best_epoch']}，耗时 {training['stage2']['training_time_s']:.1f} s，selection score {training['stage2']['selection_score']:.6f}",
        f"- 最终选择：**{training['selected_stage']}**",
        f"- Residual integration eligibility：**{'PASS' if summary['eligible_for_residual_integration'] else 'FAIL'}**",
        f"- Real-world deployment eligibility：**{'PASS' if summary['eligible_for_real_world_deployment'] else 'FAIL'}**",
        "",
        "### 各 split 总体指标",
        "",
        "| Split | Point nominal/RBF RMSE | Integral h12 nominal/RBF RMSE | h33 pb nominal/RBF | h33 vb nominal/RBF | Acceptance |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for split, values in summary["splits"].items():
        point = values["point"]["overall"]
        integral = values["integral"]["overall"]["h12"]
        rollout = values["rollout"]["overall"]["h33"]
        lines.append(
            f"| `{split}` | {point['nominal']['rmse']:.6f} / {point['rbf']['rmse']:.6f} | "
            f"{integral['nominal_rmse_mps2']:.6f} / {integral['rbf_rmse_mps2']:.6f} | "
            f"{rollout['nominal']['pb_rmse_m']:.6f} / {rollout['rbf']['pb_rmse_m']:.6f} | "
            f"{rollout['nominal']['vb_rmse_mps']:.6f} / {rollout['rbf']['vb_rmse_mps']:.6f} | "
            f"{'PASS' if values['acceptance']['passed'] else 'FAIL'} |"
        )
    validation = summary["splits"]["validation"]
    point = validation["point"]["overall"]
    derivatives = validation["derivatives"]
    diagnostic = validation["point"]["diagnostics"]
    shadow = diagnostic["shadow_full_state_ood"]
    queue = validation["command_queue_alignment"]
    response = validation["response_chain_gate"]
    lines.extend(
        [
            "",
            "### Validation 逐点与旧模型对照",
            "",
            "| 模型 | RMSE | MAE | Bias | R² |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Nominal | {point['nominal']['rmse']:.6f} | {point['nominal']['mae']:.6f} | {point['nominal']['bias']:.6f} | {point['nominal']['r2']:.4f} |",
            f"| RBF-PV | {point['rbf']['rmse']:.6f} | {point['rbf']['mae']:.6f} | {point['rbf']['bias']:.6f} | {point['rbf']['r2']:.4f} |",
        ]
    )
    if point.get("baseline_mlp") is not None:
        baseline = point["baseline_mlp"]
        lines.append(
            f"| CAP0P25 MLP | {baseline['rmse']:.6f} | {baseline['mae']:.6f} | {baseline['bias']:.6f} | {baseline['r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Validation 导数、OOD 与静态诊断",
            "",
            f"- `rp` abs P99/max：{derivatives['rp_abs_p99_sinv2']:.6f} / {derivatives['rp_abs_max_sinv2']:.6f} s⁻²",
            f"- `rv` min/P1/P50/P99/max：{derivatives['rv_min_sinv']:.6f} / {derivatives['rv_p01_sinv']:.6f} / {derivatives['rv_p50_sinv']:.6f} / {derivatives['rv_p99_sinv']:.6f} / {derivatives['rv_max_sinv']:.6f} s⁻¹",
            f"- exogenous residual gradient max：theta={derivatives['direct_theta_residual_gradient_max']:.3e}，omega={derivatives['direct_omega_residual_gradient_max']:.3e}，a_nom={derivatives['direct_anom_residual_gradient_max']:.3e}",
            f"- residual saturation rate：{100.0 * diagnostic['saturation_rate']:.3f}%",
            f"- OOD fraction/mean weight/min weight：{100.0 * diagnostic['ood_fraction']:.2f}% / {diagnostic['ood_weight_mean']:.6f} / {diagnostic['ood_weight_min']:.6f}",
            f"- shadow full-state OOD fraction/P99 distance：{100.0 * shadow['fraction_outside']:.2f}% / {shadow['distance_p99']:.6f}（仅诊断，不进入动力学）",
            f"- command queue alignment RMSE/max：{queue['rmse_mps']:.3e} / {queue['max_abs_error_mps']:.3e} m/s；{'PASS' if queue['passed'] else 'FAIL'}",
            f"- response-chain theta/vrz bias：{response['theta_bias_rad']:.6f} rad / {response['vrz_bias_mps']:.6f} m/s；{'PASS' if response['passed'] else 'FAIL'}",
            f"- static total acceleration median：{diagnostic['static_total_acc_median_mps2']:.6f} m/s²",
            "",
            "### 结果解释",
            "",
            "- 二维结构达成了首要安全目标：残差对 theta/omega/a_nom 的直接导数严格为零，控制通道不再依赖 soft loss 维持。",
            f"- Validation h33 的位置/速度 MSE 改善为 {100.0 * validation['rollout']['overall']['h33']['improvement']['pb_mse_fraction']:.2f}% / {100.0 * validation['rollout']['overall']['h33']['improvement']['vb_mse_fraction']:.2f}%，属于保守但有限的预测收益。",
            f"- Validation h3/h6/h12 积分改善分别为 {100.0 * validation['integral']['overall']['h3']['improvement_fraction']:.2f}% / {100.0 * validation['integral']['overall']['h6']['improvement_fraction']:.2f}% / {100.0 * validation['integral']['overall']['h12']['improvement_fraction']:.2f}%；短 horizon 未达到5% gate。",
            "- 当前未通过不代表重新出现旧模型式控制通道破坏；主要剩余问题是残差可辨识收益偏小、静态偏差和 outer-grid 速度梯度轻微越界。",
            "",
            "### Validation hard gates",
            "",
            *_gate_table(validation["acceptance"]),
            "",
            "### 部署阻塞项",
            "",
            *[f"- {value}" for value in summary["deployment_blockers"]],
            "",
            "### 图表",
            "",
            "![Training history](results_rbf_pv/figures/training_history.png)",
            "",
            "![Point validation](results_rbf_pv/figures/point_validation.png)",
            "",
            "![Integral validation](results_rbf_pv/figures/integral_validation.png)",
            "",
            "![Validation rollout](results_rbf_pv/figures/rollout_validation.png)",
            "",
            "![Derivative summary](results_rbf_pv/figures/derivative_summary.png)",
            "",
            "![Control sensitivity](results_rbf_pv/figures/control_sensitivity.png)",
            "",
            "完整的 session、group、position-velocity cell 和 a_nom regime 分层指标保存在 `results_rbf_pv/summary.json`。",
            "",
            "### 完整性",
            "",
            f"- dataset hash：`{summary['dataset_hash']}`",
            f"- config hash：`{summary['config_hash']}`",
            f"- checkpoint hash：`{training['checkpoint_sha256']}`",
            f"- portable model hash：`{summary['portable_model_sha256']}`",
        ]
    )
    return "\n".join(lines)


def generate_report(config: Config, run_dir: str | Path | None = None) -> Path:
    summary_path = config.paths.results_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run evaluate before report.")
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    resolved = resolve_run_dir(config, run_dir)
    with (resolved / "training_summary.json").open("r", encoding="utf-8") as stream:
        training = json.load(stream)
    text = config.paths.report_path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise ValueError("REPORT_RBF_PV.md is missing generated-result markers.")
    prefix, remainder = text.split(BEGIN, 1)
    _, suffix = remainder.split(END, 1)
    config.paths.report_path.write_text(
        f"{prefix}{BEGIN}\n{_results(summary, training)}\n{END}{suffix}",
        encoding="utf-8",
    )
    return config.paths.report_path
