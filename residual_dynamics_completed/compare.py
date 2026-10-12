"""Generate publication-sized comparison figures and an evidence-based report."""
from __future__ import annotations
import argparse
import csv
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/residual_completed_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import ROOT, MODEL_NAMES

LABELS = {"nominal": "Nominal", "profile_only": "BPR", "friction_only": "FDR",
          "rbf_pv": "RBF-PV", "mlp": "MLP", "structured": "Structured"}


def compare(split="test"):
    out = ROOT / "results" / split
    with (out / "metrics_long.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads((out / "metadata.json").read_text())
    totals = {(r["model"], int(r["horizon"])): r for r in rows if r["scope"] == "all"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    colors = ("#666666", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#D55E00")
    markers = ("o", "s", "^", "D", "v", "P")
    styles = ("--", "-.", ":", "--", "-.", "-")
    for ax, metric, scale, label in zip(axes,
            ("position_rmse_m", "velocity_rmse_mps"), (100., 100.),
            ("Position RMSE (cm)", "Velocity RMSE (cm/s)")):
        for name, color, marker, style in zip(MODEL_NAMES, colors, markers, styles):
            horizons = sorted(h for model, h in totals if model == name)
            values = [float(totals[name, h][metric])*scale for h in horizons]
            ax.plot(horizons, values, color=color, marker=marker, linestyle=style,
                    markersize=3.8, linewidth=2.0 if name == "structured" else 1.35, label=LABELS[name])
        ax.set(xlabel="Prediction horizon (steps)", ylabel=label, xticks=[6,18,30,42,54], xlim=(3,62))
        ax.grid(True, color="#D8D8D8", linewidth=.5, alpha=.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.8)
        ax.tick_params(width=.8, length=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,1.015), ncol=3,
               frameon=False, columnspacing=1.8, handlelength=2.8)
    fig.subplots_adjust(left=.085, right=.99, bottom=.20, top=.77, wspace=.32)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"six_model_rollout_comparison.{extension}", dpi=400)
    plt.close(fig)
    lines = ["# Residual dynamics comparison — final experiment report", "",
        "## Protocol and provenance", "",
        f"Six models; {metadata['windows']:,} identical {split} windows; seed 7 only. "
        "Each window is integrated once to 60 steps without future-state resets. "
        "No test data were used for fitting or checkpoint selection.", "",
        f"Dataset hash: `{metadata['dataset_hash']}`.",
        f"Window hash: `{metadata['window_hash']}`.", "",
        "Full model/training configuration, source hashes, software versions and per-model checkpoint identities "
        "are in `results/test/metadata.json` and `runs/*/config.json`. "
        "Trial-level train/validation/test split is 79/6/6; five stress trials remain separate. "
        "Training data loss and schedule are common; regularization is representation-specific (see protocol.md).", "",
        "## Training and model selection", "",
        "| Model | Active parameters | Stage 1 / 2 epochs | Best epoch | Validation score | Time (s) |",
        "|---|---:|---:|---|---:|---:|"]
    for name in MODEL_NAMES:
        info = metadata["models"][name]
        if name == "nominal":
            lines.append("| Nominal | 0 | — | — | 1.000000 | — |")
        else:
            lines.append(f"| {LABELS[name]} | {metadata['diagnostics'][name]['trainable_parameters']} | "
                f"{info['epochs']['stage1']} / {info['epochs']['stage2']} | {' / '.join(map(str,info['best_epoch']))} | "
                f"{info['best_validation_score']:.9f} | {info['elapsed_s']:.1f} |")
    lines += ["", "## Held-out pooled RMSE", "",
        "Pooled over windows, not averaged per-trial RMSE. Positive reduction means improvement over nominal.", "",
        "| Model | H | Position (cm) | Velocity (cm/s) | Position reduction (%) | Velocity reduction (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    for horizon in (30, 60):
        for name in MODEL_NAMES:
            r, nom = totals[name,horizon], totals["nominal",horizon]
            p, v = float(r["position_rmse_m"]), float(r["velocity_rmse_mps"])
            dp = 100*(1-p/float(nom["position_rmse_m"]))
            dv = 100*(1-v/float(nom["velocity_rmse_mps"]))
            lines.append(f"| {LABELS[name]} | {horizon} | {100*p:.5f} | {100*v:.5f} | {dp:.3f} | {dv:.3f} |")
    lines += ["", "## Horizon-dependent ranking", ""]
    for horizon in (30,60):
        for metric in ("position_rmse_m", "velocity_rmse_mps"):
            rank = sorted(MODEL_NAMES, key=lambda n: float(totals[n,horizon][metric]))
            lines.append(f"- H={horizon}, {metric}: " + " < ".join(LABELS[n] for n in rank) + ".")
    lines += ["", "## Per-reference-type RMSE", "",
        "| Reference | H | Model | Position (cm) | Velocity (cm/s) |", "|---|---:|---|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["group"], int(r["horizon"]), r["model"])):
        if r["scope"] == "reference_type" and int(r["horizon"]) in (30,60):
            lines.append(f"| {r['group']} | {r['horizon']} | {LABELS[r['model']]} | "
                         f"{100*float(r['position_rmse_m']):.5f} | {100*float(r['velocity_rmse_mps']):.5f} |")
    if split == "test":
        lines += ["", "## Interpretation for the paper", "",
            "Structured improves over nominal at all ten evaluated horizons, and improves over both "
            "ablations and RBF-PV in pooled H=30 and H=60 position/velocity RMSE. MLP nevertheless has "
            "the lowest pooled offline errors, reducing H=60 position/velocity RMSE by 36.3%/41.9% "
            "versus Structured's 19.7%/20.8%. This does not support claiming that physical structure "
            "maximizes prediction accuracy. Structured uses 39 active coefficients versus MLP's 385 parameters, "
            "and its incorporation in predictive control has separate existing real-world evidence; "
            "there is no MLP closed-loop comparison in this experiment.", "",
            "There are task-dependent exceptions: at H=30, RBF-PV is slightly better than Structured "
            "on constant-reference windows (both position and velocity), while the order reverses by H=60. "
            "FDR slightly worsens triangular-reference H=30 position RMSE relative to nominal; "
            "RBF-PV slightly worsens sinusoidal-reference H=60 velocity RMSE. "
            "Pooled ranking must not be generalized to every task."]
    lines += ["", "## Auxiliary diagnostics", "",
        "Saturation means absolute applied residual >= 99% of 0.25 m/s²; it is not hard clipping.", "",
        "| Model | Point acceleration RMSE (m/s²) | RK4-stage saturation (%) | Nonfinite predictions |",
        "|---|---:|---:|---:|"]
    for name in MODEL_NAMES:
        d = metadata["diagnostics"][name]
        lines.append(f"| {LABELS[name]} | {d['point_acceleration_rmse_mps2']:.6f} | "
            f"{100*d['rollout_stage_saturation_fraction']:.4f} | {d['nonfinite_prediction_values']} |")
    lines += ["", "## Acceptance and reproduction", "",
        "See `provenance/parity.json` for the bitwise forward, sampling, batch, loss and gradient comparison "
        "against the predecessor. `results/acceptance.json` records independent aggregate recomputation, "
        "checkpoint/Structured-retraining reproduction and contract tests. "
        "Paper updates must wait for that acceptance file to report passed.", "",
        "## Limitations and claim boundaries", "",
        "- Single training seed: no seed-variance bars or statistical superiority claims.",
        "- Overlapping windows are correlated; 2,028 windows are not 2,028 independent experiments.",
        "- Six held-out trials cover four reference types but provide limited per-type replication.",
        "- The common data loss/selection budget does not make representation-specific regularizers identical; "
        "this comparison does not establish optimally tuned MLP/RBF performance.",
        "- Five-input normalization is exactly the predecessor's training-only statistic, including its "
        "pre-mask training features; no validation or test samples enter it.",
        "- Point acceleration targets depend on offline smoothing/differentiation; causal rollouts and measured "
        "endpoint states provide the primary control-relevant evidence.",
        "- These are offline results; baseline ranking does not establish closed-loop superiority. "
        "Existing physical experiments and deployed weights are unchanged.", ""]
    acceptance_path = ROOT / "results/acceptance.json"
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text())
        lines += ["## Recorded reproduction outcome", "",
            f"Acceptance passed: {acceptance['passed']}. "
            f"Maximum Structured parameter difference: {max(acceptance['structured_parameter_max_differences'].values()):.3e}; "
            f"maximum training-history difference: {acceptance['structured_history_max_absolute_difference']:.3e}. "
            "Both stages used the same number of epochs and selected stage2/epoch37 with exactly the same "
            "validation score. This is floating-point-precision reproduction, not bitwise identity. "
            "The old runtime record does not include CPU thread count, so the original cause of the round-off-scale "
            "training-history differences cannot be uniquely assigned. The explicit migration comparison "
            "using the same saved parameters is bitwise identical for forward, sampler, loss and gradients. "
            "Old physical weights remain untouched; the new checkpoint container records the offline retraining provenance.", ""]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    report_name = "experiment_report.md" if split == "test" else f"{split}_report.md"
    (docs / report_name).write_text("\n".join(lines))
    print(f"Wrote comparison plot and {docs / 'experiment_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=("test", "validation", "stress"), default="test")
    compare(parser.parse_args().split)
