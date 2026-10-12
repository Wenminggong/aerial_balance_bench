"""Validation, retrospective test, stress analysis, and v1/v1.5 comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aerial_balance_residual_v1_5_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import AppConfig
from .data import TrialData, load_manifest, load_trials
from .dynamics import rollout_ball_dynamics
from .model import ModelOutput, SmoothDeadzoneResidualModel, load_checkpoint
from .training import WindowStore, resolve_run_dir


MODEL_NAMES = ("nominal", "v1", "dynamic_mlp", "v1_5")
EXPECTED_V1_WEIGHTS_SHA256 = "be25c1652ce47feeffe55aecf39ccc337dd33cc765f22832ae6c9d99265e9e0e"


class V1Adapter(nn.Module):
    """Read the fixed v1 portable arrays without importing v1 Python code."""
    def __init__(self, path: Path) -> None:
        super().__init__()
        with np.load(path, allow_pickle=False) as archive:
            self.register_buffer("mean", torch.from_numpy(archive["input_mean"].copy()))
            self.register_buffer("std", torch.from_numpy(archive["input_std"].copy()))
            self.limit = float(archive["output_limit_mps2"])
            dims = [5, archive["layer_0_weight"].shape[0], archive["layer_1_weight"].shape[0], 1]
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1], dtype=torch.float64) for i in range(3)])
            with torch.no_grad():
                for i, layer in enumerate(self.layers):
                    layer.weight.copy_(torch.from_numpy(archive[f"layer_{i}_weight"].copy()))
                    layer.bias.copy_(torch.from_numpy(archive[f"layer_{i}_bias"].copy()))

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        value = (features - self.mean) / self.std
        value = torch.tanh(self.layers[0](value)); value = torch.tanh(self.layers[1](value))
        return self.limit * torch.tanh(self.layers[2](value).squeeze(-1))

    def forward(self, features: torch.Tensor) -> ModelOutput:
        residual = self.residual(features); drive = features[..., 4] + residual
        ones, zeros = torch.ones_like(drive), torch.zeros_like(drive)
        return ModelOutput(residual, drive, ones, drive, zeros, drive, residual)


def evaluate_model(config: AppConfig, run_dir: str | Path | None = None) -> dict[str, Any]:
    resolved, model, checkpoint, v1, hashes = _load_models(config, run_dir)
    manifest = load_manifest(config, "fit")
    if checkpoint["fit_dataset_hash"] != manifest["dataset_hash"]:
        raise RuntimeError("Checkpoint and fit manifest hash mismatch")
    summary, arrays = _evaluate_scope(model, v1, load_trials(config, "validation"), manifest, config)
    summary.update(scope="validation", split="validation", model_hashes=hashes)
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.paths.results_dir / "validation_summary.json", summary)
    np.savez_compressed(resolved / "validation_predictions_v1_5.npz", **arrays)
    _plot_scope(summary, config.paths.results_dir / "figures", "validation")
    return summary


def test_model(config: AppConfig, run_dir: str | Path | None = None) -> dict[str, Any]:
    resolved, model, checkpoint, v1, hashes = _load_models(config, run_dir)
    fit_manifest, test_manifest = load_manifest(config, "fit"), load_manifest(config, "test")
    if checkpoint["fit_dataset_hash"] != fit_manifest["dataset_hash"]:
        raise RuntimeError("Checkpoint and fit manifest hash mismatch")
    validation_path = config.paths.results_dir / "validation_summary.json"
    if not validation_path.is_file():
        raise FileNotFoundError("Run evaluate before test")
    validation = _read_json(validation_path)
    test, test_arrays = _evaluate_scope(model, v1, load_trials(config, "test"), fit_manifest, config)
    stress, stress_arrays = _evaluate_scope(model, v1, load_trials(config, "stress"), fit_manifest, config)
    test.update(scope="retrospective_test", split="test", model_hashes=hashes,
                test_dataset_hash=test_manifest["dataset_hash"])
    stress.update(scope="stress", split="stress", model_hashes=hashes,
                  test_dataset_hash=test_manifest["dataset_hash"])
    training = _read_json(resolved / "training_summary.json")
    acceptance = _acceptance(validation, test, model, config)
    final = {
        "schema_version": 1, "model_version": "1.5.0", "run_name": training["run_name"],
        "config_hash": config.config_hash, "fit_dataset_hash": fit_manifest["dataset_hash"],
        "test_dataset_hash": test_manifest["dataset_hash"], "seed": config.training.seed,
        "training": training, "validation": validation, "test": test, "stress": stress,
        "acceptance": acceptance,
        "test_interpretation": "retrospective external test; 20260817 informed model design",
    }
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.paths.results_dir / "test_summary.json", test)
    _write_json(config.paths.results_dir / "summary.json", final)
    _write_metrics_csv(config.paths.results_dir / "metrics.csv", final)
    np.savez_compressed(resolved / "test_predictions_v1_5.npz", **test_arrays)
    np.savez_compressed(resolved / "stress_predictions_v1_5.npz", **stress_arrays)
    _plot_scope(test, config.paths.results_dir / "figures", "test")
    _plot_training(resolved, config.paths.results_dir / "figures")
    return final


def _load_models(config, run_dir):
    resolved = resolve_run_dir(config, run_dir)
    model, checkpoint = load_checkpoint(resolved / "best_composite_checkpoint.pt", "cpu")
    model.eval()
    if not config.paths.v1_weights_path.is_file():
        raise FileNotFoundError(f"Fixed v1 weights missing: {config.paths.v1_weights_path}")
    v1_hash = _sha256(config.paths.v1_weights_path)
    if v1_hash != EXPECTED_V1_WEIGHTS_SHA256:
        raise RuntimeError(
            f"Fixed v1 weights hash mismatch: expected {EXPECTED_V1_WEIGHTS_SHA256}, got {v1_hash}"
        )
    v1 = V1Adapter(config.paths.v1_weights_path).eval()
    return resolved, model, checkpoint, v1, {
        "v1_weights_sha256": v1_hash,
        "v1_5_checkpoint_sha256": _sha256(resolved / "best_composite_checkpoint.pt"),
    }


def _evaluate_scope(model: SmoothDeadzoneResidualModel, v1: V1Adapter,
                    trials: Sequence[TrialData], fit_manifest: dict[str, Any],
                    config: AppConfig) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    features = np.concatenate([t.arrays["features"] for t in trials])
    target_residual = np.concatenate([t.arrays["residual_target"] for t in trials])
    measured_ab = np.concatenate([t.arrays["ab"] for t in trials])
    valid = np.concatenate([t.arrays["valid_mask"] for t in trials])
    dynamic = np.concatenate([t.arrays["dynamic_mask"] for t in trials])
    static = np.concatenate([t.arrays["static_mask"] for t in trials])
    transition = np.concatenate([t.arrays["transition_mask"] for t in trials])
    groups = np.concatenate([np.full(t.size, t.group, dtype="U32") for t in trials])
    trial_ids = np.concatenate([np.full(t.size, t.trial_id, dtype="U96") for t in trials])
    breakaway = np.concatenate([_breakaway_mask(t, config) for t in trials])
    tensor = torch.from_numpy(features).to(torch.float64)
    with torch.no_grad():
        output = model(tensor); v1_output = v1(tensor)
    predictions = {
        "nominal": features[:, 4], "v1": v1_output.total_acceleration.numpy(),
        "dynamic_mlp": output.drive_acceleration.numpy(), "v1_5": output.total_acceleration.numpy(),
    }
    residual_predictions = {name: pred - features[:, 4] for name, pred in predictions.items()}
    point = {
        "dynamic": {name: _regression(pred[dynamic], target_residual[dynamic])
                    for name, pred in residual_predictions.items()},
        "total_valid": {name: _regression(pred[valid], measured_ab[valid])
                        for name, pred in predictions.items()},
        "transition": {name: _regression(pred[transition], measured_ab[transition])
                       for name, pred in predictions.items()},
        "breakaway": {name: _regression(pred[breakaway], measured_ab[breakaway])
                      for name, pred in predictions.items()},
        "by_group": {}, "by_trial": {}, "by_position_bin": {}, "by_speed_mode": {},
    }
    for group in sorted(set(groups.tolist())):
        mask = valid & (groups == group)
        point["by_group"][group] = {name: _regression(pred[mask], measured_ab[mask])
                                    for name,pred in predictions.items()}
    for trial in trials:
        mask = valid & (trial_ids == trial.trial_id)
        dyn = dynamic & (trial_ids == trial.trial_id)
        point["by_trial"][trial.trial_id] = {
            "group": trial.group, "reference_type": trial.reference_type,
            "total_acceleration": {name: _regression(pred[mask], measured_ab[mask])
                                   for name, pred in predictions.items()},
            "dynamic_residual": {name: _regression(pred[dyn], target_residual[dyn])
                                 for name, pred in residual_predictions.items()},
        }
    position_edges = (-np.inf, 0.25, 0.35, 0.45, np.inf)
    position_names = ("pb_lt_0p25", "pb_0p25_0p35", "pb_0p35_0p45", "pb_ge_0p45")
    for name, low, high in zip(position_names, position_edges[:-1], position_edges[1:]):
        mask = valid & (features[:, 0] >= low) & (features[:, 0] < high)
        point["by_position_bin"][name] = {
            "range_m": [None if not np.isfinite(low) else low,
                        None if not np.isfinite(high) else high],
            "models": {model_name: _regression(pred[mask], measured_ab[mask])
                       for model_name, pred in predictions.items()},
        }
    speed_masks = {
        "static": static, "transition": transition,
        "dynamic_negative": dynamic & (features[:, 1] < 0),
        "dynamic_positive": dynamic & (features[:, 1] > 0),
    }
    for speed_name, mask in speed_masks.items():
        point["by_speed_mode"][speed_name] = {
            model_name: _regression(pred[mask], measured_ab[mask])
            for model_name, pred in predictions.items()
        }
    static_abs = {name: np.abs(pred[static]) for name,pred in predictions.items()}
    static_diag = {name: {
        "count": int(static.sum()), "median_abs_mps2": _quantile(value, 0.5),
        "p95_abs_mps2": _quantile(value, 0.95),
        "false_motion_rate": float(np.mean(value > 0.02)) if value.size else None,
    } for name,value in static_abs.items()}
    train_min, train_max = np.asarray(fit_manifest["normalization"]["min"]), np.asarray(fit_manifest["normalization"]["max"])
    ood = np.any((features < train_min) | (features > train_max), axis=1)
    normalized = ((tensor - model.mlp.input_mean) / model.mlp.input_std).detach().requires_grad_(True)
    normalized_output = model.mlp.normalized_residual(normalized)
    gradient = torch.autograd.grad(normalized_output.sum(), normalized)[0]
    gradient_norm = torch.linalg.vector_norm(gradient, dim=-1).detach().numpy()
    diagnostics = {
        "sample_count": int(len(features)), "valid_count": int(valid.sum()),
        "dynamic_count": int(dynamic.sum()), "static_count": int(static.sum()),
        "transition_count": int(transition.sum()), "breakaway_count": int(breakaway.sum()),
        "ood_fraction": float(ood[valid].mean()),
        "mlp_saturation_rate": float(np.mean(np.abs(output.mlp_residual.numpy()[valid]) >= 0.95 * model.mlp.output_limit_mps2)),
        "deadzone_threshold_mps2": float(model.deadzone_threshold.detach()),
        "deadzone_suppression_mean_mps2": float(output.deadzone_suppression.numpy()[valid].mean()),
        "deadzone_suppression_active_fraction": float(np.mean(
            np.abs(output.deadzone_suppression.numpy()[valid]) > 1e-4)),
        "deadzone_relative_suppression_mean": float(np.mean(np.clip(
            np.abs(output.deadzone_suppression.numpy()[valid])
            / np.maximum(np.abs(output.drive_acceleration.numpy()[valid]), 1e-9), 0.0, 1.0))),
        "gradient_norm_p99": float(np.quantile(gradient_norm[valid], 0.99)),
        "outputs_finite": bool(all(np.isfinite(value[valid]).all() for value in predictions.values())),
        "gradients_finite": bool(np.isfinite(gradient_norm[valid]).all()),
    }
    rollout = _rollout_metrics(model, v1, trials, config)
    summary = {"trial_count": len(trials), "groups": sorted(set(groups.tolist())),
               "point": point, "rollout": rollout, "static": static_diag,
               "diagnostics": diagnostics}
    arrays = {
        "pb": features[:,0], "vb":features[:,1], "theta":features[:,2], "omega":features[:,3],
        "a_nom":features[:,4], "measured_ab":measured_ab, "target_residual":target_residual,
        "valid_mask":valid, "dynamic_mask":dynamic, "static_mask":static,
        "transition_mask":transition, "breakaway_mask":breakaway,
        "groups":groups, "trial_ids":trial_ids,
        **{f"predicted_ab_{name}":value for name,value in predictions.items()},
    }
    return summary, arrays


def _rollout_metrics(model, v1, trials, config):
    horizon=max(config.data.evaluation_steps); store=WindowStore(trials,horizon,False)
    refs=store.references[::config.data.validation_stride]
    grouped: dict[str,list[Any]]={}; by_trial: dict[int,list[Any]]={}
    for ref in refs:
        grouped.setdefault(trials[ref.trial_index].group,[]).append(ref)
        by_trial.setdefault(ref.trial_index, []).append(ref)
    result={"overall":_empty_acc(config),"by_group":{},"by_trial":{}}
    for group,group_refs in sorted(grouped.items()):
        acc=_empty_acc(config)
        for offset in range(0,len(group_refs),config.training.batch_size):
            batch=store.make_batch(group_refs[offset:offset+config.training.batch_size],torch.device("cpu"))
            with torch.no_grad():
                values={
                    "nominal":rollout_ball_dynamics(None,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants),
                    "v1":rollout_ball_dynamics(v1,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants),
                    "dynamic_mlp":rollout_ball_dynamics(model,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants,"dynamic"),
                    "v1_5":rollout_ball_dynamics(model,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants,"composite"),
                }
            _update_acc(acc,batch,values,config); _update_acc(result["overall"],batch,values,config)
        result["by_group"][group]=_finalize_acc(acc,config)
    for trial_index, trial_refs in sorted(by_trial.items()):
        acc = _empty_acc(config)
        for offset in range(0, len(trial_refs), config.training.batch_size):
            part = trial_refs[offset:offset + config.training.batch_size]
            batch = store.make_batch(part, torch.device("cpu"))
            with torch.no_grad():
                values = {
                    "nominal": rollout_ball_dynamics(None,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants),
                    "v1": rollout_ball_dynamics(v1,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants),
                    "dynamic_mlp": rollout_ball_dynamics(model,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants,"dynamic"),
                    "v1_5": rollout_ball_dynamics(model,batch.pb[:,0],batch.vb[:,0],batch.theta,batch.omega,batch.dt,batch.constants,"composite"),
                }
            _update_acc(acc, batch, values, config)
        trial = trials[trial_index]
        result["by_trial"][trial.trial_id] = {
            "group": trial.group, "reference_type": trial.reference_type,
            "metrics": _finalize_acc(acc, config),
        }
    result["overall"]=_finalize_acc(result["overall"],config)
    return result


def _empty_acc(config):
    return {step:{"count":0,"dt":0.0,**{f"{name}_{state}":0.0 for name in MODEL_NAMES for state in ("p","v")}}
            for step in config.data.evaluation_steps}


def _update_acc(acc,batch,values,config):
    for step in config.data.evaluation_steps:
        row=acc[step]; row["count"]+=batch.pb.shape[0]; row["dt"]+=float(batch.dt[:,:step].sum())
        for name,(pb,vb) in values.items():
            row[f"{name}_p"]+=float((pb[:,step]-batch.pb[:,step]).square().sum())
            row[f"{name}_v"]+=float((vb[:,step]-batch.vb[:,step]).square().sum())


def _finalize_acc(acc,config):
    result={}
    for step,row in acc.items():
        count=max(row["count"],1); models={}
        for name in MODEL_NAMES:
            models[name]={"pb_rmse_m":math.sqrt(row[f"{name}_p"]/count),
                          "vb_rmse_mps":math.sqrt(row[f"{name}_v"]/count)}
        nominal=models["nominal"]
        for name in MODEL_NAMES[1:]:
            models[name]["pb_mse_improvement"] = 1-models[name]["pb_rmse_m"]**2/max(nominal["pb_rmse_m"]**2,1e-15)
            models[name]["vb_mse_improvement"] = 1-models[name]["vb_rmse_mps"]**2/max(nominal["vb_rmse_mps"]**2,1e-15)
        result[f"h{step}"]={"count":row["count"],"horizon_s":row["dt"]/count,"models":models}
    return result


def _acceptance(validation,test,model,config):
    checks={}
    v1_rmse=validation["point"]["dynamic"]["v1"]["rmse"]
    checks["validation_dynamic_vs_v1"] = validation["point"]["dynamic"]["v1_5"]["rmse"] <= (1+config.acceptance.dynamic_v1_degradation_max)*v1_rmse
    for scope,value in (("validation",validation),("test",test)):
        h=value["rollout"]["overall"]["h30"]["models"]["v1_5"]
        checks[f"{scope}_h30_position"] = h["pb_mse_improvement"] >= config.acceptance.rollout_improvement_min
        checks[f"{scope}_h30_velocity"] = h["vb_mse_improvement"] >= config.acceptance.rollout_improvement_min
        static=value["static"]["v1_5"]
        checks[f"{scope}_static_median"] = static["median_abs_mps2"] is not None and static["median_abs_mps2"] < config.acceptance.static_median_max_mps2
        checks[f"{scope}_static_p95"] = static["p95_abs_mps2"] is not None and static["p95_abs_mps2"] < config.acceptance.static_p95_max_mps2
        checks[f"{scope}_static_false_motion"] = static["false_motion_rate"] is not None and static["false_motion_rate"] < config.acceptance.static_false_motion_rate_max
        for trial_id, trial_value in value["rollout"]["by_trial"].items():
            if trial_value["reference_type"] == "constant": continue
            h30=trial_value["metrics"]["h30"]["models"]
            checks[f"{scope}_{trial_id}_no_degradation"] = max(
                h30["v1_5"]["pb_rmse_m"]/max(h30["nominal"]["pb_rmse_m"],1e-15),
                h30["v1_5"]["vb_rmse_mps"]/max(h30["nominal"]["vb_rmse_mps"],1e-15)) <= 1+config.acceptance.per_dynamic_degradation_max
        checks[f"{scope}_saturation"] = value["diagnostics"]["mlp_saturation_rate"] < config.acceptance.saturation_rate_max
        checks[f"{scope}_finite"] = value["diagnostics"]["outputs_finite"] and value["diagnostics"]["gradients_finite"]
    threshold=float(model.deadzone_threshold.detach()); span=model.deadzone_max_mps2-model.deadzone_min_mps2
    margin=config.acceptance.deadzone_boundary_margin_fraction*span
    checks["deadzone_not_at_boundary"] = model.deadzone_min_mps2+margin < threshold < model.deadzone_max_mps2-margin
    return {"passed":bool(all(checks.values())),"checks":{k:bool(v) for k,v in checks.items()}}


def _regression(pred,target):
    if not len(target): return {"count":0,"rmse":None,"mae":None,"bias":None,"r2":None}
    error=pred-target; den=float(np.square(target-target.mean()).sum())
    return {"count":int(len(target)),"rmse":float(np.sqrt(np.mean(error**2))),
            "mae":float(np.mean(np.abs(error))),"bias":float(np.mean(error)),
            "r2":float(1-np.square(error).sum()/max(den,1e-15))}


def _plot_scope(summary,figures_dir,label):
    figures_dir.mkdir(parents=True,exist_ok=True)
    h=summary["rollout"]["overall"]; keys=sorted(h,key=lambda x:int(x[1:])); x=np.arange(len(keys)); width=.19
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for mi,name in enumerate(MODEL_NAMES):
        for axis,state in zip(axes,("pb_rmse_m","vb_rmse_mps")):
            axis.bar(x+(mi-1.5)*width,[h[k]["models"][name][state] for k in keys],width,label=name)
    for axis,ylabel in zip(axes,("Position RMSE (m)","Velocity RMSE (m/s)")):
        axis.set_xticks(x,[f"{h[k]['horizon_s']:.2f}s" for k in keys]); axis.set_ylabel(ylabel); axis.grid(axis="y",alpha=.25)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figures_dir/f"rollout_{label}.png",dpi=160); plt.close(fig)
    static=summary["static"]; fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(MODEL_NAMES,[static[n]["median_abs_mps2"] or 0 for n in MODEL_NAMES],label="median")
    ax.scatter(MODEL_NAMES,[static[n]["p95_abs_mps2"] or 0 for n in MODEL_NAMES],color="black",label="P95")
    ax.set_ylabel("Static |predicted acceleration| (m/s²)"); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(figures_dir/f"static_{label}.png",dpi=160); plt.close(fig)


def _plot_training(run_dir,figures_dir):
    figures_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,name,title in zip(axes,("history_dynamic.csv","history_deadzone.csv"),("Dynamic MLP","Deadzone")):
        path=run_dir/name
        if path.is_file():
            data=np.atleast_1d(np.genfromtxt(path,delimiter=",",names=True)); axis.semilogy(data["epoch"],data["train_total_loss"],label="train"); axis.semilogy(data["epoch"],data["validation_total_loss"],label="validation")
        axis.set_title(title); axis.set_xlabel("Epoch"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(figures_dir/"training_history.png",dpi=160); plt.close(fig)


def _write_metrics_csv(path,summary):
    rows=[]
    for scope in ("validation","test","stress"):
        value=summary[scope]
        for mode,metrics in value["point"]["dynamic"].items(): rows.append({"scope":scope,"category":"dynamic_point","model":mode,**metrics})
        for horizon,entry in value["rollout"]["overall"].items():
            for mode,metrics in entry["models"].items(): rows.append({"scope":scope,"category":"rollout","model":mode,"horizon":horizon,"horizon_s":entry["horizon_s"],**metrics})
        for trial_id, trial in value["point"]["by_trial"].items():
            for mode,metrics in trial["total_acceleration"].items():
                rows.append({"scope":scope,"category":"trial_point","trial":trial_id,
                             "group":trial["group"],"model":mode,**metrics})
        for bin_name, bin_value in value["point"]["by_position_bin"].items():
            for mode,metrics in bin_value["models"].items():
                rows.append({"scope":scope,"category":"position_bin","bin":bin_name,
                             "model":mode,**metrics})
        for speed_mode, models in value["point"]["by_speed_mode"].items():
            for mode,metrics in models.items():
                rows.append({"scope":scope,"category":"speed_mode","bin":speed_mode,
                             "model":mode,**metrics})
    keys=sorted({k for row in rows for k in row}); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f: writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _quantile(value,q): return float(np.quantile(value,q)) if value.size else None
def _breakaway_mask(trial, config):
    static = trial.arrays["static_mask"]; transition = trial.arrays["transition_mask"]
    velocity = np.abs(trial.arrays["vb"]); result = np.zeros(trial.size, dtype=bool)
    lookback = config.data.static_persistence_samples
    for index in np.flatnonzero(transition & (velocity > config.data.static_speed_mps)):
        result[index] = bool(static[max(0, index-lookback):index].any())
    return result
def _sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def _read_json(path): return json.loads(path.read_text())
def _write_json(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
