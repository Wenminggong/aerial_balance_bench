"""Two-stage deterministic training for dynamic residual MLP and frozen-MLP deadzone."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import resource
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

from .config import AppConfig
from .data import PhysicalConstants, TrialData, load_manifest, load_trials
from .dynamics import TorchPhysicalConstants, rollout_ball_dynamics
from .model import (SmoothDeadzoneResidualModel, checkpoint_model_metadata, create_model,
                    load_checkpoint, save_portable_weights)
from .wandb_monitor import WandbMonitor


@dataclass(frozen=True)
class WindowReference:
    trial_index: int
    start: int
    regime: int


@dataclass
class WindowBatch:
    features: torch.Tensor
    residual_target: torch.Tensor
    pb: torch.Tensor
    vb: torch.Tensor
    ab: torch.Tensor
    theta: torch.Tensor
    omega: torch.Tensor
    dt: torch.Tensor
    dynamic_mask: torch.Tensor
    static_mask: torch.Tensor
    transition_mask: torch.Tensor
    constants: TorchPhysicalConstants


class WindowStore:
    def __init__(self, trials: Sequence[TrialData], horizon: int, dynamic_only: bool) -> None:
        self.trials = list(trials)
        self.horizon = int(horizon)
        self.dynamic_only = bool(dynamic_only)
        self.references = _build_references(self.trials, self.horizon, self.dynamic_only)
        if not self.references:
            raise RuntimeError(f"No {'dynamic' if dynamic_only else 'valid'} windows found")
        self.by_trial_regime: dict[int, dict[int, list[WindowReference]]] = {}
        for ref in self.references:
            self.by_trial_regime.setdefault(ref.trial_index, {}).setdefault(ref.regime, []).append(ref)
        self._cache: dict[str, list[dict[str, torch.Tensor]]] = {}

    def tensors(self, device: torch.device) -> list[dict[str, torch.Tensor]]:
        key = str(device)
        if key not in self._cache:
            names = ("features", "residual_target", "pb", "vb", "ab", "theta", "omega",
                     "dt_s", "dynamic_mask", "static_mask", "transition_mask")
            self._cache[key] = [
                {name: torch.as_tensor(trial.arrays[name], device=device,
                                       dtype=torch.bool if name.endswith("_mask") else torch.float64)
                 for name in names}
                for trial in self.trials
            ]
        return self._cache[key]

    def balanced_references(self, rng: np.random.Generator, batch_size: int) -> list[WindowReference]:
        trial_ids = tuple(sorted(self.by_trial_regime))
        result: list[WindowReference] = []
        for _ in range(batch_size):
            trial = trial_ids[int(rng.integers(len(trial_ids)))]
            regimes = tuple(sorted(self.by_trial_regime[trial]))
            regime = regimes[int(rng.integers(len(regimes)))]
            candidates = self.by_trial_regime[trial][regime]
            result.append(candidates[int(rng.integers(len(candidates)))])
        return result

    def make_batch(self, refs: Sequence[WindowReference], device: torch.device) -> WindowBatch:
        cache = self.tensors(device)
        horizon = self.horizon
        def stack(name: str, endpoint: bool = True) -> torch.Tensor:
            length = horizon + 1 if endpoint else horizon
            return torch.stack([cache[r.trial_index][name][r.start:r.start + length] for r in refs])
        constants = [self.trials[r.trial_index].constants for r in refs]
        tc = TorchPhysicalConstants(
            *[torch.tensor([getattr(value, field) for value in constants], device=device,
                           dtype=torch.float64) for field in (
                               "plank_length_m", "ball_position_offset_m",
                               "gravity_mps2", "acceleration_gain")]
        )
        return WindowBatch(
            stack("features"), stack("residual_target"), stack("pb"), stack("vb"),
            stack("ab"), stack("theta"), stack("omega"), stack("dt_s", False),
            stack("dynamic_mask"), stack("static_mask"), stack("transition_mask"), tc,
        )


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _rollout_loss(model: SmoothDeadzoneResidualModel, batch: WindowBatch,
                  config: AppConfig, mode: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_pb, pred_vb = rollout_ball_dynamics(
        model, batch.pb[:, 0], batch.vb[:, 0], batch.theta, batch.omega,
        batch.dt, batch.constants, mode=mode,
    )
    p_loss = torch.zeros((), device=pred_pb.device, dtype=torch.float64)
    v_loss = torch.zeros_like(p_loss)
    details: dict[str, torch.Tensor] = {}
    for step, weight in zip(config.data.rollout_steps, config.loss.horizon_weights):
        p_mse = (pred_pb[:, step] - batch.pb[:, step]).square().mean()
        v_mse = (pred_vb[:, step] - batch.vb[:, step]).square().mean()
        p_loss += weight * p_mse / config.loss.position_scale_m**2
        v_loss += weight * v_mse / config.loss.velocity_scale_mps**2
        details[f"position_rmse_h{step}"] = torch.sqrt(p_mse)
        details[f"velocity_rmse_h{step}"] = torch.sqrt(v_mse)
    details.update(position_rollout_loss=p_loss, velocity_rollout_loss=v_loss)
    return p_loss + v_loss, details


def compute_dynamic_loss(model: SmoothDeadzoneResidualModel, batch: WindowBatch,
                         config: AppConfig, include_jacobian: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rollout, details = _rollout_loss(model, batch, config, "dynamic")
    flat_features = batch.features.reshape(-1, 5)
    target = batch.residual_target.reshape(-1)
    prediction = model.mlp(flat_features)
    point = F.smooth_l1_loss(prediction, target, beta=config.loss.huber_beta_mps2)

    abs_velocity = flat_features[:, 1].abs()
    zero, positive, negative = flat_features.clone(), flat_features.clone(), flat_features.clone()
    zero[:, 1] = 0.0; positive[:, 1] = abs_velocity; negative[:, 1] = -abs_velocity
    anchor_target = 0.5 * (model.mlp(positive) + model.mlp(negative))
    anchor = F.smooth_l1_loss(model.mlp(zero), anchor_target,
                              beta=config.loss.huber_beta_mps2)
    jacobian = torch.zeros_like(point)
    if include_jacobian and config.loss.jacobian_weight > 0:
        normalized = ((batch.features[:, 0] - model.mlp.input_mean)
                      / model.mlp.input_std).detach().requires_grad_(True)
        output = model.mlp.normalized_residual(normalized)
        gradient = torch.autograd.grad(output.sum(), normalized, create_graph=True)[0]
        jacobian = gradient.square().sum(dim=-1).mean()
    total = (rollout + config.loss.point_weight * point
             + config.loss.zero_velocity_anchor_weight * anchor
             + config.loss.jacobian_weight * jacobian)
    error = prediction - target
    details.update(total_loss=total, point_residual_loss=point,
                   zero_velocity_anchor_loss=anchor, jacobian_loss=jacobian,
                   residual_rmse=torch.sqrt(error.square().mean()),
                   residual_mae=error.abs().mean(), residual_bias=error.mean(),
                   residual_mean=prediction.mean(), residual_std=prediction.std(),
                   residual_saturation_rate=(prediction.abs() >= 0.95 * model.mlp.output_limit_mps2).double().mean())
    return total, details


def compute_deadzone_loss(model: SmoothDeadzoneResidualModel, batch: WindowBatch,
                          config: AppConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rollout, details = _rollout_loss(model, batch, config, "composite")
    output = model(batch.features)
    static = batch.static_mask
    if static.any():
        static_error = output.total_acceleration[static] - batch.ab[static]
        static_loss = (static_error / config.acceptance.static_p95_max_mps2).square().mean()
        smooth_abs = torch.sqrt(output.drive_acceleration[static].square() + 1e-12)
        tau = model.deadzone_smoothing_mps2
        violation = tau * F.softplus((smooth_abs - model.deadzone_threshold) / tau)
        stick_bound = (violation / config.acceptance.static_p95_max_mps2).square().mean()
        false_motion = (output.total_acceleration[static].abs() > 0.02).double().mean()
    else:
        static_loss = torch.zeros_like(rollout); stick_bound = torch.zeros_like(rollout)
        false_motion = torch.zeros_like(rollout)
    total = rollout + config.loss.static_weight * static_loss + config.loss.stick_bound_weight * stick_bound
    details.update(total_loss=total, static_loss=static_loss, stick_bound_loss=stick_bound,
                   static_false_motion_rate=false_motion,
                   deadzone_threshold_mps2=model.deadzone_threshold,
                   deadzone_suppression_mean=output.deadzone_suppression.abs().mean())
    return total, details


def train_model(config: AppConfig) -> Path:
    manifest = load_manifest(config, "fit")
    train_trials, validation_trials = load_trials(config, "train"), load_trials(config, "validation")
    normalization = manifest["normalization"]
    horizon = max(config.data.rollout_steps)
    dynamic_train = WindowStore(train_trials, horizon, True)
    dynamic_validation = WindowStore(validation_trials, horizon, True)
    all_train = WindowStore(train_trials, horizon, False)
    all_validation = WindowStore(validation_trials, horizon, False)
    dynamic_val_refs = _validation_refs(dynamic_validation.references, config)
    all_val_refs = _validation_refs(all_validation.references, config)

    set_deterministic_seed(config.training.seed)
    benchmark_rng = np.random.default_rng(config.training.seed)
    device, benchmark = select_training_device(
        config, normalization, dynamic_train,
        dynamic_train.balanced_references(benchmark_rng, config.training.batch_size),
        all_train, all_train.balanced_references(benchmark_rng, config.training.batch_size),
    )
    set_deterministic_seed(config.training.seed)
    rng = np.random.default_rng(config.training.seed)
    model = create_model(config.model, normalization).to(device=device, dtype=torch.float64)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"v1_5_seed7_{config.config_hash[:8]}_{timestamp}"
    run_dir = config.paths.training_runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    dynamic_path = run_dir / "best_dynamic_checkpoint.pt"
    composite_path = run_dir / "best_composite_checkpoint.pt"
    monitor = WandbMonitor(config.wandb, config.paths.wandb_dir)
    monitor.start(run_name, {"config": _safe(config.raw), "config_hash": config.config_hash,
                             "fit_dataset_hash": manifest["dataset_hash"], "seed": 7,
                             "device": str(device), "device_benchmark": benchmark})
    started = time.perf_counter()
    try:
        dynamic_result = _train_dynamic(model, dynamic_train, dynamic_validation,
                                        dynamic_val_refs, device, rng, config,
                                        dynamic_path, run_dir, manifest, benchmark, monitor)
        model, _ = load_checkpoint(dynamic_path, device)
        mlp_hash_before = _module_hash(model.mlp)
        deadzone_result = _train_deadzone(model, all_train, all_validation, all_val_refs,
                                          device, rng, config, composite_path, run_dir,
                                          manifest, benchmark, monitor,
                                          dynamic_result["global_step"])
        model, composite_checkpoint = load_checkpoint(composite_path, "cpu")
        mlp_hash_after = _module_hash(model.mlp)
        if mlp_hash_before != mlp_hash_after:
            raise RuntimeError("MLP weights changed during frozen deadzone training")
        portable_metadata = {
            "config_hash": config.config_hash, "fit_dataset_hash": manifest["dataset_hash"],
            "dynamic_best_epoch": dynamic_result["best_epoch"],
            "deadzone_best_epoch": deadzone_result["best_epoch"],
            "training_device": str(device), "wandb": monitor.state(),
            "mlp_sha256": mlp_hash_after,
        }
        portable = run_dir / "weights_v1_5.npz"
        save_portable_weights(model, portable, portable_metadata)
        summary = {
            "schema_version": 1, "run_name": run_name,
            "run_dir": relative_path(run_dir, config.paths.training_runs_dir.parent.parent),
            "config_hash": config.config_hash, "fit_dataset_hash": manifest["dataset_hash"],
            "seed": 7, "training_device": str(device), "device_benchmark": benchmark,
            "dynamic": dynamic_result, "deadzone": deadzone_result,
            "deadzone_threshold_mps2": float(model.deadzone_threshold.detach()),
            "mlp_hash_before_deadzone": mlp_hash_before, "mlp_hash_after_deadzone": mlp_hash_after,
            "mlp_frozen_unchanged": mlp_hash_before == mlp_hash_after,
            "total_training_time_s": time.perf_counter() - started,
            "dynamic_checkpoint_sha256": _sha256(dynamic_path),
            "composite_checkpoint_sha256": _sha256(composite_path),
            "weights_sha256": _sha256(portable), "wandb": monitor.state(),
        }
        _write_json(run_dir / "training_summary.json", summary)
        _write_json(config.paths.training_runs_dir / "latest.json", {"run_dir": str(run_dir)})
        monitor.summary({"deadzone_threshold_mps2": summary["deadzone_threshold_mps2"],
                         "total_training_time_s": summary["total_training_time_s"],
                         "mlp_frozen_unchanged": True})
        monitor.finish(0)
        return run_dir
    except Exception:
        monitor.finish(1)
        raise


def _train_dynamic(model, train_store, validation_store, validation_refs, device, rng,
                   config, checkpoint_path, run_dir, manifest, benchmark, monitor):
    model.set_mlp_trainable(True); model.raw_deadzone_threshold.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=config.training.dynamic_learning_rate,
                                  weight_decay=config.training.dynamic_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.training.dynamic_lr_factor,
        patience=config.training.dynamic_lr_patience)
    history, best, best_epoch, stale = [], math.inf, 0, 0
    global_step = 0; stop_reason = "max_epochs"
    for epoch in range(1, config.training.dynamic_max_epochs + 1):
        model.train(); sums: dict[str, float] = {}; epoch_start = time.perf_counter()
        for _ in range(config.training.steps_per_epoch):
            batch = train_store.make_batch(train_store.balanced_references(rng, config.training.batch_size), device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = compute_dynamic_loss(model, batch, config, True)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite dynamic loss")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), config.training.gradient_clip_norm)
            optimizer.step(); global_step += 1; _accumulate(sums, details)
            if global_step % config.wandb.log_batch_interval == 0:
                monitor.log({**{f"dynamic/train/{k}": float(v.detach().cpu()) for k,v in details.items()},
                             "dynamic/gradient_norm": float(grad), "dynamic/epoch": epoch,
                             "optimizer/learning_rate": optimizer.param_groups[0]["lr"]}, global_step)
        train_metrics = {k: v / config.training.steps_per_epoch for k,v in sums.items()}
        val = _validate(model, validation_store, validation_refs, config, device, "dynamic")
        scheduler.step(val["total_loss"])
        if val["total_loss"] < best - 1e-12:
            best, best_epoch, stale = val["total_loss"], epoch, 0
            _save_checkpoint(checkpoint_path, model, config, manifest, "dynamic", epoch, val, benchmark, monitor.state())
        else: stale += 1
        row = {"epoch": epoch, "global_step": global_step, "epoch_time_s": time.perf_counter()-epoch_start,
               "learning_rate": optimizer.param_groups[0]["lr"], "best_epoch": best_epoch,
               **{f"train_{k}":v for k,v in train_metrics.items()}, **{f"validation_{k}":v for k,v in val.items()}}
        history.append(row); _write_history(run_dir / "history_dynamic.csv", history)
        monitor.log({"dynamic/epoch": epoch, **{f"dynamic/train_epoch/{k}":v for k,v in train_metrics.items()},
                     **{f"dynamic/validation/{k}":v for k,v in val.items()},
                     "dynamic/best_epoch": best_epoch, "dynamic/early_stopping_counter": stale,
                     "timing/dynamic_epoch_s": row["epoch_time_s"],
                     **_memory_metrics(device)}, global_step)
        if epoch % config.wandb.log_histogram_interval_epochs == 0: monitor.log_histograms(model.mlp, epoch)
        if stale >= config.training.dynamic_early_stopping_patience:
            stop_reason = "early_stopping"; break
    return {"best_epoch": best_epoch, "best_validation_loss": best,
            "epochs_completed": len(history), "stop_reason": stop_reason,
            "global_step": global_step}


def _train_deadzone(model, train_store, validation_store, validation_refs, device, rng,
                    config, checkpoint_path, run_dir, manifest, benchmark, monitor,
                    start_global_step):
    model.set_mlp_trainable(False); model.raw_deadzone_threshold.requires_grad_(True)
    optimizer = torch.optim.Adam([model.raw_deadzone_threshold], lr=config.training.deadzone_learning_rate)
    history, best, best_epoch, stale = [], math.inf, 0, 0
    global_step = int(start_global_step); stop_reason = "max_epochs"
    for epoch in range(1, config.training.deadzone_max_epochs + 1):
        model.train(); sums: dict[str,float] = {}; epoch_start = time.perf_counter()
        for _ in range(config.training.steps_per_epoch):
            batch = train_store.make_batch(train_store.balanced_references(rng, config.training.batch_size), device)
            optimizer.zero_grad(set_to_none=True); loss, details = compute_deadzone_loss(model, batch, config)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite deadzone loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_([model.raw_deadzone_threshold],
                                                   config.training.gradient_clip_norm)
            optimizer.step(); global_step += 1; _accumulate(sums, details)
            if global_step % config.wandb.log_batch_interval == 0:
                monitor.log({
                    **{f"deadzone/train/{k}": float(v.detach().cpu())
                       for k, v in details.items()},
                    "deadzone/gradient_norm": float(grad), "deadzone/epoch": epoch,
                    "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                }, global_step)
        train_metrics = {k:v/config.training.steps_per_epoch for k,v in sums.items()}
        val = _validate(model, validation_store, validation_refs, config, device, "composite")
        if val["total_loss"] < best - 1e-12:
            best, best_epoch, stale = val["total_loss"], epoch, 0
            _save_checkpoint(checkpoint_path, model, config, manifest, "composite", epoch, val, benchmark, monitor.state())
        else: stale += 1
        row = {"epoch":epoch, "global_step":global_step, "epoch_time_s":time.perf_counter()-epoch_start,
               "deadzone_threshold_mps2":float(model.deadzone_threshold.detach().cpu()), "best_epoch":best_epoch,
               **{f"train_{k}":v for k,v in train_metrics.items()}, **{f"validation_{k}":v for k,v in val.items()}}
        history.append(row); _write_history(run_dir / "history_deadzone.csv", history)
        monitor.log({"deadzone/epoch":epoch, **{f"deadzone/train/{k}":v for k,v in train_metrics.items()},
                     **{f"deadzone/validation/{k}":v for k,v in val.items()},
                     "deadzone/best_epoch":best_epoch, "deadzone/early_stopping_counter":stale,
                     "timing/deadzone_epoch_s":row["epoch_time_s"],
                     **_memory_metrics(device)}, global_step)
        if stale >= config.training.deadzone_early_stopping_patience:
            stop_reason = "early_stopping"; break
    return {"best_epoch":best_epoch, "best_validation_loss":best,
            "epochs_completed":len(history), "stop_reason":stop_reason,
            "global_step":global_step}


def _validate(model, store, refs, config, device, phase):
    model.eval(); sums: dict[str,float] = {}; count = 0
    context = torch.enable_grad() if phase == "dynamic" else torch.no_grad()
    with context:
        for offset in range(0, len(refs), config.training.batch_size):
            part = refs[offset:offset+config.training.batch_size]
            batch = store.make_batch(part, device)
            _, details = (compute_dynamic_loss(model, batch, config, False) if phase == "dynamic"
                          else compute_deadzone_loss(model, batch, config))
            for key,value in details.items(): sums[key] = sums.get(key,0)+float(value.detach().cpu())*len(part)
            count += len(part)
    return {k:v/count for k,v in sums.items()}


def select_training_device(config, normalization, dynamic_store, dynamic_refs, all_store, all_refs):
    info = {"requested":config.runtime.device, "cuda_available":torch.cuda.is_available(),
            "cpu_median_step_s":None, "cuda_median_step_s":None, "speedup":None, "reason":None}
    if config.runtime.device == "cpu" or not torch.cuda.is_available():
        info["reason"] = "cpu requested" if config.runtime.device == "cpu" else "CUDA unavailable"
        return torch.device("cpu"), info
    if config.runtime.device == "cuda": return torch.device("cuda"), {**info,"reason":"cuda requested"}
    try:
        cpu = _benchmark(torch.device("cpu"), config, normalization, dynamic_store, dynamic_refs, all_store, all_refs)
        cuda = _benchmark(torch.device("cuda"), config, normalization, dynamic_store, dynamic_refs, all_store, all_refs)
        speedup = cpu/cuda; info.update(cpu_median_step_s=cpu,cuda_median_step_s=cuda,speedup=speedup,
                                        gpu_name=torch.cuda.get_device_name(0))
        if speedup >= config.runtime.gpu_min_speedup:
            info["reason"]="CUDA met speedup threshold"; return torch.device("cuda"),info
        info["reason"]="CUDA did not meet speedup threshold"; return torch.device("cpu"),info
    except Exception as exc:
        info["reason"] = f"CUDA benchmark fallback: {type(exc).__name__}: {exc}"
        return torch.device("cpu"),info


def _benchmark(device, config, normalization, dynamic_store, dynamic_refs, all_store, all_refs):
    set_deterministic_seed(config.training.seed)
    model = create_model(config.model, normalization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    db = dynamic_store.make_batch(dynamic_refs, device); ab = all_store.make_batch(all_refs, device)
    times=[]; total=config.runtime.benchmark_warmup_steps+config.runtime.benchmark_steps
    for step in range(total):
        if device.type == "cuda": torch.cuda.synchronize(device)
        start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        ld,_=compute_dynamic_loss(model,db,config,True); la,_=compute_deadzone_loss(model,ab,config)
        (ld+la).backward(); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        if step>=config.runtime.benchmark_warmup_steps: times.append(time.perf_counter()-start)
    return float(np.median(times))


def _build_references(trials, horizon, dynamic_only):
    refs=[]
    for ti,trial in enumerate(trials):
        a=trial.arrays
        for start in range(0,trial.size-horizon):
            sl=slice(start,start+horizon+1)
            if a["segment_id"][start+horizon] != a["segment_id"][start] or not np.all(a["valid_mask"][sl]): continue
            if dynamic_only and not np.all(a["dynamic_mask"][sl]): continue
            if dynamic_only: regime=1 if a["vb"][start]>0 else -1
            elif a["static_mask"][start]: regime=0
            elif a["transition_mask"][start]: regime=1
            else: regime=2
            refs.append(WindowReference(ti,start,regime))
    return refs


def _validation_refs(refs, config):
    result=list(refs[::config.data.validation_stride])
    if len(result)>config.training.validation_max_windows:
        ids=np.linspace(0,len(result)-1,config.training.validation_max_windows,dtype=int)
        result=[result[i] for i in ids]
    return result


def _save_checkpoint(path, model, config, manifest, phase, epoch, validation, benchmark, wandb):
    torch.save({"schema_version":1,"phase":phase,"config_hash":config.config_hash,
                "fit_dataset_hash":manifest["dataset_hash"],"epoch":epoch,
                "model":checkpoint_model_metadata(model),"state_dict":copy.deepcopy(model.state_dict()),
                "validation_metrics":validation,"device_benchmark":benchmark,"wandb":wandb},path)


def resolve_run_dir(config: AppConfig, run_dir: str|Path|None) -> Path:
    if run_dir is None:
        value=json.loads((config.paths.training_runs_dir/"latest.json").read_text())["run_dir"]
        path=Path(value)
    else: path=Path(run_dir)
    path=path if path.is_absolute() else (config.paths.training_runs_dir/path)
    if not (path/"best_composite_checkpoint.pt").is_file(): raise FileNotFoundError(path)
    return path


def _module_hash(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _accumulate(acc,details):
    for k,v in details.items(): acc[k]=acc.get(k,0)+float(v.detach().cpu())


def _write_history(path,rows):
    keys=sorted({k for row in rows for k in row}); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def _sha256(path):
    h=hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def _safe(value):
    if isinstance(value,dict): return {k:_safe(v) for k,v in value.items() if "path" not in k.lower()}
    if isinstance(value,(list,tuple)): return [_safe(v) for v in value]
    return str(value) if isinstance(value,Path) else value


def _memory_metrics(device: torch.device) -> dict[str, float]:
    # Linux reports ru_maxrss in KiB. It is a peak, which is the useful signal here.
    metrics = {"runtime/cpu_peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    if device.type == "cuda":
        metrics.update({
            "runtime/gpu_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
            "runtime/gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        })
    return metrics


def relative_path(path,base):
    try:return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:return path.name
