"""Deterministic two-stage training; epoch-boundary exact resume."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import numpy as np
import torch
from .config import ROOT, MODEL_NAMES, configuration, config_hash, load_config
from .data import load_manifest, make_store, evaluation_references, window_hash, window_records, sha256_file
from .models import create_model
from .losses import _loss, _selection_metrics


def seed_all(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def runtime_sources():
    files = [ROOT / name for name in ("__init__.py", "config.py", "data.py", "dynamics.py", "losses.py", "train.py")]
    files += list((ROOT / "models").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)}


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(name, resume=False):
    if name == "nominal":
        return
    cfg = configuration()
    seed_all(cfg["seed"])
    manifest = load_manifest()
    source_hashes = runtime_sources()
    identity = {"config_hash": config_hash(), "sources": source_hashes,
                "manifest_sha256": sha256_file(ROOT / "data/processed/manifest.json")}
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    run = ROOT / "runs" / f"{name}_seed7_{run_hash[:12]}"
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run / "last.pt"
    if (run / "summary.json").exists():
        print(f"{name}: already complete: {run}", flush=True)
        return
    if checkpoint_path.exists() and not resume:
        raise RuntimeError(f"Run exists; use --resume: {run}")
    train_store, val_store = make_store("train", 33), make_store("validation", 33)
    val_refs = evaluation_references(val_store, 512)
    metadata = {**identity, "model": name, "run_hash": run_hash, "configuration": cfg,
                "dataset_hash": manifest["dataset_hash"], "train_window_hash": window_hash(train_store, train_store.references),
                "validation_window_hash": window_hash(val_store, val_refs),
                "train_windows": len(train_store.references), "validation_windows": len(val_refs),
                "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                "device": "cpu", "threads": torch.get_num_threads(), "dtype": "float64",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (run / "config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (run / "validation_windows.json").write_text(json.dumps(window_records(val_store, val_refs), indent=2)+"\n")
    model = create_model(name, manifest)
    # Match the original reseeding immediately before _fit.
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    best_state, best_score, best_epoch = copy.deepcopy(model.state_dict()), math.inf, None
    history, elapsed_before, start_stage, start_epoch, stale, resume_optimizer = [], 0., 0, 1, 0, None
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=False)
        assert saved["metadata"] == metadata, "Resume metadata changed"
        model.load_state_dict(saved["state_dict"])
        best_state, best_score, best_epoch = saved["best_state"], saved["best_score"], saved["best_epoch"]
        history, elapsed_before = saved["history"], saved["elapsed_s"]
        start_stage, start_epoch, stale = saved["next_stage"], saved["next_epoch"], saved["stale"]
        resume_optimizer = saved["optimizer"]
        rng.bit_generator.state = saved["numpy_generator_state"]
        torch.set_rng_state(saved["torch_rng_state"])
        random.setstate(saved["python_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
    began = time.monotonic()
    for stage_index in range(start_stage, len(cfg["stages"])):
        stage = cfg["stages"][stage_index]
        optimizer = torch.optim.Adam(model.parameters(), lr=stage["lr"])
        if stage_index == start_stage and resume_optimizer is not None:
            optimizer.load_state_dict(resume_optimizer)
        else:
            stale = 0
        first_epoch = start_epoch if stage_index == start_stage else 1
        for epoch in range(first_epoch, stage["epochs"]+1):
            model.train()
            accumulated = 0.
            for _ in range(stage["steps"]):
                refs = train_store.sample(rng, cfg["batch_size"])
                batch = train_store.make_batch(refs, torch.device("cpu"))
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, batch, load_config(), rollout_weight=stage["rollout_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {name}, {stage_index}, {epoch}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                model.project()
                accumulated += float(loss.detach())
            model.eval()
            metrics = _selection_metrics(model, val_store, val_refs, load_config())
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Nonfinite validation metric")
            if metrics["score"] < best_score - cfg["min_improvement"]:
                best_score, best_state, best_epoch = metrics["score"], copy.deepcopy(model.state_dict()), [stage["name"], epoch]
                stale = 0
            else:
                stale += 1
            history.append({"stage": stage["name"], "epoch": epoch, "train_loss": accumulated/stage["steps"],
                            **metrics, "best_score": best_score})
            end_stage = stale >= cfg["patience"] or epoch == stage["epochs"]
            if end_stage:
                model.load_state_dict(best_state)
            save_checkpoint(checkpoint_path, {"metadata": metadata, "state_dict": model.state_dict(),
                "best_state": best_state, "best_score": best_score, "best_epoch": best_epoch,
                "history": history, "elapsed_s": elapsed_before + time.monotonic()-began,
                "next_stage": stage_index+1 if end_stage else stage_index,
                "next_epoch": 1 if end_stage else epoch+1, "stale": 0 if end_stage else stale,
                "optimizer": None if end_stage else optimizer.state_dict(),
                "numpy_generator_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state()})
            (run / "history.json").write_text(json.dumps(history, indent=2)+"\n")
            print(f"{name} {stage['name']} {epoch}: loss={accumulated/stage['steps']:.6g} val={metrics['score']:.9f} best={best_score:.9f}", flush=True)
            if end_stage:
                break
    model.load_state_dict(best_state)
    elapsed = elapsed_before + time.monotonic()-began
    save_checkpoint(run / "best.pt", {"metadata": metadata, "state_dict": best_state,
                                     "best_score": best_score, "best_epoch": best_epoch})
    summary = {"model": name, "run": str(run.relative_to(ROOT)), "best_validation_score": best_score,
               "best_epoch": best_epoch, "elapsed_s": elapsed,
               "epochs": {s["name"]: sum(h["stage"]==s["name"] for h in history) for s in cfg["stages"]},
               "checkpoint_sha256": sha256_file(run / "best.pt"), "metadata": metadata}
    (run / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (ROOT / "runs" / f"{name}_completed.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"COMPLETE {name}: {best_score:.12f} ({elapsed:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    names = ("structured", "profile_only", "friction_only", "rbf_pv", "mlp") if args.model=="all" else (args.model,)
    for name in names:
        train(name, args.resume)

if __name__ == "__main__":
    main()
