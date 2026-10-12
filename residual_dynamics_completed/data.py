"""Independent processed-data loader, queue reconstruction and common sampler."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Mapping
import numpy as np
import torch
from .config import ROOT, load_config
from .dynamics import TorchPhysicalConstants, TorchGeometryConstants, TorchResponseConstants

@dataclass(frozen=True)
class PhysicalConstants:
    plank_length_m: float
    rope_length_m: float
    ball_position_offset_m: float
    gravity_mps2: float
    acceleration_gain: float

    def as_dict(self) -> dict[str, float]:
        return {
            "plank_length_m": self.plank_length_m,
            "rope_length_m": self.rope_length_m,
            "ball_position_offset_m": self.ball_position_offset_m,
            "gravity_mps2": self.gravity_mps2,
            "acceleration_gain": self.acceleration_gain,
        }


@dataclass(frozen=True)
class ResponseConstants:
    enabled: bool
    tau_s: float
    gain: float
    bias: float
    max_velocity_mps: float
    delay_steps: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "enabled": self.enabled,
            "tau_s": self.tau_s,
            "gain": self.gain,
            "bias": self.bias,
            "max_velocity_mps": self.max_velocity_mps,
            "delay_steps": self.delay_steps,
        }


@dataclass
class TrialData:
    trial_id: str
    group: str
    session: str
    split: str
    reference_type: str
    source_file: str
    arrays: dict[str, np.ndarray]
    constants: PhysicalConstants
    response: ResponseConstants | None = None
    end_reason: str = "tracking_duration"
    is_failure_prefix: bool = False
    terminal_time_s: float | None = None
    raw_sample_count: int = 0
    truncated_sample_count: int = 0
    policy_model_id: str = ""
    policy_scale: float = 0.0
    source_weights_sha256: str = ""

    @property
    def size(self) -> int:
        return int(self.arrays["time_s"].size)

    @property
    def trial_uid(self) -> str:
        return f"{self.session}/{self.group}/{self.trial_id}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_processed_trials(config: AppConfig, split: str | None = None) -> list[TrialData]:
    """Load processed trial arrays without consulting the raw log directory."""

    manifest = load_manifest()
    trials: list[TrialData] = []
    for entry in manifest["entries"]:
        if not entry.get("included") or (split is not None and entry["split"] != split):
            continue
        path = config.paths.processed_dir / entry["processed_path"]
        if sha256_file(path) != entry["processed_sha256"]:
            raise RuntimeError(f"Processed trial hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        constants_dict = entry["constants"]
        constants = PhysicalConstants(**constants_dict)
        response_dict = entry.get("response")
        response = None if response_dict is None else ResponseConstants(**response_dict)
        trials.append(
            TrialData(
                trial_id=entry["trial_id"],
                group=entry["group"],
                session=entry["session"],
                split=entry["split"],
                reference_type=_reference_type(entry["group"]),
                source_file=entry["samples_path"],
                arrays=arrays,
                constants=constants,
                response=response,
                end_reason=str(entry.get("end_reason", "tracking_duration")),
                is_failure_prefix=bool(entry.get("is_failure_prefix", False)),
                terminal_time_s=entry.get("terminal_time_s"),
                raw_sample_count=int(entry.get("raw_sample_count", arrays["time_s"].size)),
                truncated_sample_count=int(
                    entry.get("truncated_sample_count", arrays["time_s"].size)
                ),
                policy_model_id=str(entry.get("policy_model_id", "")),
                policy_scale=float(entry.get("policy_scale", 0.0)),
                source_weights_sha256=str(entry.get("source_weights_sha256", "")),
            )
        )
    if not trials:
        raise RuntimeError(f"No processed trials found for split={split!r}.")
    return trials


def _reference_type(group: str) -> str:
    if group.startswith("constant"):
        return "constant"
    return group.split("_", maxsplit=1)[0]


def position_bins(values: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    edge_array = np.asarray(tuple(edges), dtype=np.float64)
    result = np.searchsorted(edge_array, values, side="right") - 1
    result[values == edge_array[-1]] = edge_array.size - 2
    outside = (values < edge_array[0]) | (values > edge_array[-1])
    result[outside] = -1
    return result.astype(np.int64)


def velocity_bins(values: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    edge_array = np.asarray(tuple(edges), dtype=np.float64)
    result = np.searchsorted(edge_array, values, side="right") - 1
    # The central bin is closed on both sides by contract.
    central_lower = int(np.flatnonzero(np.isclose(edge_array, -0.02))[0])
    result[np.isclose(values, -0.02)] = central_lower
    result[np.isclose(values, 0.02)] = central_lower
    return np.clip(result, 0, edge_array.size - 2).astype(np.int64)


def actuation_regimes(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(values < -threshold, -1, np.where(values > threshold, 1, 0)).astype(
        np.int8
    )


@dataclass
class WindowBatch:
    features: torch.Tensor
    residual_target: torch.Tensor
    pb: torch.Tensor
    vb: torch.Tensor
    theta: torch.Tensor
    omega: torch.Tensor
    dt: torch.Tensor
    constants: TorchPhysicalConstants
    causal_state: torch.Tensor | None = None
    smooth_pb: torch.Tensor | None = None
    smooth_vb: torch.Tensor | None = None
    commands: torch.Tensor | None = None
    geometry: TorchGeometryConstants | None = None
    response: TorchResponseConstants | None = None
    state_scales: torch.Tensor | None = None


class BatchBuilder:
    def __init__(self, trials, horizon, config, normalization):
        self.trials = list(trials)
        self.horizon = int(horizon)
        self.causal = True
        self.state_scales = np.asarray(normalization["state_scales"], dtype=np.float64)
        self._cache = {}

    def tensors(self, device: torch.device) -> list[dict[str, torch.Tensor]]:
        key = str(device)
        if key not in self._cache:
            names = ["features", "residual_target", "pb", "vb", "theta", "omega", "dt_s"]
            if self.causal:
                names.extend(
                    (
                        "causal_state",
                        "smooth_pb",
                        "smooth_vb",
                        "command_z",
                        "pending_commands",
                        "response_command",
                    )
                )
            self._cache[key] = [
                {
                    name: torch.as_tensor(
                        trial.arrays[name], device=device, dtype=torch.float64
                    )
                    for name in names
                }
                for trial in self.trials
            ]
        return self._cache[key]

    def make_batch(
        self,
        references: Sequence[WindowReference],
        device: torch.device,
    ) -> WindowBatch:
        tensors = self.tensors(device)
        horizon = self.horizon

        def stack(name: str, include_endpoint: bool = True) -> torch.Tensor:
            length = horizon + 1 if include_endpoint else horizon
            return torch.stack(
                [
                    tensors[ref.trial_index][name][ref.start : ref.start + length]
                    for ref in references
                ],
                dim=0,
            )

        constants = [self.trials[ref.trial_index].constants for ref in references]
        torch_constants = TorchPhysicalConstants(
            plank_length_m=torch.tensor(
                [item.plank_length_m for item in constants], device=device, dtype=torch.float64
            ),
            ball_position_offset_m=torch.tensor(
                [item.ball_position_offset_m for item in constants],
                device=device,
                dtype=torch.float64,
            ),
            gravity_mps2=torch.tensor(
                [item.gravity_mps2 for item in constants], device=device, dtype=torch.float64
            ),
            acceleration_gain=torch.tensor(
                [item.acceleration_gain for item in constants],
                device=device,
                dtype=torch.float64,
            ),
        )
        batch = WindowBatch(
            features=stack("features"),
            residual_target=stack("residual_target"),
            pb=stack("pb"),
            vb=stack("vb"),
            theta=stack("theta"),
            omega=stack("omega"),
            dt=stack("dt_s", include_endpoint=False),
            constants=torch_constants,
        )
        if not self.causal:
            return batch

        pending = stack("pending_commands")[:, 0, :]
        delay_steps = pending.shape[1]
        committed = stack("command_z")
        commands = reconstruct_deployment_commands(pending, committed, horizon)
        expected_commands = stack("response_command", include_endpoint=False)
        if not torch.allclose(commands, expected_commands, rtol=0.0, atol=1.0e-9):
            error = torch.max(torch.abs(commands - expected_commands)).item()
            raise RuntimeError(
                f"Pending-command reconstruction disagrees with logged queue (max={error:.3e})."
            )
        responses = [self.trials[ref.trial_index].response for ref in references]
        if any(item is None for item in responses):
            raise RuntimeError("Causal rollout trial is missing response metadata.")
        response_items = [item for item in responses if item is not None]
        if any(item.delay_steps != delay_steps for item in response_items):
            raise RuntimeError("Window pending queue length disagrees with response metadata.")
        batch.causal_state = stack("causal_state")
        batch.smooth_pb = stack("smooth_pb")
        batch.smooth_vb = stack("smooth_vb")
        batch.commands = commands
        batch.geometry = TorchGeometryConstants(
            plank_length_m=torch.tensor(
                [item.plank_length_m for item in constants], device=device, dtype=torch.float64
            ),
            rope_length_m=torch.tensor(
                [item.rope_length_m for item in constants], device=device, dtype=torch.float64
            ),
        )
        batch.response = TorchResponseConstants(
            enabled=torch.tensor(
                [item.enabled for item in response_items], device=device, dtype=torch.bool
            ),
            tau_s=torch.tensor(
                [item.tau_s for item in response_items], device=device, dtype=torch.float64
            ),
            gain=torch.tensor(
                [item.gain for item in response_items], device=device, dtype=torch.float64
            ),
            bias=torch.tensor(
                [item.bias for item in response_items], device=device, dtype=torch.float64
            ),
            max_velocity_mps=torch.tensor(
                [item.max_velocity_mps for item in response_items],
                device=device,
                dtype=torch.float64,
            ),
        )
        batch.state_scales = torch.as_tensor(
            self.state_scales, device=device, dtype=torch.float64
        )
        return batch


def reconstruct_deployment_commands(
    initial_pending: torch.Tensor,
    committed: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Consume the initial ring buffer, then future post-window commits."""

    if initial_pending.ndim != 2 or committed.ndim != 2:
        raise ValueError("initial_pending and committed must be batched matrices.")
    if initial_pending.shape[0] != committed.shape[0] or horizon < 1:
        raise ValueError("Invalid batch size or rollout horizon.")
    pending_count = min(horizon, initial_pending.shape[1])
    remaining = horizon - pending_count
    if committed.shape[1] < remaining + 1:
        raise ValueError("Committed command sequence is too short for this horizon.")
    parts = [initial_pending[:, :pending_count]]
    if remaining > 0:
        parts.append(committed[:, 1 : remaining + 1])
    return torch.cat(parts, dim=1)


@dataclass(frozen=True)
class WindowReference:
    trial_index: int
    start: int
    position_bin: int
    velocity_bin: int
    actuation_regime: int


class WindowStore:
    """Filter causal windows and provide the approved three-stream sampler."""

    def __init__(self, trials: Sequence[Any], horizon: int, config: Config, manifest: Mapping[str, Any]):
        self.trials = list(trials)
        self.horizon = int(horizon)
        self.config = config
        self.legacy = BatchBuilder(
            self.trials, self.horizon, config.legacy, dict(manifest["normalization"])
        )
        self.references: list[WindowReference] = []
        for trial_index, trial in enumerate(self.trials):
            valid = trial.arrays["rbf_valid"].astype(bool)
            segments = trial.arrays["segment_id"]
            for start in range(0, trial.size - self.horizon):
                endpoint = start + self.horizon
                if segments[start] != segments[endpoint]:
                    continue
                if not bool(valid[start : endpoint + 1].all()):
                    continue
                middle = start + self.horizon // 2
                pbin = int(
                    position_bins(
                        np.asarray([trial.arrays["pb"][middle]]),
                        config.legacy.data.position_bin_edges_m,
                    )[0]
                )
                if pbin < 0:
                    continue
                vbin = int(
                    velocity_bins(
                        np.asarray([trial.arrays["vb"][middle]]),
                        config.sampling.velocity_bin_edges_mps,
                    )[0]
                )
                act = int(
                    actuation_regimes(
                        np.asarray([trial.arrays["a_nom"][middle]]),
                        config.sampling.actuation_threshold_mps2,
                    )[0]
                )
                self.references.append(WindowReference(trial_index, start, pbin, vbin, act))
        if not self.references:
            raise RuntimeError("No valid RBF windows were found.")
        self.by_joint: dict[tuple[int, int], dict[str, dict[str, list[WindowReference]]]] = {}
        self.by_actuation: dict[
            tuple[int, int], dict[int, dict[str, dict[str, list[WindowReference]]]]
        ] = {}
        for ref in self.references:
            trial = self.trials[ref.trial_index]
            key = (ref.position_bin, ref.velocity_bin)
            self.by_joint.setdefault(key, {}).setdefault(trial.session, {}).setdefault(
                trial.group, []
            ).append(ref)
            self.by_actuation.setdefault(key, {}).setdefault(
                ref.actuation_regime, {}
            ).setdefault(trial.session, {}).setdefault(trial.group, []).append(ref)

    def make_batch(self, references: Sequence[WindowReference], device: torch.device):
        return self.legacy.make_batch(references, device)

    @staticmethod
    def _hierarchical_pick(
        rng: np.random.Generator,
        value: dict[str, dict[str, list[WindowReference]]],
    ) -> WindowReference:
        sessions = sorted(value)
        session = sessions[int(rng.integers(len(sessions)))]
        groups = sorted(value[session])
        group = groups[int(rng.integers(len(groups)))]
        candidates = value[session][group]
        return candidates[int(rng.integers(len(candidates)))]

    def sample(self, rng: np.random.Generator, batch_size: int) -> list[WindowReference]:
        empirical = int(round(batch_size * self.config.sampling.empirical_fraction))
        joint = int(round(batch_size * self.config.sampling.joint_fraction))
        actuation = batch_size - empirical - joint
        result = [
            self.references[int(rng.integers(len(self.references)))] for _ in range(empirical)
        ]
        joint_keys = sorted(self.by_joint)
        for _ in range(joint):
            key = joint_keys[int(rng.integers(len(joint_keys)))]
            result.append(self._hierarchical_pick(rng, self.by_joint[key]))
        act_keys = sorted(self.by_actuation)
        for _ in range(actuation):
            key = act_keys[int(rng.integers(len(act_keys)))]
            regimes = sorted(self.by_actuation[key])
            regime = regimes[int(rng.integers(len(regimes)))]
            result.append(self._hierarchical_pick(rng, self.by_actuation[key][regime]))
        rng.shuffle(result)
        return result



def load_manifest():
    return json.loads((ROOT / "data/processed/manifest.json").read_text())

def load_trials(split):
    if split not in ("train", "validation", "test", "stress"):
        raise ValueError(split)
    return load_processed_trials(load_config(), split)

def evaluation_references(store, maximum=2048):
    refs = store.references[::5]
    if len(refs) > maximum:
        refs = [refs[i] for i in np.linspace(0, len(refs)-1, maximum, dtype=int)]
    return refs

def window_records(store, refs):
    return [{"trial_uid": store.trials[r.trial_index].trial_uid, "start": r.start}
            for r in refs]

def window_hash(store, refs):
    return hashlib.sha256(json.dumps(window_records(store, refs), sort_keys=True).encode()).hexdigest()

def make_store(split, horizon):
    return WindowStore(load_trials(split), horizon, load_config(), load_manifest())
