from __future__ import annotations

import sys
import types
from pathlib import Path

from residual_dynamics_v1_5.config import WandbConfig
from residual_dynamics_v1_5.wandb_monitor import WandbMonitor


class FakeRun:
    def __init__(self, mode: str) -> None:
        self.id = f"id-{mode}"
        self.url = f"https://example.invalid/{mode}"
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.finished = False

    def log(self, values: dict, step: int | None = None) -> None:
        self.logged.append({**values, "_step": step})

    def finish(self, exit_code: int = 0) -> None:
        self.finished = True


def _config(mode: str = "auto") -> WandbConfig:
    return WandbConfig(True, mode, "project", None, "group", ("tag",), 20, 10, 1)


def test_online_success_and_no_artifact_upload(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("wandb")
    run = FakeRun("online")
    module.init = lambda **kwargs: run  # type: ignore[attr-defined]
    module.Histogram = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    monitor = WandbMonitor(_config(), tmp_path)
    monkeypatch.setattr(monitor, "_online_reachable", lambda: True)
    monitor.start("run", {"seed": 7, "fit_dataset_hash": "abc"})
    monitor.log({"dynamic/train/total_loss": 1.0}, step=20)
    assert monitor.actual_mode == "online"
    assert run.logged[0]["dynamic/train/total_loss"] == 1.0
    assert not hasattr(run, "log_artifact")
    monitor.finish()
    assert run.finished


def test_online_failure_falls_back_offline(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("wandb")
    modes: list[str] = []
    def init(**kwargs):
        modes.append(kwargs["mode"])
        if kwargs["mode"] == "online":
            raise RuntimeError("network unavailable")
        return FakeRun("offline")
    module.init = init  # type: ignore[attr-defined]
    module.Histogram = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    monitor = WandbMonitor(_config(), tmp_path)
    monkeypatch.setattr(monitor, "_online_reachable", lambda: True)
    monitor.start("run", {"seed": 7})
    assert modes == ["online", "offline"]
    assert monitor.actual_mode == "offline"
    assert "network unavailable" in (monitor.initialization_error or "")


def test_disabled_mode_does_not_import_or_write(tmp_path: Path) -> None:
    monitor = WandbMonitor(WandbConfig(False, "disabled", "p", None, "g", (), 20, 10, 1), tmp_path)
    monitor.start("run", {})
    assert monitor.state()["actual_mode"] == "disabled"
    assert list(tmp_path.iterdir()) == []
