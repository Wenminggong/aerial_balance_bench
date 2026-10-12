"""Failure-tolerant Weights & Biases monitoring for offline training."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Mapping

from .config import WandbConfig


class WandbMonitor:
    """Log aggregate metrics without making W&B a training dependency at runtime."""

    def __init__(self, config: WandbConfig, directory: Path) -> None:
        self.config = config
        self.directory = directory
        self.module: Any | None = None
        self.run: Any | None = None
        self.actual_mode = "disabled"
        self.run_id: str | None = None
        self.run_url: str | None = None
        self.initialization_error: str | None = None

    def start(self, run_name: str, run_config: Mapping[str, Any]) -> None:
        if not self.config.enabled or self.config.mode == "disabled":
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.initialization_error = f"wandb import failed: {exc}"
            return
        self.module = wandb
        os.environ.setdefault("WANDB_INIT_TIMEOUT", str(self.config.init_timeout_s))
        os.environ.setdefault("WANDB_SILENT", "true")
        requested_mode = "online" if self.config.mode == "auto" else self.config.mode
        if self.config.mode == "auto" and not self._online_reachable():
            requested_mode = "offline"
            self.initialization_error = "online preflight failed; using offline mode"
        try:
            self.run = self._init_run(requested_mode, run_name, run_config)
            self.actual_mode = requested_mode
        except Exception as exc:
            self.initialization_error = f"{type(exc).__name__}: {exc}"
            self._safe_finish(exit_code=1)
            if self.config.mode not in {"auto", "offline"}:
                return
            try:
                self.run = self._init_run("offline", run_name, run_config)
                self.actual_mode = "offline"
            except Exception as offline_exc:  # pragma: no cover - rare package failure
                self.initialization_error += (
                    f"; offline fallback failed: {type(offline_exc).__name__}: {offline_exc}"
                )
                self.run = None
                self.actual_mode = "disabled"
                return
        self.run_id = str(getattr(self.run, "id", "")) or None
        url = getattr(self.run, "url", None)
        self.run_url = str(url) if url else None

    def _online_reachable(self) -> bool:
        """Bound online initialization when DNS or outbound HTTPS is unavailable."""

        timeout = min(3.0, float(self.config.init_timeout_s))
        try:
            with socket.create_connection(("api.wandb.ai", 443), timeout=timeout):
                return True
        except OSError:
            return False

    def _init_run(
        self,
        mode: str,
        run_name: str,
        run_config: Mapping[str, Any],
    ) -> Any:
        assert self.module is not None
        kwargs = {
            "project": self.config.project,
            "entity": self.config.entity,
            "group": self.config.group,
            "tags": list(self.config.tags),
            "name": run_name,
            "config": dict(run_config),
            "dir": str(self.directory),
            "mode": mode,
            "reinit": True,
        }
        return self.module.init(**kwargs)

    def log(self, values: Mapping[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(dict(values), step=step)
        except Exception:
            return

    def log_histograms(self, model: Any, epoch: int) -> None:
        if self.run is None or self.module is None:
            return
        values: dict[str, Any] = {"epoch": epoch}
        try:
            for name, parameter in model.named_parameters():
                values[f"parameters/{name}"] = self.module.Histogram(
                    parameter.detach().cpu().numpy()
                )
                if parameter.grad is not None:
                    values[f"gradients/{name}"] = self.module.Histogram(
                        parameter.grad.detach().cpu().numpy()
                    )
            self.run.log(values)
        except Exception:
            return

    def summary(self, values: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        try:
            for key, value in values.items():
                self.run.summary[key] = value
        except Exception:
            return

    def finish(self, exit_code: int = 0) -> None:
        self._safe_finish(exit_code=exit_code)

    def _safe_finish(self, exit_code: int) -> None:
        if self.run is None:
            return
        try:
            self.run.finish(exit_code=exit_code)
        except TypeError:
            try:
                self.run.finish()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self.run = None

    def state(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "requested_mode": self.config.mode,
            "actual_mode": self.actual_mode,
            "run_id": self.run_id,
            "run_url": self.run_url,
            "initialization_error": self.initialization_error,
        }
