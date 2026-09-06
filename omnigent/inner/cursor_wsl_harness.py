"""Harness wrap for Cursor's official CLI subscription login in WSL."""

from __future__ import annotations

import math
import os

from fastapi import FastAPI

from omnigent.inner.executor import Executor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter

from .cursor_wsl_executor import CursorWslExecutor

_ENV_DISTRO = "HARNESS_CURSOR_WSL_DISTRO"
_ENV_USER = "HARNESS_CURSOR_WSL_USER"
_ENV_CWD = "HARNESS_CURSOR_WSL_CWD"
_ENV_TIMEOUT = "HARNESS_CURSOR_WSL_TIMEOUT_S"
_ENV_MODEL = "HARNESS_CURSOR_WSL_MODEL"


def _timeout() -> float:
    raw = os.environ.get(_ENV_TIMEOUT, "120").strip()
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    return value if math.isfinite(value) and value > 0 else 120.0


def _build_cursor_wsl_executor() -> Executor:
    # Missing values are intentional: the executor reports an actionable
    # unavailable result instead of guessing an identity, distro, or workspace.
    return CursorWslExecutor(
        os.environ.get(_ENV_DISTRO, "").strip() or None,
        os.environ.get(_ENV_USER, "").strip() or None,
        os.environ.get(_ENV_CWD, "").strip() or None,
        timeout_s=_timeout(),
        model=os.environ.get(_ENV_MODEL, "").strip() or None,
    )


def create_app() -> FastAPI:
    """Build the shared HTTP harness app; executor construction stays lazy."""
    return ExecutorAdapter(
        executor_factory=_build_cursor_wsl_executor,
        harness_label="Cursor WSL",
    ).build()


__all__ = ["create_app"]
