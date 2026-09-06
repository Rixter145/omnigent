"""Harness wrap for the official, subscription-backed Gemini CLI."""

from __future__ import annotations

import math
import os

from fastapi import FastAPI

from omnigent.inner.executor import Executor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter

from .gemini_cli_executor import GeminiCliExecutor

_ENV_TIMEOUT = "HARNESS_GEMINI_CLI_TIMEOUT_S"
_ENV_MODEL = "HARNESS_GEMINI_CLI_MODEL"


def _timeout() -> float:
    raw = os.environ.get(_ENV_TIMEOUT, "120").strip()
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    return value if math.isfinite(value) and value > 0 else 120.0


def _build_gemini_cli_executor() -> Executor:
    # GeminiCliTransport discovers only the official ``gemini`` executable and
    # owns Google-account OAuth; this wrap never reads or forwards keys.
    return GeminiCliExecutor(
        timeout_s=_timeout(),
        model=os.environ.get(_ENV_MODEL, "").strip() or None,
    )


def create_app() -> FastAPI:
    """Build the shared HTTP harness app; executor construction stays lazy."""
    return ExecutorAdapter(
        executor_factory=_build_gemini_cli_executor,
        harness_label="Gemini CLI",
    ).build()


__all__ = ["create_app"]
