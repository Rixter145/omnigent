"""Safe local readiness and explicitly opt-in subscription verification.

``python dev/verify_subscription_mvp.py`` never sends a provider prompt.  Add
``--providers`` with one or more supported names to opt into bounded probes.
The command prints stable fields only; provider output and credential values are
never emitted.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnigent.onboarding.subscription_readiness import (
    MAX_PROBE_TIMEOUT_S,
    SUPPORTED_PROVIDERS,
    canonical_provider,
    subscription_readiness_map,
    verify_subscription_provider,
)


def _provider_names(values: Sequence[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        for item in value.split(","):
            name = canonical_provider(item)
            if name not in names:
                names.append(name)
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        metavar="NAME",
        help=(
            "explicitly probe claude, codex, or cursor-wsl; gemini-cli returns "
            "the permanent Google subscription-policy gate without sending a prompt"
        ),
    )
    parser.add_argument(
        "--no-api-keys",
        action="store_true",
        help="ban competing API-key environment variables for every probe",
    )
    parser.add_argument(
        "--cursor-distro",
        help="explicit Cursor WSL distro; absence is reported as a setup gate",
    )
    parser.add_argument(
        "--cursor-user",
        help="explicit non-root Cursor Linux user; absence is reported as a setup gate",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="probe timeout in seconds")
    return parser


def _summary(readiness: object) -> str:
    data = readiness.to_dict()
    fields = (
        "provider",
        "state",
        "installed",
        "auth_present",
        "auth_verified",
        "transport_ready",
        "last_verified",
        "reason",
    )
    return " ".join(
        f"{field}={data[field] if data[field] is not None else 'none'}" for field in fields
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        names = _provider_names(args.providers or SUPPORTED_PROVIDERS)
    except ValueError as exc:
        print(f"provider=unknown state=not-ready reason={exc}")
        return 2
    if args.timeout <= 0 or args.timeout > MAX_PROBE_TIMEOUT_S:
        print(
            "provider=unknown state=not-ready reason=timeout must be between "
            f"0 and {MAX_PROBE_TIMEOUT_S:g} seconds"
        )
        return 2

    common = {"cursor_distro": args.cursor_distro, "cursor_user": args.cursor_user}
    if args.providers:
        results = [
            verify_subscription_provider(
                provider,
                timeout_s=args.timeout,
                no_api_keys=args.no_api_keys,
                **common,
            )
            for provider in names
        ]
        for result in results:
            print(_summary(result))
        if results and all(result.transport_ready and result.auth_verified for result in results):
            print("SUBSCRIPTION_MVP_LIVE_OK")
            return 0
        return 1

    results = subscription_readiness_map(**common)
    for provider in names:
        print(_summary(results[provider]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
