"""Run the agent on one incident and print its diagnosis.

Usage:
    uv run python scripts/diagnose.py inc_001
    uv run python scripts/diagnose.py inc_012 --truth   # also show the real answer

The --truth flag reveals the label AFTER diagnosing, so you can eyeball whether
the agent got it right. (The agent itself never sees the label.)
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

# make the `app` package importable when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agent import diagnose  # noqa: E402
from app.incidents import load_incident, resolve_incident_folder  # noqa: E402


def _print_diagnosis(inc_id, dx) -> None:
    print("\n" + "=" * 60)
    print(f"DIAGNOSIS for {inc_id}")
    print("=" * 60)
    print(f"  root cause : {dx.root_cause}")
    print(f"  service    : {dx.service}")
    print(f"  confidence : {dx.confidence:.2f}")
    print(f"  fix        : {dx.suggested_fix}")
    print("  evidence   :")
    for e in dx.evidence:
        print(f"    - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose an incident with the agent.")
    parser.add_argument("incident_id", help="e.g. inc_001")
    parser.add_argument(
        "--truth", action="store_true", help="reveal the true label after diagnosing"
    )
    args = parser.parse_args()

    dx = asyncio.run(diagnose(args.incident_id))
    _print_diagnosis(args.incident_id, dx)

    if args.truth:
        folder = resolve_incident_folder(args.incident_id)
        label = load_incident(folder).label
        correct = label.get("root_cause") == dx.root_cause
        print("\n" + "-" * 60)
        print(f"  TRUE cause : {label.get('root_cause')}  ({label.get('service')})")
        print(f"  agent was  : {'CORRECT ✅' if correct else 'WRONG ❌'}")
        print("-" * 60)

    print("\nOpen https://smith.langchain.com to watch the full investigation trace.")


if __name__ == "__main__":
    main()
