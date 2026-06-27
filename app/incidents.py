"""Load and query the incident dataset.

This is the shared "data access" layer. Later phases use it:
  - the MCP tools (Phase 2) read logs/metrics/deploys through these helpers
  - the agent (Phase 3) is handed an Incident
  - the eval (Phase 4) iterates over the test split and compares to labels

Keeping all file-reading in one place means the rest of the codebase never has
to know how incidents are stored on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# data/incidents lives next to this app/ package, one level up
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "incidents"


@dataclass
class Incident:
    """One incident: its id, the raw telemetry, and (for eval) its label."""

    incident_id: str
    folder: Path
    logs: list[dict]
    metrics: dict
    deploys: list[dict]
    label: dict = field(default_factory=dict)

    # convenience accessors -------------------------------------------------- #
    @property
    def services(self) -> list[str]:
        return list(self.metrics.get("series", {}).keys())

    @property
    def root_cause(self) -> str:
        """The ground-truth answer. The AGENT must never read this — only eval does."""
        return self.label.get("root_cause", "")

    @property
    def true_service(self) -> str:
        return self.label.get("service", "")

    @property
    def split(self) -> str:
        return self.label.get("split", "train")


def load_incident(folder: str | Path, *, with_label: bool = True) -> Incident:
    """Load a single incident folder into an Incident object.

    Set with_label=False to simulate what the agent sees in production: telemetry
    only, no answer key.
    """
    folder = Path(folder)

    logs = []
    logs_path = folder / "logs.jsonl"
    if logs_path.exists():
        with open(logs_path) as f:
            logs = [json.loads(line) for line in f if line.strip()]

    metrics = {}
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    deploys = []
    deploys_path = folder / "deploys.json"
    if deploys_path.exists():
        deploys = json.loads(deploys_path.read_text())

    label = {}
    if with_label:
        label_path = folder / "label.yaml"
        if label_path.exists():
            label = yaml.safe_load(label_path.read_text()) or {}

    return Incident(
        incident_id=label.get("incident_id", folder.name),
        folder=folder,
        logs=logs,
        metrics=metrics,
        deploys=deploys,
        label=label,
    )


def list_incidents(
    split: str | None = None, root: str | Path = DATA_DIR
) -> list[Incident]:
    """Load all incidents, optionally filtered to 'train' or 'test'."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"No incidents found at {root}. Run: uv run python data/generate_incidents.py"
        )
    incidents = [load_incident(p) for p in sorted(root.iterdir()) if p.is_dir()]
    if split:
        incidents = [i for i in incidents if i.split == split]
    return incidents


if __name__ == "__main__":
    # Quick sanity check: print a one-line summary of every incident.
    all_inc = list_incidents()
    print(f"{len(all_inc)} incidents loaded from {DATA_DIR}\n")
    for inc in all_inc:
        print(
            f"  {inc.incident_id:>9}  cause={inc.root_cause:<28} "
            f"service={inc.true_service:<16} split={inc.split:<5} "
            f"logs={len(inc.logs):>3} services={len(inc.services)}"
        )
