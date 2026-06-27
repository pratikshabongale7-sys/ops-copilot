"""Tests for the incident dataset + loader.

These assume you've generated the dataset first:
    uv run python data/generate_incidents.py

They check the dataset is well-formed and that the loader can hide the answer key
(important: the agent must never see the label).
"""

import pytest

from app.incidents import DATA_DIR, list_incidents, load_incident

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="dataset not generated yet — run: uv run python data/generate_incidents.py",
)


def test_dataset_has_incidents():
    incidents = list_incidents()
    assert len(incidents) >= 20  # we generate 24


def test_every_incident_is_well_formed():
    for inc in list_incidents():
        assert inc.root_cause, f"{inc.incident_id} missing root_cause"
        assert inc.true_service, f"{inc.incident_id} missing service"
        assert inc.logs, f"{inc.incident_id} has no logs"
        assert inc.metrics.get("series"), f"{inc.incident_id} has no metric series"
        assert inc.split in {"train", "test"}


def test_train_test_split_exists():
    assert list_incidents(split="train")
    assert list_incidents(split="test")


def test_loader_can_hide_the_answer_key():
    # production view: telemetry only, no label leaked to the agent
    any_folder = next(p for p in DATA_DIR.iterdir() if p.is_dir())
    inc = load_incident(any_folder, with_label=False)
    assert inc.label == {}
    assert inc.logs  # but telemetry is still there
    assert inc.metrics
