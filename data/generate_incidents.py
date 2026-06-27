"""Phase 1 — synthetic incident dataset generator.

Why this exists
---------------
Your Ops Copilot needs realistic, *labeled* incidents to learn from and be tested
against. We can't use a real company's outages, so we manufacture them: for each
incident we know the true root cause (that's the label), and we generate the
logs / metrics / deploy history that such a failure would produce.

Each incident becomes one folder under data/incidents/:

    inc_001_bad_deploy/
        logs.jsonl     # one JSON log line per row, over a ~30 min window
        metrics.json   # time-series metrics per service (error rate, latency, ...)
        deploys.json   # recent deploy / config events
        label.yaml     # THE ANSWER: root cause, service, expected diagnosis, split

The data is realistic enough that finding the cause requires *correlating*
evidence (e.g. "errors spiked right when this deploy happened") — which is
exactly what the agent will have to do in Phase 3.

Run it:
    uv run python data/generate_incidents.py
    # writes ~24 incidents into data/incidents/

It is deterministic (fixed random seed) so you get the same dataset every time —
good MLOps hygiene and means your eval numbers are reproducible.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Service names mirror the OpenTelemetry Demo app, so when you later capture real
# chaos-injected incidents they slot into the same schema.
SERVICES = [
    "frontend",
    "checkout",
    "payment",
    "cart",
    "shipping",
    "product-catalog",
    "currency",
    "email",
]

WINDOW_MINUTES = 30          # each incident covers a 30-minute window
STEP_SECONDS = 60            # one metric data point per minute -> 30 points
POINTS = WINDOW_MINUTES      # 30 data points

# The 6 failure classes and how many of each to generate.
FAILURE_CLASSES = [
    "bad_deploy",
    "db_connection_pool_exhaustion",
    "memory_leak_oom",
    "downstream_timeout",
    "bad_config",
    "slow_query_saturation",
]
PER_CLASS = 4                # 6 classes x 4 = 24 incidents
HOLDOUT_INDEX = 3            # the 4th of each class (index 3) goes to the test split

BASE_SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "incidents"


# --------------------------------------------------------------------------- #
# Small helpers for building noisy time series
# --------------------------------------------------------------------------- #

def _noisy(base: float, jitter: float, rng: random.Random) -> float:
    """A baseline value plus a little random noise (never negative)."""
    return max(0.0, base + rng.uniform(-jitter, jitter))


def baseline_series(base: float, jitter: float, rng: random.Random) -> list[float]:
    """A flat, healthy metric series with noise — what 'normal' looks like."""
    return [round(_noisy(base, jitter, rng), 2) for _ in range(POINTS)]


def step_anomaly(
    base: float, peak: float, onset: int, jitter: float, rng: random.Random
) -> list[float]:
    """Normal until `onset`, then jumps to ~peak and stays elevated.
    Models a sudden failure (e.g. a bad deploy flips errors on)."""
    out = []
    for i in range(POINTS):
        level = base if i < onset else peak
        out.append(round(_noisy(level, jitter, rng), 2))
    return out


def ramp_anomaly(
    base: float, peak: float, onset: int, jitter: float, rng: random.Random
) -> list[float]:
    """Normal until `onset`, then climbs steadily toward `peak`.
    Models a gradual failure (e.g. a memory leak filling up)."""
    out = []
    for i in range(POINTS):
        if i < onset:
            level = base
        else:
            frac = (i - onset) / max(1, POINTS - 1 - onset)
            level = base + (peak - base) * frac
        out.append(round(_noisy(level, jitter, rng), 2))
    return out


# --------------------------------------------------------------------------- #
# Log + deploy builders
# --------------------------------------------------------------------------- #

def _ts(start: datetime, minute: int) -> str:
    return (start + timedelta(minutes=minute)).isoformat()


def normal_logs(start: datetime, service: str, rng: random.Random) -> list[dict]:
    """A handful of healthy INFO logs spread across the window (background noise)."""
    msgs = [
        "request handled",
        "health check ok",
        "cache hit",
        "processed message",
        "request handled",
    ]
    logs = []
    for minute in range(0, POINTS, max(2, POINTS // 6)):
        logs.append(
            {
                "ts": _ts(start, minute),
                "level": "INFO",
                "service": service,
                "message": rng.choice(msgs),
            }
        )
    return logs


def error_logs(
    start: datetime, service: str, onset: int, level: str, messages: list[str],
    rng: random.Random, count: int = 8,
) -> list[dict]:
    """Cluster of WARN/ERROR logs after the onset — the failure's fingerprint."""
    logs = []
    for _ in range(count):
        minute = rng.randint(onset, POINTS - 1)
        logs.append(
            {
                "ts": _ts(start, minute),
                "level": level,
                "service": service,
                "message": rng.choice(messages),
            }
        )
    return logs


# --------------------------------------------------------------------------- #
# Per-class incident builders
# Each returns (metrics_series_for_service, extra_logs, deploys, expected_diag, signals)
# --------------------------------------------------------------------------- #

def build_incident(failure_class: str, service: str, onset: int, version: str,
                   start: datetime, rng: random.Random):
    """Return the telemetry + answer for one incident of the given class."""
    # Healthy defaults for the affected service; classes override what matters.
    series = {
        "error_rate": baseline_series(0.5, 0.3, rng),       # errors/min
        "latency_p95_ms": baseline_series(120, 25, rng),
        "cpu_pct": baseline_series(35, 8, rng),
        "memory_mb": baseline_series(380, 30, rng),
        "db_active_connections": baseline_series(20, 5, rng),
        "request_rate": baseline_series(200, 30, rng),
    }
    deploys: list[dict] = []
    logs: list[dict] = []
    # `signals` is set in full inside every branch below (the else raises), so it
    # needs no default here — unlike deploys/logs, which get appended to.

    if failure_class == "bad_deploy":
        series["error_rate"] = step_anomaly(0.5, rng.uniform(18, 32), onset, 3, rng)
        series["latency_p95_ms"] = step_anomaly(120, 260, onset, 30, rng)
        deploys.append({
            "ts": _ts(start, onset), "service": service,
            "version": version, "change": "code release",
        })
        logs += error_logs(start, service, onset, "ERROR", [
            "HTTP 500 Internal Server Error",
            f"NullPointerException in {service.title()}Handler.process()",
            "unhandled exception while serving request",
        ], rng)
        expected = (
            f"Error rate on '{service}' jumped ~{round(series['error_rate'][-1]/max(0.1,series['error_rate'][0]))}x "
            f"within minutes of deploy {version} at {_ts(start, onset)}. The new release is the likely cause; "
            f"recommend rolling back {version}."
        )
        signals = [
            f"error_rate step-change at minute {onset}",
            f"deploy of {service} {version} at the same time",
            "new exception type appears in logs",
        ]

    elif failure_class == "db_connection_pool_exhaustion":
        series["error_rate"] = ramp_anomaly(0.5, rng.uniform(8, 16), onset, 2, rng)
        series["latency_p95_ms"] = ramp_anomaly(120, rng.uniform(900, 1600), onset, 60, rng)
        series["db_active_connections"] = step_anomaly(20, 100, onset, 1, rng)  # pinned at max
        logs += error_logs(start, service, onset, "ERROR", [
            "Timeout acquiring JDBC connection from pool",
            "HikariPool-1 - Connection is not available, request timed out after 30000ms",
            "could not get a connection from the pool",
        ], rng)
        expected = (
            f"'{service}' latency and errors climbed while DB active connections sat pinned at the pool max (100). "
            f"Connection-pool exhaustion is the likely cause; recommend raising pool size or fixing connection leaks."
        )
        signals = [
            "db_active_connections saturated at 100 (the pool limit)",
            "rising latency and errors (gradual, not a step)",
            "'connection pool / HikariPool timeout' log messages",
        ]

    elif failure_class == "memory_leak_oom":
        series["memory_mb"] = ramp_anomaly(380, 1024, onset, 20, rng)
        # OOM kill near the end: memory drops back to baseline (restart) and errors spike there
        restart = POINTS - 3
        for i in range(restart, POINTS):
            series["memory_mb"][i] = round(_noisy(360, 30, rng), 2)
        series["error_rate"] = step_anomaly(0.5, 14, restart, 3, rng)
        logs += error_logs(start, service, restart, "ERROR", [
            "java.lang.OutOfMemoryError: Java heap space",
            "Container killed due to OOMKilled (exit code 137)",
            "pod restarted: memory limit exceeded",
        ], rng)
        expected = (
            f"'{service}' memory climbed steadily to its ~1GB limit, then the pod was OOMKilled and restarted "
            f"near minute {restart} (memory dropped to baseline, errors spiked). A memory leak is the likely cause."
        )
        signals = [
            "memory_mb ramps steadily toward the limit",
            "sudden memory drop + error spike = an OOM restart",
            "'OutOfMemoryError / OOMKilled' in logs",
        ]

    elif failure_class == "downstream_timeout":
        # caller service is `service`; it depends on `dependency`
        dependency = rng.choice([s for s in SERVICES if s != service])
        series["error_rate"] = ramp_anomaly(0.5, rng.uniform(6, 12), onset, 2, rng)
        series["latency_p95_ms"] = step_anomaly(120, rng.uniform(1200, 2200), onset, 80, rng)
        logs += error_logs(start, service, onset, "ERROR", [
            f"Timeout calling downstream service '{dependency}' after 5000ms",
            f"upstream request to {dependency} timed out",
            f"circuit breaker OPEN for {dependency}",
        ], rng)
        expected = (
            f"'{service}' latency jumped and errors rose because calls to its dependency '{dependency}' "
            f"started timing out around minute {onset}. Root cause is downstream '{dependency}', not '{service}' itself."
        )
        signals = [
            f"latency step-change on {service} at minute {onset}",
            f"logs name '{dependency}' as the slow/timing-out dependency",
            "the symptom service is the caller, not the true cause",
        ]
        # stash dependency for label
        series["_dependency"] = dependency  # type: ignore

    elif failure_class == "bad_config":
        series["error_rate"] = step_anomaly(0.5, rng.uniform(40, 70), onset, 4, rng)  # almost everything fails
        series["request_rate"] = step_anomaly(200, 30, onset, 10, rng)  # successful traffic collapses
        deploys.append({
            "ts": _ts(start, onset), "service": service,
            "version": version, "change": "config update",
        })
        logs += error_logs(start, service, onset, "ERROR", [
            "Configuration error: required environment variable DB_URL is not set",
            "Failed to initialize application context",
            "FATAL: invalid configuration, shutting down",
        ], rng)
        expected = (
            f"'{service}' began failing almost all requests right after a config update ({version}) at minute {onset}; "
            f"logs show a missing/invalid env var. Root cause is the bad configuration change; recommend reverting it."
        )
        signals = [
            "near-total failure (very high error rate) with a sharp onset",
            f"a config-change deploy ({version}) at the onset",
            "'configuration error / missing env var' in logs",
        ]

    elif failure_class == "slow_query_saturation":
        series["cpu_pct"] = ramp_anomaly(35, rng.uniform(92, 99), onset, 3, rng)
        series["latency_p95_ms"] = ramp_anomaly(120, rng.uniform(700, 1300), onset, 50, rng)
        series["error_rate"] = ramp_anomaly(0.5, rng.uniform(3, 7), onset, 1.5, rng)
        logs += error_logs(start, service, onset, "WARN", [
            "Slow query detected: SELECT ... took 3280ms",
            "query exceeded threshold (2000ms)",
            "CPU saturation: throttling requests",
        ], rng)
        expected = (
            f"'{service}' CPU climbed toward saturation (~95%+) with rising latency and slow-query warnings from "
            f"minute {onset}. A slow/expensive query saturating resources is the likely cause; recommend query/index optimization."
        )
        signals = [
            "cpu_pct ramps to near 100% (saturation)",
            "rising latency correlated with CPU",
            "'slow query' warnings in logs",
        ]
    else:
        raise ValueError(failure_class)

    return series, logs, deploys, expected, signals


# --------------------------------------------------------------------------- #
# Assemble + write one incident
# --------------------------------------------------------------------------- #

def write_incident(idx: int, failure_class: str) -> str:
    rng = random.Random(BASE_SEED + idx)

    affected = rng.choice(SERVICES)
    onset = rng.randint(8, 18)               # failure starts mid-window
    version = f"v1.{rng.randint(2, 9)}.{rng.randint(0, 9)}"
    # a realistic-ish start time, each incident on a different day
    start = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(days=idx, hours=rng.randint(0, 20))

    series, extra_logs, deploys, expected, signals = build_incident(
        failure_class, affected, onset, version, start, rng
    )
    dependency = series.pop("_dependency", None) if "_dependency" in series else None

    # --- metrics.json: affected service + 2 healthy services for realism ---
    healthy_others = rng.sample([s for s in SERVICES if s != affected], 2)
    metrics_series = {affected: series}
    for s in healthy_others:
        metrics_series[s] = {
            "error_rate": baseline_series(0.5, 0.3, rng),
            "latency_p95_ms": baseline_series(120, 25, rng),
            "cpu_pct": baseline_series(35, 8, rng),
            "memory_mb": baseline_series(380, 30, rng),
            "db_active_connections": baseline_series(20, 5, rng),
            "request_rate": baseline_series(200, 30, rng),
        }

    metrics = {
        "window": {
            "start": start.isoformat(),
            "end": (start + timedelta(minutes=WINDOW_MINUTES)).isoformat(),
            "step_seconds": STEP_SECONDS,
        },
        "series": metrics_series,
    }

    # --- logs.jsonl: healthy logs for every service + the failure's error logs ---
    logs = []
    for s in metrics_series:
        logs += normal_logs(start, s, rng)
    logs += extra_logs
    logs.sort(key=lambda r: r["ts"])

    # --- deploys.json: a couple of unrelated recent deploys + any culprit deploy ---
    all_deploys = list(deploys)
    for _ in range(rng.randint(1, 2)):
        s = rng.choice([x for x in SERVICES if x != affected])
        all_deploys.append({
            "ts": _ts(start, rng.randint(0, 6)),
            "service": s,
            "version": f"v2.{rng.randint(0, 4)}.{rng.randint(0, 9)}",
            "change": "code release",
        })
    all_deploys.sort(key=lambda r: r["ts"])

    # --- label.yaml: the ground truth ---
    split = "test" if (idx % PER_CLASS) == HOLDOUT_INDEX else "train"
    label = {
        "incident_id": f"inc_{idx + 1:03d}",
        "root_cause": failure_class,
        "service": affected,
        "onset_minute": onset,
        "split": split,
        "expected_diagnosis": expected,
        "key_signals": signals,
    }
    if dependency:
        label["true_dependency"] = dependency

    # --- write files ---
    folder = OUT_DIR / f"inc_{idx + 1:03d}_{failure_class}"
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "logs.jsonl", "w") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    with open(folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(folder / "deploys.json", "w") as f:
        json.dump(all_deploys, f, indent=2)
    with open(folder / "label.yaml", "w") as f:
        yaml.safe_dump(label, f, sort_keys=False)

    return f"{folder.name}  (service={affected}, onset=min {onset}, split={split})"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = 0
    summary = []
    for failure_class in FAILURE_CLASSES:
        for _ in range(PER_CLASS):
            summary.append(write_incident(idx, failure_class))
            idx += 1

    n_test = sum(1 for s in summary if "split=test" in s)
    print(f"Generated {idx} incidents into {OUT_DIR}")
    print(f"  train: {idx - n_test}   test: {n_test}")
    print()
    for line in summary:
        print("  " + line)


if __name__ == "__main__":
    main()
