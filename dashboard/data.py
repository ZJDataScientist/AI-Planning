from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd

@dataclass
class ProjectData:
    root: Path
    qwen_summary: dict[str, Any] | None = None
    qwen_cached_summary: dict[str, Any] | None = None
    qwen_summary_path: Path | None = None
    qwen_cached_summary_path: Path | None = None
    qwen_compute_frame: pd.DataFrame | None = None

    def qwen_profile_frame(self) -> pd.DataFrame:
        if not self.qwen_summary:
            return pd.DataFrame()
        p = self.qwen_summary.get("per_profile_summary", {})
        rows = []
        for profile in ["U0","U1","U2","U3"]:
            block = p.get(profile, {})
            reachable = block.get("reachable", {})
            unreachable = block.get("unreachable", {})
            success = (reachable.get("success") or {}).get("proportion")
            planning = reachable.get("planning_cost") or {}
            missed = (unreachable.get("missed_dead_end") or {}).get("proportion")
            rows.append({
                "profile": profile,
                "reachable_episodes": reachable.get("n_episodes"),
                "reachable_success_pct": None if success is None else 100*float(success),
                "planning_cost_mean": planning.get("mean"),
                "planning_cost_median": planning.get("median"),
                "unreachable_episodes": unreachable.get("n_episodes"),
                "missed_dead_end_pct": None if missed is None else 100*float(missed),
            })
        return pd.DataFrame(rows)

def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _wall_clock(summary):
    if not summary:
        return None
    return (summary.get("metadata") or {}).get("wall_clock_seconds")

def load_project_data(root: Path) -> ProjectData:
    primary = root / "results/phase5a/qwen3_14b/hpc/nocache_sensitivity/merged/llm_summary.json"
    primary_shard = root / "results/phase5a/qwen3_14b/hpc/nocache_sensitivity/shard_0_of_1/llm_summary.json"
    cached = root / "results/phase5a/qwen3_14b/hpc/merged/llm_summary.json"
    cached_shard = root / "results/phase5a/qwen3_14b/hpc/shards/shard_0_of_1/llm_summary.json"

    qwen = _load_json(primary) or _load_json(primary_shard)
    qwen_cached = _load_json(cached) or _load_json(cached_shard)
    primary_wc = _wall_clock(_load_json(primary_shard)) or _wall_clock(qwen)
    cached_wc = _wall_clock(_load_json(cached_shard)) or _wall_clock(qwen_cached)

    rows = []
    if cached_wc:
        rows.append({"run":"Cached","wall_clock_minutes":cached_wc/60})
    if primary_wc:
        rows.append({"run":"No cache","wall_clock_minutes":primary_wc/60})

    return ProjectData(
        root=root,
        qwen_summary=qwen,
        qwen_cached_summary=qwen_cached,
        qwen_summary_path=primary if primary.exists() else (primary_shard if primary_shard.exists() else None),
        qwen_cached_summary_path=cached if cached.exists() else (cached_shard if cached_shard.exists() else None),
        qwen_compute_frame=pd.DataFrame(rows),
    )

def load_algorithm_seed_metrics(root: Path) -> pd.DataFrame:
    p = root / "dashboard/seed_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _candidate_files(root: Path):
    """Find genuine RL training histories, excluding evaluation outputs."""
    rr = root / "results"

    if not rr.exists():
        return []

    out = []

    for p in rr.rglob("*.csv"):
        full = str(p).lower()
        name = p.name.lower()

        algorithm_match = any(
            k in full
            for k in ["q_learning", "qlearning", "q-learning", "dqn", "phase3"]
        )

        training_match = any(
            k in name
            for k in [
                "training_history",
                "training",
                "train_history",
                "convergence",
            ]
        )

        evaluation_like = any(
            k in name
            for k in [
                "evaluation",
                "evaluation_results",
                "eval_results",
            ]
        )

        if algorithm_match and training_match and not evaluation_like:
            out.append(p)

    return sorted(out)


def _choose_col(cols, candidates):
    lower = {c.lower(): c for c in cols}

    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]

    for col in cols:
        lc = col.lower()
        if any(candidate in lc for candidate in candidates):
            return col

    return None


def _algorithm_name(path: Path):
    s = str(path).lower()

    if "dqn" in s:
        return "DQN"

    if (
        "q_learning" in s
        or "qlearning" in s
        or "q-learning" in s
    ):
        return "Q-Learning"

    return "RL"


def discover_convergence_series(root: Path):
    """
    Create presentation-safe convergence curves.

    Evaluation files are excluded. Duplicate episode indices are averaged,
    episode order is sorted, and a rolling mean is displayed.
    """
    out = []
    algorithms_seen = set()

    for p in _candidate_files(root):
        try:
            raw = pd.read_csv(p)
        except Exception:
            continue

        if raw.empty or len(raw.columns) < 2:
            continue

        cols = list(raw.columns)

        x = _choose_col(
            cols,
            [
                "episode",
                "training_episode",
                "episode_index",
                "iteration",
                "epoch",
                "step",
            ],
        )

        y = _choose_col(
            cols,
            [
                "episode_return",
                "return",
                "reward",
                "mean_reward",
                "success_rate",
                "loss",
            ],
        )

        if not x or not y:
            continue

        df = raw[[x, y]].copy()

        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna()

        # A handful of evaluation rows is not a convergence curve.
        if len(df) < 20:
            continue

        # Multiple seeds/restarts may reuse episode indices.
        df = (
            df.groupby(x, as_index=False)[y]
            .mean()
            .sort_values(x)
            .reset_index(drop=True)
        )

        if len(df) < 20:
            continue

        if len(df) >= 2000:
            window = 100
        elif len(df) >= 500:
            window = 50
        elif len(df) >= 100:
            window = 20
        else:
            window = 10

        smooth = f"{y}_rolling_mean"

        df[smooth] = (
            df[y]
            .rolling(
                window=window,
                min_periods=max(3, window // 5),
            )
            .mean()
        )

        plot_df = df[[x, smooth]].dropna()

        algorithm = _algorithm_name(p)

        # One clear convergence curve per algorithm.
        if algorithm in algorithms_seen:
            continue

        algorithms_seen.add(algorithm)

        out.append(
            {
                "title": f"{algorithm} convergence",
                "frame": plot_df,
                "x": x,
                "y": smooth,
                "label": f"{window}-episode rolling mean",
                "caption": (
                    f"{window}-episode rolling mean from {p.name}. "
                    "Duplicate episode indices averaged; evaluation files excluded."
                ),
                "path": p,
            }
        )

    return out
