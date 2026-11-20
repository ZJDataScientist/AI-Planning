from pathlib import Path
import json

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Experimental Setup · CSC 668",
    page_icon="🧪",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[2]

st.markdown("""
<style>
.stApp { background:#fbfaf7; color:#101114; }
.block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:5rem; }
h1,h2,h3 { letter-spacing:-0.035em; }
h1 { font-size:3.2rem !important; }
.note {
    border-left:4px solid #3157d5;
    background:#f1f4ff;
    border-radius:0 12px 12px 0;
    padding:.9rem 1rem;
    margin:.8rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.caption("CSC 668 · MODEL DEVELOPMENT · EXPERIMENTS")

st.title("Experimental setup & raw metrics")

st.markdown("""
This page shows how each planner was built and the results we collected during the experiments.

We record evidence at three levels:

- **Episode level** — what happened across one complete planning attempt.
- **Decision level** — every action selected during that episode.
- **Execution level** — training, runtime, caching, inference, and compute behavior.
""")


# ============================================================
# COMMON PROTOCOL
# ============================================================

st.header("1 · Common experimental protocol")

protocol = pd.DataFrame(
    [
        [
            "Environment",
            "SocialServiceTransitionEnv v0.2-stochastic",
            "Same environment for every planner",
        ],
        [
            "Uncertainty",
            "U0 → U3",
            "Increasing stochastic disruption",
        ],
        [
            "Provider scope",
            "Maryland medical day care",
            "Real provider identity/geography + simulated operations",
        ],
        [
            "Client data",
            "Synthetic",
            "No real patient data",
        ],
        [
            "Action validation",
            "Environment",
            "Planner cannot declare its own action valid",
        ],
        [
            "Reachability truth",
            "Environment / search",
            "Separates feasible and formally unreachable cases",
        ],
        [
            "LLM protocol",
            "One action per model call",
            "No A*, MCTS, or RL assistance during Phase 5A",
        ],
    ],
    columns=["Element", "Setup", "Purpose"],
)

st.dataframe(
    protocol,
    hide_index=True,
    width="stretch",
)


# ============================================================
# MODELING PHASE
# ============================================================

st.header("2 · Modeling phase")

models = pd.DataFrame(
    [
        [
            "A* / BFS",
            "Classical search",
            "No training",
            "Explicit state-space search",
            "Optimality / reachability baseline",
        ],
        [
            "MCTS",
            "Simulation search",
            "No training",
            "Repeated future rollouts",
            "Sampling-based planning",
        ],
        [
            "Q-Learning",
            "Tabular RL",
            "Training required",
            "Learn Q(s,a)",
            "Learned policy under uncertainty",
        ],
        [
            "DQN",
            "Neural RL",
            "Training required",
            "Neural Q(s,a)",
            "Generalization beyond table lookup",
        ],
        [
            "Qwen3:14b",
            "LLM planner",
            "No task-specific training",
            "Rendered state → structured action",
            "Language-model planning baseline",
        ],
    ],
    columns=[
        "Planner",
        "Method",
        "Learning phase",
        "Decision mechanism",
        "Research role",
    ],
)

st.dataframe(
    models,
    hide_index=True,
    width="stretch",
)


# ============================================================
# FROZEN HEADLINE RESULTS
# ============================================================

st.header("3 · Results so far")

results = pd.DataFrame(
    [
        [
            "A*",
            "100% reachable success",
            "U0 optimal cost = 7",
            "Optimality + dead-end proof",
        ],
        [
            "MCTS",
            "100% reachable success",
            "Reliable but computationally expensive",
            "No optimality proof",
        ],
        [
            "Q-Learning",
            "100% reachable success",
            "Robust but cost-inefficient",
            "Training required",
        ],
        [
            "DQN",
            "≈91% → 20%",
            "Provider-disruption stress test",
            "High-index provider OOD brittleness",
        ],
        [
            "Qwen3:14b",
            "733 / 733 reachable solved",
            "0 / 227 dead ends recognized",
            "76.354% strict primary correctness",
        ],
    ],
    columns=[
        "Planner",
        "Headline result",
        "Secondary result",
        "Main boundary / guarantee",
    ],
)

st.dataframe(
    results,
    hide_index=True,
    width="stretch",
)

st.markdown("""
<div class="note">
<b>Interpretation:</b> this is not one simple leaderboard.
A* contributes formal guarantees, MCTS contributes simulation-based search,
Q-Learning and DQN contribute learned policies, and Qwen contributes strong
feasible-workflow execution while exposing a clear weakness in formal
infeasibility recognition.
</div>
""", unsafe_allow_html=True)


# ============================================================
# QWEN RAW METRICS
# ============================================================

st.header("4 · Qwen3:14b · raw Phase 5A metrics")

st.caption(
    "Comparison between the cached run and the final no-cache H100 run."
)

QWEN_RAW = """=== PRIMARY BENCHMARK ===
total episodes: 1040
primary episodes: 960
repeated sensitivity episodes: 80

reachable primary episodes: 733
reachable success: 733 / 733
reachable success rate: 100.000%

proven unreachable primary episodes: 227
correct no_solution claims: 0 / 227
missed dead ends: 227 / 227

strict primary correctness:
733 / 960 = 76.354%

malformed responses: 0


=== DECISION-LEVEL AGREEMENT ===
cached decisions: 13275
nocache decisions: 13275
common decision positions: 13275
cached-only positions: 0
nocache-only positions: 0

executed action agreement: 90.524%
proposed action agreement: 90.524%
state hash agreement: 100.000%

episodes containing decision divergence: 145 / 1040


=== OUTCOME AGREEMENT ===
success                       100.000% (0 different)
planning_cost                  91.058% (93 different)
invalid_actions                99.712% (3 different)
malformed_responses           100.000% (0 different)
missed_dead_end               100.000% (0 different)
loop_detected                  94.904% (53 different)
successful_recoveries         100.000% (0 different)
recovery_rate                 100.000% (0 different)
final_provider_slot_index     100.000% (0 different)


=== CACHE EXECUTION CHECK ===
cached run decisions marked from_cache: 8882
nocache run decisions marked from_cache: 0


=== PER PROFILE ===

U0
executed action agreement: 100.000%
planning-cost agreement: 100.000%

U1
executed action agreement: 97.991%
planning-cost agreement: 99.231%
episodes with divergence: 7

U2
executed action agreement: 89.758%
planning-cost agreement: 90.385%
episodes with divergence: 43

U3
executed action agreement: 84.970%
planning-cost agreement: 74.615%
episodes with divergence: 95


=== DISTURBANCE RECOVERY ===
approval:
112 / 112 = 100.000%

housing:
93 / 214 = 43.458%

provider:
96 / 218 = 44.037%

provider recovery given alternate available:
96 / 144 = 66.667%


=== CACHE-SENSITIVITY FORENSICS ===
divergent decision positions: 1258
episodes containing divergence: 145

divergences occurring after state was already:
unreachable_no_valid_provider

1258 / 1258 = 100.000%

both candidate actions invalid: 1247
validity-changing divergences: 11

divergences between two valid planning actions:
0
"""

st.code(
    QWEN_RAW,
    language="text",
)

st.warning(
    "Important correction: the earlier 1040 / 1040 'exact trajectory match' "
    "calculation used episode_actions, which stores an action COUNT rather "
    "than the action sequence. Decision-level JSONL is therefore the "
    "authoritative trajectory evidence."
)

st.markdown("""
<div class="note">
<b>Main result:</b> removing caching changed individual actions,
particularly under U2 and U3, but all 1,258 action divergences occurred only
after the environment was already formally infeasible. There were
<b>zero divergences between two valid planning actions</b>.
</div>
""", unsafe_allow_html=True)


# ============================================================
# RAW ARTIFACT EXPLORER
# ============================================================

st.header("5 · Experiment outputs")

st.markdown("""
Here we can look at the saved outputs from each planner, including training results,
evaluation summaries, and benchmark logs.
""")


def discover_artifacts(root, tokens, maximum=10):

    evidence = [
        "summary",
        "metric",
        "result",
        "evaluation",
        "eval",
        "report",
        "freeze",
        "benchmark",
        "training",
        "convergence",
        "comparison",
        "stdout",
    ]

    allowed = {
        ".txt",
        ".log",
        ".out",
        ".json",
        ".csv",
        ".jsonl",
    }

    matches = []

    for folder in [
        root / "results",
        root / "hpc",
        root / "logs",
    ]:

        if not folder.exists():
            continue

        for path in folder.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in allowed:
                continue

            rel = str(
                path.relative_to(root)
            ).lower()

            if not any(
                token in rel
                for token in tokens
            ):
                continue

            if not any(
                word in rel
                for word in evidence
            ):
                continue

            # Avoid dumping huge decision-level JSONL files.
            if (
                path.suffix.lower() == ".jsonl"
                and path.stat().st_size > 2_000_000
            ):
                continue

            matches.append(path)

    return sorted(matches)[:maximum]


def preview(path, max_chars=12000):

    try:

        if path.suffix.lower() == ".json":

            obj = json.loads(
                path.read_text(
                    errors="replace"
                )
            )

            text = json.dumps(
                obj,
                indent=2,
                sort_keys=True,
            )

        elif path.suffix.lower() == ".csv":

            frame = pd.read_csv(path)

            text = (
                frame.head(40)
                .to_string(index=False)
            )

            if len(frame) > 40:

                text += (
                    f"\n\n... "
                    f"{len(frame)-40:,} "
                    "additional rows ..."
                )

        else:

            text = path.read_text(
                errors="replace"
            )

    except Exception as exc:

        return f"Could not preview artifact: {exc}"

    if len(text) > max_chars:

        text = (
            text[:max_chars]
            + "\n\n... preview truncated ..."
        )

    return text


planner_tokens = {
    "A* / BFS": [
        "astar",
        "a_star",
        "bfs",
        "phase2a",
    ],
    "MCTS": [
        "mcts",
        "phase2b",
    ],
    "Q-Learning": [
        "q_learning",
        "qlearning",
        "phase3a",
    ],
    "DQN": [
        "dqn",
        "phase3b",
    ],
    "Qwen3:14b": [
        "qwen3_14b",
        "qwen3",
        "phase5a",
    ],
}


tabs = st.tabs(
    list(planner_tokens.keys())
)

for tab, (
    planner,
    tokens,
) in zip(
    tabs,
    planner_tokens.items(),
):

    with tab:

        files = discover_artifacts(
            ROOT,
            tokens,
        )

        if not files:

            st.info(
                "No matching raw artifact was automatically found. "
                "The dashboard intentionally does not invent terminal output."
            )

            continue

        choices = [
            str(
                path.relative_to(ROOT)
            )
            for path in files
        ]

        selected = st.selectbox(
            f"{planner} artifact",
            choices,
            key=f"raw-{planner}",
        )

        selected_path = (
            ROOT / selected
        )

        st.caption(
            f"File: `{selected}`"
        )

        st.code(
            preview(
                selected_path
            ),
            language="text",
        )


# ============================================================
# METRIC DEFINITIONS
# ============================================================

st.header("6 · Metric definitions")

definitions = pd.DataFrame(
    [
        [
            "Scenario",
            "One synthetic transition configuration and uncertainty condition.",
        ],
        [
            "Episode",
            "One complete planner attempt.",
        ],
        [
            "Decision",
            "One planner action and corresponding environment step.",
        ],
        [
            "Success",
            "Successful terminal transition state.",
        ],
        [
            "Planning cost",
            "Accumulated environment cost; lower is more efficient.",
        ],
        [
            "Invalid action",
            "Action rejected because modeled constraints were violated.",
        ],
        [
            "Reachable",
            "At least one valid solution still exists.",
        ],
        [
            "Proven unreachable",
            "No valid solution remains under the modeled state.",
        ],
        [
            "Dead-end recognition",
            "Planner correctly declares no solution in an unreachable state.",
        ],
        [
            "Agreement",
            "Same measured outcome between repeated versions of one experiment.",
        ],
    ],
    columns=[
        "Metric",
        "Meaning",
    ],
)

st.dataframe(
    definitions,
    hide_index=True,
    width="stretch",
)

st.caption(
    "Experimental evidence page · v1"
)
