from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from data import load_project_data, load_algorithm_seed_metrics, discover_convergence_series

st.set_page_config(
    page_title="CSC 668 · Planning Under Uncertainty",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]

DATA = load_project_data(ROOT)

ALGORITHM_COLORS = {
    "A*": "#3157D5",
    "A* / BFS": "#3157D5",
    "MCTS": "#7C3AED",
    "Q-Learning": "#059669",
    "DQN": "#DC2626",
    "Qwen3:14b": "#D97706",
}

RUN_COLORS = {
    "Cached control": "#6B7280",
    "No-cache primary": "#3157D5",
}

RL_COLORS = {
    "Q-Learning": "#059669",
    "DQN": "#DC2626",
}

UNCERTAINTY_COLORS = {
    "U1": "#60A5FA",
    "U2": "#F59E0B",
    "U3": "#EF4444",
}

st.markdown("""
<style>
:root {
  --ink:#101114; --muted:#676b73; --paper:#fbfaf7; --card:#ffffff;
  --line:#e8e5df; --accent:#3157d5; --soft:#f1f4ff;
}
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:5rem; }
h1,h2,h3 { letter-spacing:-0.035em; }
h1 { font-size:3.5rem !important; line-height:1.02 !important; }
.hero-kicker { color:var(--accent); font-size:.82rem; font-weight:800;
  letter-spacing:.11em; text-transform:uppercase; margin-bottom:.7rem; }
.hero-sub { color:var(--muted); font-size:1.28rem; line-height:1.55;
  max-width:900px; margin-bottom:1.4rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:1.25rem 1.35rem; min-height:100%; box-shadow:0 1px 2px rgba(0,0,0,.025); }
.metric-big { font-size:2.15rem; font-weight:800; letter-spacing:-.04em; }
.metric-label { color:var(--muted); font-size:.93rem; margin-top:.15rem; }
.note { border-left:4px solid var(--accent); background:var(--soft);
  border-radius:0 12px 12px 0; padding:.9rem 1rem; margin:.8rem 0 1rem; }
.muted { color:var(--muted); }
.pill { display:inline-block; padding:.27rem .58rem; margin:.15rem .2rem .15rem 0;
  border:1px solid var(--line); border-radius:999px; background:#fff; font-size:.82rem; }
.small { font-size:.88rem; color:var(--muted); }
.section-rule { border-top:1px solid var(--line); margin:2.7rem 0 1.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-kicker">CSC 668 · AI Planning Research</div>', unsafe_allow_html=True)
st.title("From structured search to social-service workflows")

st.markdown(
    "<div style='font-size:1.05rem; color:#676b73; margin-top:-0.4rem; margin-bottom:1.4rem;'>"
    "Zhander Junco · Research Project"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("""
<div class="hero-sub">
We ask one practical question: <b>when a complex transition workflow becomes uncertain,
which planning methods keep working — and how do they fail?</b><br><br>
The simulator models a Maryland brain-injury waiver transition from institutional care
toward community living. Real provider identities and geography are combined with
synthetic clients and simulated operational constraints, so the work stays reproducible
without using real patient data.
</div>
""", unsafe_allow_html=True)

for col, (value, label) in zip(st.columns(4), [
    ("5", "planner families studied"),
    ("U0 → U3", "increasing uncertainty"),
    ("1,040", "Qwen evaluation episodes"),
    ("13,275", "Qwen decision records"),
]):
    with col:
        st.markdown(f'<div class="card"><div class="metric-big">{value}</div>'
                    f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.header("1 · What the planner is trying to do")
left, right = st.columns([1.15, 1])
with left:
    st.markdown("""
The planner sees a structured transition state and chooses **one action at a time**.
The environment independently checks prerequisites, applies costs, introduces
disruptions, and returns the next state.

We use the system to compare how different planning methods move through the same workflow.
Medical eligibility, diagnosis, and final provider assignment are not part of the experiment.
""")
    st.markdown("""
<div class="note"><b>Scope:</b> Every planner works on the same transition workflow.
The experiment focuses on planning decisions, costs, disruptions, and recovery.</div>
""", unsafe_allow_html=True)
with right:
    st.graphviz_chart("""
digraph {
  rankdir=TB;
  graph [bgcolor="transparent", pad="0.1"];
  node [shape=box, style="rounded,filled", fillcolor="white", color="#d9d7d2",
        fontname="Helvetica", fontsize=11, margin="0.15,0.10"];
  edge [color="#777777", arrowsize=0.7];
  S [label="Current transition state"];
  P [label="Planner chooses one action"];
  V [label="Environment validates it"];
  E [label="Cost / reward + stochastic event"];
  N [label="Next state"];
  S -> P -> V -> E -> N -> P;
}
""")

st.header("2 · How the experiment works")
for col, (title, body) in zip(st.columns(3), [
    ("Scenario", "One synthetic transition situation: client constraints, provider landscape, housing conditions, and disturbance seeds."),
    ("Episode", "One complete attempt to navigate a scenario until success, truncation, or an unreachable condition."),
    ("Decision", "One planner choice inside an episode. Qwen alone produced 13,275 decision-level records in the frozen Phase 5A run."),
]):
    with col:
        st.markdown(f'<div class="card"><h3>{title}</h3><div class="muted">{body}</div></div>', unsafe_allow_html=True)

st.markdown("#### Uncertainty profiles")
for col, (code, title, body) in zip(st.columns(4), [
    ("U0", "Deterministic baseline", "No stochastic disruption."),
    ("U1", "Light uncertainty", "Limited operational disturbance."),
    ("U2", "Moderate uncertainty", "More frequent recovery decisions."),
    ("U3", "High uncertainty", "Most difficult disruption profile."),
]):
    with col:
        st.markdown(f'<div class="card"><div class="metric-big">{code}</div>'
                    f'<b>{title}</b><div class="small">{body}</div></div>', unsafe_allow_html=True)

st.markdown("#### Action space")
st.markdown("""
<span class="pill">develop plan of care</span>
<span class="pill">secure housing</span>
<span class="pill">select provider slot</span>
<span class="pill">request / wait for approval</span>
<span class="pill">recover after disruption</span>
<span class="pill">complete transition</span>
<span class="pill">declare no solution</span>
""", unsafe_allow_html=True)
st.caption("The implementation uses a fixed action catalog. The environment — not the model — decides whether a proposed action is valid.")

st.markdown("#### Reward / cost logic")
for col, (title, body) in zip(st.columns(3), [
    ("Progress is cheap", "Valid forward progress should minimize planning cost."),
    ("Mistakes are expensive", "Invalid or unnecessary actions add cost and expose failure modes."),
    ("Success is not enough", "We also track optimality, recovery, loops, dead-end recognition, and compute."),
]):
    with col:
        st.markdown(f'<div class="card"><h3>{title}</h3><div class="muted">{body}</div></div>', unsafe_allow_html=True)

st.header("3 · What we compared")
planner_df = pd.DataFrame([
    ["BFS / A*", "Classical search", "Explicit search over the workflow graph", "No training"],
    ["MCTS", "Simulation search", "Samples future trajectories", "No training"],
    ["Q-Learning", "Tabular RL", "Learns state-action values", "Training required"],
    ["DQN", "Neural RL", "Approximates action values with a neural network", "Training required"],
    ["Qwen3:14b", "LLM planner", "Reads the rendered state and proposes one action", "Inference only"],
], columns=["Planner", "Family", "Core idea", "Compute mode"])
st.dataframe(planner_df, hide_index=True, width="stretch")

st.header("4 · Headline quantitative results")

seed = load_algorithm_seed_metrics(ROOT)

headline_colors = {
    "A*": "#3157D5",
    "MCTS": "#7C3AED",
    "Q-Learning": "#059669",
    "DQN": "#DC2626",
    "Qwen3:14b": "#D97706",
}

if not seed.empty:

    st.markdown("#### Planner performance at a glance")

    fig = px.bar(
        seed.dropna(subset=["headline_success_pct"]),
        x="algorithm",
        y="headline_success_pct",
        text="headline_success_pct",
        color="algorithm",
        color_discrete_map=headline_colors,
        labels={
            "algorithm": "",
            "headline_success_pct": "Success (%)",
        },
        title="Headline success / robustness metric",
    )

    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="outside",
    )

    fig.update_layout(
        yaxis_range=[0, 108],
        height=390,
        showlegend=False,
        bargap=0.20,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.markdown("""
<div class="note">
<b>What the numbers show:</b>
A*, MCTS, and Q-Learning remained successful on reachable cases in the
completed robustness experiments. DQN showed the clearest weakness under
provider disruption, where its success dropped to about <b>20%</b>.
Qwen solved <b>100% of reachable cases</b>, although its dead-end behavior
is examined separately later in the dashboard.
</div>
""", unsafe_allow_html=True)


if DATA.qwen_summary:

    qprof = DATA.qwen_profile_frame()

    if not qprof.empty:

        st.markdown(
            "#### Qwen3:14b under increasing uncertainty"
        )

        profile_colors = {
            "U0": "#3157D5",
            "U1": "#60A5FA",
            "U2": "#F59E0B",
            "U3": "#EF4444",
        }

        fig = px.bar(
            qprof,
            x="profile",
            y="reachable_success_pct",
            text="reachable_success_pct",
            color="profile",
            color_discrete_map=profile_colors,
            labels={
                "profile": "Uncertainty profile",
                "reachable_success_pct": "Reachable success (%)",
            },
            title="Reachable-task success remains 100%",
        )

        fig.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="outside",
        )

        fig.update_layout(
            yaxis_range=[0, 108],
            height=350,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        fig2 = px.line(
            qprof,
            x="profile",
            y="planning_cost_mean",
            markers=True,
            labels={
                "profile": "Uncertainty profile",
                "planning_cost_mean": "Mean planning cost",
            },
            title="Planning cost rises with uncertainty",
        )

        fig2.update_traces(
            line=dict(
                color="#D97706",
                width=3,
            ),
            marker=dict(
                size=9,
            ),
        )

        fig2.update_layout(
            height=350,
        )

        st.plotly_chart(
            fig2,
            width="stretch",
        )

        st.markdown("""
<div class="note">
<b>What the numbers show:</b>
Qwen maintained <b>100% reachable-case success</b> across U0–U3,
but mean planning cost increased from <b>7.0 in U0</b> to about
<b>37.3 in U3</b>. The median stayed near 7–8 actions, so most of
the increase comes from a smaller group of difficult episodes.
</div>
""", unsafe_allow_html=True)

st.header("5 · Computational cost and efficiency")

st.markdown("""
Different planners pay for computation in different places.

- **Classical search** spends compute while planning.
- **MCTS** spends more compute on repeated simulations.
- **Reinforcement learning** pays most of its cost during training, then acts cheaply.
- **LLMs** require no task-specific training here, but pay for neural inference at every decision.

This makes runtime part of the comparison—not just success rate.
""")

compute_profile = pd.DataFrame(
    [
        [
            "A* / BFS",
            "None",
            "Search at planning time",
            "Low–moderate",
            "Exact search; A* can prove optimality and dead ends.",
        ],
        [
            "MCTS",
            "None",
            "Repeated rollouts at planning time",
            "Moderate–high",
            "More computationally expensive because each choice requires simulation.",
        ],
        [
            "Q-Learning",
            "Required",
            "Table lookup after training",
            "Very low",
            "Moves most computational cost into the training phase.",
        ],
        [
            "DQN",
            "Required",
            "Neural-network forward pass",
            "Low",
            "Training is expensive, but deployment decisions are comparatively cheap.",
        ],
        [
            "Qwen3:14b",
            "No task-specific training",
            "LLM inference for each decision",
            "High",
            "Each fresh action requires a 14B-parameter model inference.",
        ],
    ],
    columns=[
        "Planner",
        "Training cost",
        "Where compute is spent",
        "Decision-time burden",
        "Interpretation",
    ],
)

st.dataframe(
    compute_profile,
    hide_index=True,
    width="stretch",
)

st.markdown("#### Measured Qwen3:14b compute cost")

cached_seconds = 3318.7429
nocache_seconds = 10683.4128

logical_decisions = 13275
cache_hits = 8882

cached_minutes = cached_seconds / 60.0
nocache_minutes = nocache_seconds / 60.0

# Prefer artifact-derived execution times when available.
if DATA.qwen_compute_frame is not None and not DATA.qwen_compute_frame.empty:
    measured = (
        DATA.qwen_compute_frame
        .set_index("run")["wall_clock_minutes"]
        .to_dict()
    )

    cached_minutes = float(
        measured.get("Cached", cached_minutes)
    )

    nocache_minutes = float(
        measured.get("No cache", nocache_minutes)
    )

cache_hit_rate = cache_hits / logical_decisions
runtime_ratio = nocache_minutes / cached_minutes

nocache_seconds_per_decision = (
    nocache_minutes * 60.0
) / logical_decisions

qwen_compute = pd.DataFrame(
    {
        "run": [
            "Cached control",
            "No-cache primary",
        ],
        "wall_clock_minutes": [
            cached_minutes,
            nocache_minutes,
        ],
    }
)

fig = px.bar(
    qwen_compute,
    x="run",
    y="wall_clock_minutes",
    text="wall_clock_minutes",
    color="run",
    color_discrete_map=RUN_COLORS,
    labels={
        "run": "",
        "wall_clock_minutes": "Wall-clock time (minutes)",
    },
    title="Qwen3:14b H100 benchmark runtime",
)

fig.update_traces(
    texttemplate="%{text:.1f} min",
    textposition="outside",
)

fig.update_layout(
    height=380,
    showlegend=False,
)

st.plotly_chart(
    fig,
    width="stretch",
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Fresh-inference runtime",
    f"{nocache_minutes:.1f} min",
)

c2.metric(
    "LLM decisions",
    f"{logical_decisions:,}",
)

c3.metric(
    "Mean wall time / decision",
    f"{nocache_seconds_per_decision:.2f} s",
)

c4.metric(
    "Cache speed-up",
    f"{runtime_ratio:.2f}×",
)

st.markdown(
    f"""
<div class="note">
<b>What the numbers show:</b>
the no-cache benchmark required
<b>{nocache_minutes:.1f} minutes</b> on one NVIDIA H100 for
<b>{logical_decisions:,} decisions</b>, or about
<b>{nocache_seconds_per_decision:.2f} seconds per logical decision</b>
including benchmark overhead.

The cached control finished in <b>{cached_minutes:.1f} minutes</b>
because <b>{cache_hits:,} decisions ({cache_hit_rate:.1%})</b>
reused an earlier response instead of invoking Qwen again.

Caching reduced wall-clock runtime by roughly
<b>{runtime_ratio:.2f}×</b>, but the no-cache experiment is the version used for the final comparison.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("#### What the compute comparison tells us")

st.markdown("""
The algorithms do not simply differ in accuracy—they have different
**computational strategies**.

**A\\*** requires no training and spends its computation during search.
In return, it provides strong guarantees such as optimality and explicit
dead-end detection.

**MCTS** also avoids training, but repeatedly simulates possible futures,
making its decision-time cost higher.

**Q-Learning and DQN** move most of their computational cost into training.
Once trained, action selection is comparatively inexpensive, although a
learned policy can become brittle outside its training distribution.

**Qwen3:14b** requires no environment-specific training in this experiment,
but each fresh planning decision requires large-model inference.

The comparison therefore asks not only **which planner succeeds**, but
**when computation is spent and what guarantees that computation buys**.
""")

st.caption("Dashboard build: v4 · final computational-cost section")

st.header("6 · Learning curves / convergence")
conv = discover_convergence_series(ROOT)

if conv:
    cols = st.columns(min(2, len(conv)))

    for idx, series in enumerate(conv[:4]):
        with cols[idx % len(cols)]:
            fig = px.line(
                series["frame"],
                x=series["x"],
                y=series["y"],
                title=series["title"],
                labels={
                    series["x"]: "Training episode",
                    series["y"]: series["label"],
                },
            )
            if "Q-Learning" in series["title"]:
                curve_color = RL_COLORS["Q-Learning"]
            elif "DQN" in series["title"]:
                curve_color = RL_COLORS["DQN"]
            else:
                curve_color = "#3157D5"

            fig.update_traces(
                line=dict(
                    color=curve_color,
                    width=3,
                )
            )

            fig.update_layout(height=340)

            st.plotly_chart(
                fig,
                width="stretch",
            )
            st.caption(series["caption"])

    st.markdown("""
<div class="note"><b>What the numbers show:</b> These curves use genuine training histories,
not evaluation outputs. Episode indices are sorted, duplicate training episodes are averaged,
and a rolling mean is shown to expose the learning trend without connecting unrelated
restarts or individual noisy episodes.</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div class="card"><h3>Training curves not located yet</h3>
<div class="muted">Only genuine Q-learning and DQN training-history files are eligible here.
Evaluation outputs are deliberately excluded rather than being misrepresented as convergence.</div>
</div>
""", unsafe_allow_html=True)

st.header("7 · The most interesting result so far")
for col, (value, label) in zip(st.columns(4), [
    ("733 / 733", "reachable primary cases solved"),
    ("227", "proven unreachable primary cases"),
    ("0 / 227", "dead ends correctly declared"),
    ("0", "malformed responses"),
]):
    with col:
        st.markdown(f'<div class="card"><div class="metric-big">{value}</div>'
                    f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("""
### Qwen is stable while a solution exists — and gets stuck when none does.

The no-cache sensitivity run changed **1,258 individual decisions across 145 episodes**.
But every divergence occurred after the environment had already reached
`unreachable_no_valid_provider`.

- **1,247 / 1,258** divergences were between two invalid actions.
- The remaining **11** switched between a valid `wait` and an invalid county-mismatched provider.
- **0** divergences were between two different valid planning actions.
- Reachable success and disturbance-recovery conclusions did not change.

<div class="note"><b>What this suggests:</b> the LLM is very good at executing a feasible workflow.
Its sharp failure boundary is formal infeasibility recognition — exactly the capability
classical search can provide.</div>
""", unsafe_allow_html=True)

div_df = pd.DataFrame({
    "profile":["U1","U2","U3"],
    "divergent_decisions":[45,368,845],
})
fig = px.bar(
    div_df,
    x="profile",
    y="divergent_decisions",
    text="divergent_decisions",
    color="profile",
    color_discrete_map=UNCERTAINTY_COLORS,
    labels={
        "profile": "Uncertainty profile",
        "divergent_decisions": "Divergent decisions",
    },
    title="Action-level stochasticity appears only after infeasibility",
)
fig.update_layout(height=360)
st.plotly_chart(fig, width="stretch")

st.header("8 · What we have learned")
for col, (title, body) in zip(st.columns(3), [
    ("Classical search proves", "A* gives us optimality and explicit dead-end reasoning."),
    ("RL adapts", "Q-learning and DQN show how learned policies respond to repeated uncertainty."),
    ("LLMs execute", "Qwen follows feasible workflows extremely well, but does not reliably know when no solution remains."),
]):
    with col:
        st.markdown(f'<div class="card"><h3>{title}</h3><div class="muted">{body}</div></div>', unsafe_allow_html=True)

st.markdown("""
### Next experiment: combine the strengths.

**LLM + A-star** is no longer just an architectural idea.

> Let the LLM handle natural workflow execution, while symbolic search verifies feasibility,
> detects dead ends, and prevents the model from wandering after the problem becomes impossible.
""")

with st.expander("Data provenance shown by this dashboard"):
    st.write(f"Project root: `{ROOT}`")
    st.write(f"Qwen primary summary: `{DATA.qwen_summary_path or 'not found'}`")
    st.write(f"Qwen cached summary: `{DATA.qwen_cached_summary_path or 'not found'}`")
    st.write("Real: provider identities/geography/licensure inputs where available.")
    st.write("Synthetic/simulated: client records, operational capacity/availability, disturbances.")
    st.write("No real patient data; no SSN/name/DOB/Medicaid ID is modeled.")
