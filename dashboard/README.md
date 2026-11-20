# CSC 668 Research Dashboard

Copy `dashboard/` and `.streamlit/` into:

`/Users/zhanderjunco/Documents/CSC-668 AI/`

Then:

```bash
cd "/Users/zhanderjunco/Documents/CSC-668 AI"
python3 -m venv .venv-dashboard
source .venv-dashboard/bin/activate
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The dashboard reads the frozen Qwen Phase 5A summaries directly from `results/`.
It also scans for recognizable Phase 3 Q-learning/DQN training CSVs and will render
convergence curves when it finds safe episode/reward/loss columns.

`seed_metrics.csv` is a transparent temporary bridge for the cross-planner headline chart.
As we identify the exact Phase 2–4 summary artifacts, we can replace those rows with
fully artifact-derived metrics.
