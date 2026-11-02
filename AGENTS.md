# AGENTS.md

Guidance for coding agents working in this repo.

## What this is

NBA game predictor. Given two teams, it predicts each player's points, both team totals, and win probability. It is trained on 1970-present player game logs. The UI is a Streamlit app that also shows today's slate, live scores, injuries, and grades predictions against actual box scores.

## Layout

- `app.py`: Streamlit UI (today's slate, single matchup, results/grading, model freshness, status banner).
- `src/config.py`: paths and season list. All data paths come from here.
- `src/ingest_player_logs.py`, `src/ingest_team_logs.py`, `src/ingest_bios.py`: pull from nba_api into `data/raw/` (parquet). Resumable.
- `src/build_dataset.py`: merges logs and bios into `data/processed/training_expansive.parquet`.
- `src/features_rolling.py`: rolling and career features into `training_model_ready.parquet`.
- `src/train.py`: LightGBM player-points model. Writes `models/player_pts_lgbm.pkl`, `feature_importance.csv`, `metrics.json`.
- `src/predict.py`: `predict_matchup(team_a, team_b, ...)`. Live rosters, injury adjustments, Monte Carlo win probability.
- `src/injuries.py`: ESPN league injury report (cached 6h), fuzzy name matching.
- `src/today.py`: today's games, ESPN live scores (cached 60s), slate prediction, saves slates to `data/processed/slates/`.
- `src/grade.py`: compares saved slates against actual box scores.
- `src/update_daily.py`: appends the last 7 days of logs. `src/daily_update.sh` runs the full chain (update, build, features, train) and writes `data/processed/retrain_status.json`.
- `src/progress.py`: pipeline stage status used by the UI.

## Commands

```bash
pip3 install --break-system-packages -r requirements.txt   # system Python is externally managed
python3 -m src.predict LAL BOS --home LAL
python3 -m src.today --date YYYY-MM-DD
python3 -m src.grade --date YYYY-MM-DD
bash src/daily_update.sh                                    # full refresh + retrain, logs to data/raw/daily_update.log
streamlit run app.py --server.headless true                 # http://localhost:8501
```

Quick validation: `python3 -c "import ast; ast.parse(open('app.py').read())"`. Streamlit only shows runtime errors when a page loads, so after changing `app.py` run the server and check the log for `Traceback`.

## Rules

- **No leakage in training.** Features must use only information available before tip-off. `train.py` has a `SAME_GAME` set and an assert that fails the run if a same-game stat leaks in. Do not remove entries from it. If you add a new box-score-derived column, add it to that set. A suspiciously low MAE (around 0.05) means leakage. A realistic MAE is around 4.5.
- **Rolling features are shifted.** Any new rolling feature must `shift(1)` before aggregating so the current game is excluded.
- **Keep the UI on the existing style.** It uses HTML via `st.markdown(..., unsafe_allow_html=True)`. Every call that renders `logo()`, `team_block()`, or `game_card()` must pass `unsafe_allow_html=True`, or the raw HTML shows on screen. Build each HTML block on one line, since indented lines are parsed as code blocks.
- **UI verification:** `streamlit.testing.v1.AppTest` runs the page headlessly. Click the buttons and check `at.exception` is empty before restarting the server.
- **Streamlit fragments:** the banner and progress panel use `@st.fragment(run_every=15)`. Keep them free of side effects.
- **Data is not in git.** `data/` is gitignored (about 700MB). Do not commit it. Tests and smoke checks should not rewrite the real `data/processed/training_*` files. Remove any synthetic test parquet you create.
- **`models/` is committed.** Only commit a new model after `train.py` finishes and `metrics.json` is updated.

## Gotchas

- `lightgbm` needs `libomp` on macOS. `train.py` falls back to `HistGradientBoostingRegressor` if LightGBM can't load.
- `predict.py` falls back to a rolling-average baseline when no model file exists. Check `used_model` in the output.
- Injury feed (ESPN) knows Out vs. Day-To-Day only. Questionable players are discounted 25%. `extra_out` is the manual scratch path.
- Team totals run a little hot, and home teams can be overrated. `BENCH_CORRECTION` and `REDIST_SHARE` in `predict.py` are the calibration knobs.
- A daily launchd job (`~/Library/LaunchAgents/com.nbapredictor.daily.plist`) runs `src/daily_update.sh` at 6am. It is skipped if the Mac is asleep.
- nba_api occasionally wedges. Bio fetching runs in chunks (`python3 -m src.ingest_bios --max 600`) for this reason.
- `.agents/skills/` holds the installed impeccable design skill. It is not part of the app.

## Style

- Python 3.14, no type-check or lint config in the repo. Match the surrounding code.
- Keep comments sparse and explain non-obvious behavior only.
- Prefer small, targeted edits. Verify numeric changes with a real run (`src.predict`, `src.today`, `src.grade`) rather than only checking syntax.
