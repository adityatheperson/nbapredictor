# NBA Predictor

Predict any NBA matchup: per-player points, both team totals, and win probability — trained on 1.3M player games going back to 1970, with rolling form, rest, home court, age, and draft pedigree. Player scoring MAE **4.52** on 196k held-out games.

## Quickstart (predict only)

```bash
pip install -r requirements.txt
python -m src.predict LAL BOS --home LAL   # CLI
streamlit run app.py                        # UI -> http://localhost:8501
```

A trained model (`models/player_pts_lgbm.pkl`) is included, so predictions work immediately. Without it, the predictor falls back to a rolling-average baseline.

## Retrain / update

```bash
python -m src.ingest_player_logs   # every game since 1970-71, Regular + Playoffs
python -m src.ingest_team_logs     # team-level logs (context)
python -m src.ingest_bios          # age/height/draft/school (~5,196 players)
python -m src.build_dataset        # merge + derived efficiency + rest/context
python -m src.features_rolling     # 3/5/10/20-game avgs, career avgs, form
python -m src.train                # walk-forward LightGBM (leakage-guarded)
python -m src.update_daily         # append last 7 days (run daily to stay current)
python -m src.progress             # pipeline status bars
```

`data/` is git-ignored (pulled locally, ~700MB). Everything else needed to run is committed.

## How it works

1. **Player PTS regressor** (LightGBM, 194 pre-game features): rolling scoring/minutes/usage trends, career averages, rest days, back-to-backs, home court, age + age², draft pedigree. Same-game box-score columns are hard-excluded with an assert — the model can't cheat by learning `PTS = 2*FGM + FG3M + FTM`.
2. **Team totals**: sum of top-10 rotation scorers scaled by a bench correction factor.
3. **Win probability**: 10k Monte Carlo simulations over per-player uncertainty + analytic Normal estimate.

## Repo layout

```
src/
  config.py            seasons/paths
  ingest_player_logs.py  ingest_team_logs.py  ingest_bios.py
  build_dataset.py     features_rolling.py
  train.py             predict.py           # predict_matchup(team_a, team_b)
  update_daily.py      progress.py
app.py                 Streamlit test UI
models/                trained model + metrics + feature importance
```
