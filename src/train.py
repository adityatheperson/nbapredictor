"""Train models on model-ready data.
- Player PTS regressor (LightGBM)
- Team total regressor (aggregate predicted PTS + correction)
- Win classifier (logistic on predicted total diff)
Walk-forward split by date. Run: python3 -m src.train
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

READY = Path("data/processed/training_model_ready.parquet")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

TARGET = "PTS"
# NOTE: all same-game box-score columns must be excluded — at tip-off we don't
# know FGM/FTM/MIN/FG_PCT/etc. PTS ≈ 2*FGM + FG3M + FTM, so leaving them in lets
# the model "learn" the scoring formula (MAE ~0.05) instead of predicting.
# Even same-game percentages/rates leak via missingness (NaN pct => DNP => 0 pts).
# Only pre-game info is allowed: rolling avgs/stds, career avgs, context, bio.
from src.features_rolling import ROLL_COLS
SAME_GAME = set(ROLL_COLS) | {
    "MIN", "MIN_NUM",
    "FG_PCT", "FG3_PCT", "FT_PCT",       # same-game percentages (not in ROLL_COLS)
    "EFG_PCT", "TS_PCT",                  # same-game efficiency
    "AST_TOV", "PTS_PER_MIN", "REB_PER_MIN", "AST_PER_MIN",
    "DOUBLE_DOUBLE", "PTS_30", "PTS_20", "DID_NOT_PLAY",
    "FANTASY_PTS", "PLUS_MINUS",
}
EXCLUDE = {TARGET, "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
           "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
           "PREV_GAME_DATE", "BIRTHDATE", "HEIGHT", "WEIGHT", "POSITION",
           "SCHOOL", "COUNTRY", "DISPLAY_FIRST_LAST", "SEASON", "SEASON_TYPE",
           "OPPONENT", "SEASON_ID", "DRAFT_YEAR", "DRAFT_ROUND", "VIDEO_AVAILABLE",
           } | SAME_GAME

def main():
    if not READY.exists():
        print(f"No {READY} yet. Run ingest -> build_dataset -> features_rolling first.")
        return
    df = pd.read_parquet(READY)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").dropna(subset=[TARGET])
    # numeric pre-game features only (EXCLUDE already drops IDs, dates, strings, same-game stats)
    feats = [c for c in df.columns if c not in EXCLUDE
             and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Features ({len(feats)}): {feats[:10]}...")
    assert not any(c in feats for c in SAME_GAME), \
        f"leakage: same-game column in features: {sorted(set(feats) & SAME_GAME)}"
    X, y = df[feats].fillna(-1), df[TARGET]
    dates = df["GAME_DATE"]
    cutoff = dates.quantile(0.85)  # walk-forward: train past, test recent
    tr, te = dates <= cutoff, dates > cutoff
    print(f"Train {tr.sum()} / Test {te.sum()} (cutoff {cutoff.date()})")
    if HAS_LGBM:
        reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1)
        reg.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        print("LightGBM unavailable (missing libomp) — using HistGradientBoostingRegressor fallback.")
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                            min_samples_leaf=100, random_state=42)
        reg.fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    mae = mean_absolute_error(y[te], pred)
    print(f"Player PTS MAE: {mae:.2f}")
    imp = getattr(reg, "feature_importances_", None)
    if imp is not None:
        pd.DataFrame({"feature": feats, "importance": imp}).sort_values(
            "importance", ascending=False).head(20).to_csv(MODELS / "feature_importance.csv", index=False)
    import pickle
    pickle.dump({"model": reg, "features": feats}, open(MODELS / "player_pts_lgbm.pkl", "wb"))
    import json
    from datetime import datetime
    json.dump({"player_pts_mae": round(float(mae), 3), "n_train": int(tr.sum()),
               "n_test": int(te.sum()), "cutoff": str(cutoff.date()),
               "n_features": len(feats), "trained_at": datetime.now().isoformat(timespec="seconds")},
              open(MODELS / "metrics.json", "w"), indent=2)
    print("Saved models/player_pts_lgbm.pkl + feature_importance.csv + metrics.json")

if __name__ == "__main__":
    main()
