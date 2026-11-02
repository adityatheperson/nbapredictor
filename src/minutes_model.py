"""Minutes model: predicts a player's minutes from pre-game features.
Expected points then scale by predicted/typical minutes, so DNPs, rest nights and
rotation changes move the scoring prediction instead of being ignored.
Shares train.py's leakage guard (same-game stats excluded; MIN_NUM raw is the target).
- Train: python3 -m src.minutes_model   (saves models/player_min_lgbm.pkl)
"""
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from src.train import EXCLUDE as TRAIN_EXCLUDE  # reuse: PTS raw + same-game stay out

READY = Path("data/processed/training_model_ready.parquet")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)
TARGET = "MIN_NUM"


def main():
    if not READY.exists():
        print(f"No {READY} yet.")
        return
    df = pd.read_parquet(READY)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").dropna(subset=[TARGET])
    feats = [c for c in df.columns if c not in TRAIN_EXCLUDE
             and c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
    assert not any(c in feats for c in ["PTS", "MIN", "FGM", "FGA", "DID_NOT_PLAY"]), "leakage!"
    print(f"Features ({len(feats)})")
    X, y = df[feats].fillna(-1), df[TARGET]
    dates = df["GAME_DATE"]
    cutoff = dates.quantile(0.85)
    tr, te = dates <= cutoff, dates > cutoff
    print(f"Train {tr.sum()} / Test {te.sum()} (cutoff {cutoff.date()})")
    if HAS_LGBM:
        reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1)
        reg.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                            min_samples_leaf=100, random_state=42)
        reg.fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    mae = mean_absolute_error(y[te], pred)
    print(f"Player MIN MAE: {mae:.2f}")
    imp = getattr(reg, "feature_importances_", None)
    if imp is not None:
        pd.DataFrame({"feature": feats, "importance": imp}).sort_values(
            "importance", ascending=False).head(20).to_csv(MODELS / "minutes_importance.csv", index=False)
    pickle.dump({"model": reg, "features": feats}, open(MODELS / "player_min_lgbm.pkl", "wb"))
    import json
    metrics = json.loads(Path("models/metrics.json").read_text())
    metrics["player_min_mae"] = round(float(mae), 3)
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/player_min_lgbm.pkl")


if __name__ == "__main__":
    main()
