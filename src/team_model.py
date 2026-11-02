"""Team-total model: predicts a team's score from offense, opponent defense, pace,
rest and home court — the context the player-sum totals lack.
- Train:  python3 -m src.team_model            (walk-forward, saves models/team_total_lgbm.pkl)
- Blend happens in predict.py (50/50 with the injury-adjusted player sum).
"""
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from src.config import APP_DIR

TEAM_LOGS = Path("data/raw/team_game_logs_all.parquet")
MODEL_FILE = Path("models/team_total_lgbm.pkl")
W = 10


def parse_opp(matchup, mine):
    m = str(matchup)
    at = re.match(r"([A-Z]{2,3})\s*@\s*([A-Z]{2,3})", m)
    if at:
        return at.group(2) if at.group(1) == mine else at.group(1)
    vs = re.match(r"([A-Z]{2,3})\s*vs\.\s*([A-Z]{2,3})", m)
    if vs:
        return vs.group(2) if vs.group(1) == mine else vs.group(1)
    return None


def team_games_df():
    """One row per team-game from full logs or the committed snapshot."""
    if TEAM_LOGS.exists():
        df = pd.read_parquet(TEAM_LOGS)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    recent = pd.read_parquet(APP_DIR / "player_recent.parquet")
    recent["GAME_DATE"] = pd.to_datetime(recent["GAME_DATE"])
    g = recent.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False).agg(
        GAME_DATE=("GAME_DATE", "min"), PTS=("PTS", "sum"), FGA=("FGA", "sum"),
        FTA=("FTA", "sum"), OREB=("OREB", "sum"), TOV=("TOV", "sum"))
    g["MATCHUP"] = None
    g["SEASON_TYPE"] = "Regular Season"
    opp = g[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "OPP", "PTS": "OPP_PTS"})
    g = g.merge(opp, on="GAME_ID")
    g = g[g["OPP"] != g["TEAM_ABBREVIATION"]]
    g["ALLOWED"] = g["OPP_PTS"]
    return g


def build_frame(df=None):
    df = team_games_df() if df is None else df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["PACE"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] - df["TOV"]
    # opponent score + abbrev via self-join on game
    opp = df[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "OPP", "PTS": "OPP_PTS"})
    df = df.merge(opp, on="GAME_ID")
    df = df[df["OPP"] != df["TEAM_ABBREVIATION"]]
    df["ALLOWED"] = df["OPP_PTS"]
    df["IS_HOME"] = df["MATCHUP"].fillna("").str.contains("vs\\.").astype(int) if "MATCHUP" in df else 0
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    grp = df.groupby("TEAM_ABBREVIATION")
    df["PREV_DATE"] = grp["GAME_DATE"].shift(1)
    df["REST_DAYS"] = ((df["GAME_DATE"] - df["PREV_DATE"]).dt.days - 1).fillna(7).clip(0, 14)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 0).astype(int)
    for c in ["PTS", "PACE", "ALLOWED"]:
        df[f"{c}_AVG{W}"] = grp[c].shift(1).rolling(W, min_periods=1).mean().values
    df["IS_PLAYOFF"] = (df["SEASON_TYPE"] == "Playoffs").astype(int) if "SEASON_TYPE" in df else 0
    df["GAME_MONTH"] = df["GAME_DATE"].dt.month
    # opponent strength as of game date (strictly prior games only).
    # NOTE: the main frame uses shift(1) rolling avgs (exclude current game). For the
    # opponent lookup that would go stale by one game under a strict asof merge, so
    # the series carries UNSHIFTED trailing avgs and the merge stays strictly exclusive.
    df = df.reset_index(drop=True)
    _s = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    _g = _s.groupby("TEAM_ABBREVIATION")
    series = _s[["TEAM_ABBREVIATION", "GAME_DATE"]].copy()
    series[f"ALLOWED_AVG{W}"] = _g["ALLOWED"].rolling(W, min_periods=1).mean().values
    series[f"PACE_AVG{W}"] = _g["PACE"].rolling(W, min_periods=1).mean().values
    series = series.sort_values("GAME_DATE")
    key = df[["OPP", "GAME_DATE"]]
    key["_pos"] = key.index  # original row labels BEFORE sorting (RangeIndex)
    m = pd.merge_asof(key.sort_values("GAME_DATE"), series, left_on="GAME_DATE", right_on="GAME_DATE",
                      left_by="OPP", right_by="TEAM_ABBREVIATION",
                      direction="backward", allow_exact_matches=False)
    m = m.sort_values("_pos")
    df[f"OPP_ALLOWED_AVG{W}"] = m[f"ALLOWED_AVG{W}"].values
    df[f"OPP_PACE_AVG{W}"] = m[f"PACE_AVG{W}"].values
    return df


FEATURES = ["IS_HOME", "REST_DAYS", "BACK_TO_BACK", "IS_PLAYOFF", "GAME_MONTH",
            f"PTS_AVG{W}", f"PACE_AVG{W}", f"ALLOWED_AVG{W}",
            f"OPP_ALLOWED_AVG{W}", f"OPP_PACE_AVG{W}"]


def main():
    try:
        import lightgbm as lgb
        reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1)
        use_lgbm = True
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                            max_leaf_nodes=63, min_samples_leaf=200, random_state=42)
        use_lgbm = False
    print("Building team-game frame...")
    df = build_frame().dropna(subset=["PTS"]).sort_values("GAME_DATE")
    feats = [f for f in FEATURES if f in df.columns]
    X, y = df[feats].fillna(-1), df["PTS"]
    cutoff = pd.Timestamp("2019-01-01")
    tr, te = df["GAME_DATE"] < cutoff, df["GAME_DATE"] >= cutoff
    print(f"Train {int(tr.sum())} / Test {int(te.sum())} (cutoff 2019-01-01)")
    if use_lgbm:
        reg.fit(X[tr], y[tr], eval_set=[(X[te], y[te])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        reg.fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    mae = mean_absolute_error(y[te], pred)
    print(f"Team total MAE: {mae:.2f}")
    importances = getattr(reg, "feature_importances_", [0] * len(feats))
    pd.DataFrame({"feature": feats, "importance": importances}).sort_values(
        "importance", ascending=False).to_csv("models/team_feature_importance.csv", index=False)
    pickle.dump({"model": reg, "features": feats}, open(MODEL_FILE, "wb"))
    import json
    metrics = json.loads(Path("models/metrics.json").read_text())
    metrics["team_total_mae"] = round(float(mae), 3)
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/team_total_lgbm.pkl")


def team_total_for(team, opp, date, is_home, rest_days=2, back_to_back=0):
    """Single inference from rolling team-game avgs (full logs or snapshot)."""
    bundle = pickle.load(open(MODEL_FILE, "rb"))
    df = team_games_df()
    date = pd.to_datetime(date)
    past = df[(df["TEAM_ABBREVIATION"] == team.upper()) & (df["GAME_DATE"] < date)].tail(W)
    opast = df[(df["TEAM_ABBREVIATION"] == opp.upper()) & (df["GAME_DATE"] < date)].tail(W)
    if not len(past) or not len(opast):
        return None
    own_pace = (past["FGA"] + 0.44 * past["FTA"] - past["OREB"] - past["TOV"]).mean()
    opp_pace = (opast["FGA"] + 0.44 * opast["FTA"] - opast["OREB"] - opast["TOV"]).mean()
    row = {"IS_HOME": int(is_home), "REST_DAYS": rest_days, "BACK_TO_BACK": back_to_back,
           "IS_PLAYOFF": 0, "GAME_MONTH": date.month,
           f"PTS_AVG{W}": past["PTS"].mean(), f"PACE_AVG{W}": own_pace,
           f"ALLOWED_AVG{W}": past["ALLOWED"].mean() if "ALLOWED" in past else np.nan,
           f"OPP_ALLOWED_AVG{W}": opast["ALLOWED"].mean() if "ALLOWED" in opast else np.nan,
           f"OPP_PACE_AVG{W}": opp_pace}
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in bundle["features"]}]).fillna(-1)
    return float(bundle["model"].predict(X)[0])


if __name__ == "__main__":
    main()
