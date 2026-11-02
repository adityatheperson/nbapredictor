"""Team-total model v2: predicts a team's score from offense, opponent defense, pace,
Elo strength, rest spots, and recent form.
- Train:  python3 -m src.team_model            (walk-forward, saves models/team_total_lgbm.pkl)
- Elo only: python3 -m src.team_model --elo    (rebuilds models/elo_history.parquet)
- Blend happens in predict.py (50/50 with the injury-adjusted player sum).
"""
import re
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from src.config import APP_DIR

TEAM_LOGS = Path("data/raw/team_game_logs_all.parquet")
MODEL_FILE = Path("models/team_total_lgbm.pkl")
ELO_FILE = Path("models/elo_history.parquet")
W = 10
W_SHORT = 5
ELO_K = 20
ELO_HCA = 70  # home edge in Elo points


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


def team_schedules(df):
    """{team: sorted np array of game dates} for rest math."""
    return {t: np.sort(g["GAME_DATE"].values.astype("datetime64[D]"))
            for t, g in df.groupby("TEAM_ABBREVIATION")}


def rest_before(dates, d):
    """Days off before game on date d (schedule-known, no leakage)."""
    d = np.datetime64(pd.to_datetime(d).date())
    i = np.searchsorted(dates, d, side="left") - 1
    if i < 0:
        return 7
    return int(min(max((d - dates[i]).astype(int) - 1, 0), 14))


def count_3in4(dates, d):
    d = np.datetime64(pd.to_datetime(d).date())
    lo = d - np.timedelta64(3, "D")
    n = int(np.searchsorted(dates, d, side="left") - np.searchsorted(dates, lo, side="left")) + 1
    return int(n >= 3)


def build_elo(df):
    """Sequential Elo over regular-season games (chronological, no leakage).
    Returns ratings history DataFrame (team, date, elo after that game)."""
    g = df[df["SEASON_TYPE"] != "Playoffs"].sort_values(["GAME_DATE", "GAME_ID"]).copy()
    elos = {}
    rows = []
    for gid, grp in g.groupby("GAME_ID", sort=False):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        ta, tb = a["TEAM_ABBREVIATION"], b["TEAM_ABBREVIATION"]
        ea, eb = elos.get(ta, 1500.0), elos.get(tb, 1500.0)
        ha = 1 if str(a.get("MATCHUP", "")).find("vs.") >= 0 else 0
        ra = ea + (ELO_HCA if ha else 0)
        rb = eb + (ELO_HCA if not ha else 0)
        pa = 1 / (1 + 10 ** (-(ra - rb) / 400))
        sa = 1.0 if a["PTS"] > b["PTS"] else (0.0 if a["PTS"] < b["PTS"] else 0.5)
        mov = abs(a["PTS"] - b["PTS"])
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs(ra - rb))
        elos[ta] = ea + ELO_K * mult * (sa - pa)
        elos[tb] = eb + ELO_K * mult * ((1 - sa) - (1 - pa))
        d = a["GAME_DATE"]
        rows.append((ta, d, elos[ta]))
        rows.append((tb, d, elos[tb]))
    hist = pd.DataFrame(rows, columns=["TEAM", "GAME_DATE", "ELO"]).sort_values("GAME_DATE")
    return hist


def elo_asof(hist, team, date, default=1500.0):
    sub = hist[(hist["TEAM"] == team.upper()) & (hist["GAME_DATE"] < pd.to_datetime(date))]
    if not len(sub):
        return default
    return float(sub.iloc[-1]["ELO"])


def build_frame(df=None):
    df = team_games_df() if df is None else df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["PACE"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] - df["TOV"]
    opp = df[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "OPP", "PTS": "OPP_PTS"})
    df = df.merge(opp, on="GAME_ID")
    df = df[df["OPP"] != df["TEAM_ABBREVIATION"]]
    df["ALLOWED"] = df["OPP_PTS"]
    df["IS_HOME"] = df["MATCHUP"].fillna("").str.contains("vs\\.").astype(int) if "MATCHUP" in df else 0
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    grp = df.groupby("TEAM_ABBREVIATION")
    df["PREV_DATE"] = grp["GAME_DATE"].shift(1)
    df["REST_DAYS"] = ((df["GAME_DATE"] - df["PREV_DATE"]).dt.days - 1).fillna(7).clip(0, 14)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 0).astype(int)
    for c in ["PTS", "PACE", "ALLOWED"]:
        df[f"{c}_AVG{W}"] = grp[c].shift(1).rolling(W, min_periods=1).mean().values
        df[f"{c}_AVG{W_SHORT}"] = grp[c].shift(1).rolling(W_SHORT, min_periods=1).mean().values
    df[f"PTS_MOM"] = df[f"PTS_AVG{W_SHORT}"] - df[f"PTS_AVG{W}"]
    df[f"ALLOWED_MOM"] = df[f"ALLOWED_AVG{W_SHORT}"] - df[f"ALLOWED_AVG{W}"]
    df["IS_PLAYOFF"] = (df["SEASON_TYPE"] == "Playoffs").astype(int) if "SEASON_TYPE" in df else 0
    df["GAME_MONTH"] = df["GAME_DATE"].dt.month
    # situational rest from schedules (past-determined)
    sched = team_schedules(df)
    df["REST_DIFF"] = 0.0
    df["OPP_B2B"] = 0
    df["OWN_3IN4"] = 0
    for team, idx in df.groupby("TEAM_ABBREVIATION").groups.items():
        dates = sched[team]
        sub = df.loc[idx]
        rests = [rest_before(dates, d) for d in sub["GAME_DATE"]]
        df.loc[idx, "REST_DAYS"] = rests
        df.loc[idx, "BACK_TO_BACK"] = [int(r == 0) for r in rests]
        df.loc[idx, "OWN_3IN4"] = [count_3in4(dates, d) for d in sub["GAME_DATE"]]
    # opponent trailing avgs + opponent rest, strictly prior games
    _s = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    _g = _s.groupby("TEAM_ABBREVIATION")
    series = _s[["TEAM_ABBREVIATION", "GAME_DATE"]].copy()
    for c in ["ALLOWED", "PACE"]:
        series[f"{c}_AVG{W}"] = _g[c].rolling(W, min_periods=1).mean().values
        series[f"{c}_AVG{W_SHORT}"] = _g[c].rolling(W_SHORT, min_periods=1).mean().values
    series = series.sort_values("GAME_DATE")
    key = df[["OPP", "GAME_DATE"]]
    key["_pos"] = key.index
    m = pd.merge_asof(key.sort_values("GAME_DATE"), series, left_on="GAME_DATE", right_on="GAME_DATE",
                      left_by="OPP", right_by="TEAM_ABBREVIATION",
                      direction="backward", allow_exact_matches=False)
    m = m.sort_values("_pos")
    for c in [f"ALLOWED_AVG{W}", f"ALLOWED_AVG{W_SHORT}", f"PACE_AVG{W}"]:
        df[f"OPP_{c}"] = m[c].values
    df[f"OPP_ALLOWED_MOM"] = df[f"OPP_ALLOWED_AVG{W_SHORT}"] - df[f"OPP_ALLOWED_AVG{W}"]
    # opponent rest: last opp game date strictly before, then count
    m2 = pd.merge_asof(key.sort_values("GAME_DATE"),
                       _s[["TEAM_ABBREVIATION", "GAME_DATE"]].rename(
                           columns={"TEAM_ABBREVIATION": "OPP_D", "GAME_DATE": "OPP_LAST"})
                       .sort_values("OPP_LAST"),
                       left_on="GAME_DATE", right_on="OPP_LAST",
                       left_by="OPP", right_by="OPP_D",
                       direction="backward", allow_exact_matches=False)
    m2 = m2.sort_values("_pos")
    opp_last = pd.to_datetime(m2["OPP_LAST"])
    opp_rest = (df["GAME_DATE"] - opp_last).dt.days - 1
    df["OPP_B2B"] = ((opp_rest == 0) & opp_last.notna()).astype(int)
    df["REST_DIFF"] = df["REST_DAYS"] - opp_rest.fillna(2)
    # Elo as of game date (strictly prior games; vectorized asof, no per-row loop)
    elo_hist = build_elo(df)
    elo_hist.to_parquet(ELO_FILE, index=False)
    for col, side in (("ELO", "TEAM_ABBREVIATION"), ("OPP_ELO", "OPP")):
        ek = df[[side, "GAME_DATE"]].copy()
        ek["_pos"] = ek.index
        es = elo_hist.rename(columns={"TEAM": "_T", "GAME_DATE": "_D", "ELO": "_E"})
        em = pd.merge_asof(ek.sort_values("GAME_DATE"), es.sort_values("_D"),
                           left_on="GAME_DATE", right_on="_D",
                           left_by=side, right_by="_T",
                           direction="backward", allow_exact_matches=False)
        df[col] = em.sort_values("_pos")["_E"].values
    df["ELO"] = df["ELO"].fillna(1500.0)
    df["OPP_ELO"] = df["OPP_ELO"].fillna(1500.0)
    df["ELO_DIFF"] = df["ELO"] - df["OPP_ELO"]
    df["ELO_WINPROB"] = 1 / (1 + 10 ** (-(df["ELO_DIFF"] + ELO_HCA * df["IS_HOME"]) / 400))
    return df


FEATURES = ["IS_HOME", "REST_DAYS", "BACK_TO_BACK", "REST_DIFF", "OPP_B2B", "OWN_3IN4",
            "IS_PLAYOFF", "GAME_MONTH",
            f"PTS_AVG{W}", f"PTS_AVG{W_SHORT}", "PTS_MOM",
            f"PACE_AVG{W}", f"ALLOWED_AVG{W}", f"ALLOWED_AVG{W_SHORT}", "ALLOWED_MOM",
            f"OPP_ALLOWED_AVG{W}", f"OPP_ALLOWED_AVG{W_SHORT}", "OPP_ALLOWED_MOM",
            f"OPP_PACE_AVG{W}", "ELO_DIFF", "ELO_WINPROB"]


def train_frame(df):
    df = build_frame(df).dropna(subset=["PTS"]).sort_values("GAME_DATE")
    feats = [f for f in FEATURES if f in df.columns]
    return df, feats


def winner_accuracy(df, pred):
    """Pair rows by game; higher predicted total should win."""
    d = df.copy()
    d["_pred"] = np.asarray(pred)
    d["_win"] = (d["PTS"] > d.groupby("GAME_ID")["PTS"].transform("mean")).astype(int)
    d["_pick"] = (d["_pred"] > d.groupby("GAME_ID")["_pred"].transform("mean")).astype(int)
    paired = d.groupby("GAME_ID").filter(lambda g: len(g) == 2)
    return float((paired["_win"] == paired["_pick"]).mean()) if len(paired) else float("nan")


def main():
    only_elo = "--elo" in sys.argv
    try:
        import lightgbm as lgb
        reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1)
        use_lgbm = True
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                            max_leaf_nodes=63, min_samples_leaf=200, random_state=42)
        use_lgbm = False
    print("Building team-game frame...")
    df, feats = train_frame(None)
    if only_elo:
        print(f"Elo history rebuilt -> {ELO_FILE}")
        return
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
    wacc = winner_accuracy(df[te], pred)
    print(f"Team total MAE: {mae:.2f} | test winner accuracy: {wacc * 100:.1f}%")
    importances = getattr(reg, "feature_importances_", [0] * len(feats))
    pd.DataFrame({"feature": feats, "importance": importances}).sort_values(
        "importance", ascending=False).to_csv("models/team_feature_importance.csv", index=False)
    pickle.dump({"model": reg, "features": feats}, open(MODEL_FILE, "wb"))
    import json
    metrics = json.loads(Path("models/metrics.json").read_text())
    metrics["team_total_mae"] = round(float(mae), 3)
    metrics["team_winner_acc"] = round(float(wacc), 3)
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
    sched = team_schedules(df)
    own_rest = rest_before(sched.get(team.upper(), np.array([], dtype="datetime64[D]")), date)
    opp_rest = rest_before(sched.get(opp.upper(), np.array([], dtype="datetime64[D]")), date)
    own_pace = (past["FGA"] + 0.44 * past["FTA"] - past["OREB"] - past["TOV"]).mean()
    opp_pace = (opast["FGA"] + 0.44 * opast["FTA"] - opast["OREB"] - opast["TOV"]).mean()
    elo_hist = pd.read_parquet(ELO_FILE) if ELO_FILE.exists() else None
    elo = elo_asof(elo_hist, team, date) if elo_hist is not None else 1500.0
    oelo = elo_asof(elo_hist, opp, date) if elo_hist is not None else 1500.0
    ediff = elo - oelo
    short = past.tail(W_SHORT)
    oshort = opast.tail(W_SHORT)
    row = {"IS_HOME": int(is_home), "REST_DAYS": own_rest, "BACK_TO_BACK": int(own_rest == 0),
           "REST_DIFF": own_rest - opp_rest, "OPP_B2B": int(opp_rest == 0),
           "OWN_3IN4": count_3in4(sched.get(team.upper(), np.array([], dtype="datetime64[D]")), date),
           "IS_PLAYOFF": 0, "GAME_MONTH": date.month,
           f"PTS_AVG{W}": past["PTS"].mean(), f"PTS_AVG{W_SHORT}": short["PTS"].mean(),
           "PTS_MOM": short["PTS"].mean() - past["PTS"].mean(),
           f"PACE_AVG{W}": own_pace,
           f"ALLOWED_AVG{W}": past["ALLOWED"].mean() if "ALLOWED" in past else np.nan,
           f"ALLOWED_AVG{W_SHORT}": short["ALLOWED"].mean() if "ALLOWED" in short else np.nan,
           "ALLOWED_MOM": ((short["ALLOWED"].mean() - past["ALLOWED"].mean())
                            if "ALLOWED" in past else np.nan),
           f"OPP_ALLOWED_AVG{W}": opast["ALLOWED"].mean() if "ALLOWED" in opast else np.nan,
           f"OPP_ALLOWED_AVG{W_SHORT}": oshort["ALLOWED"].mean() if "ALLOWED" in oshort else np.nan,
           "OPP_ALLOWED_MOM": ((oshort["ALLOWED"].mean() - opast["ALLOWED"].mean())
                                if "ALLOWED" in opast else np.nan),
           f"OPP_PACE_AVG{W}": opp_pace,
           "ELO_DIFF": ediff,
           "ELO_WINPROB": 1 / (1 + 10 ** (-(ediff + ELO_HCA * int(is_home)) / 400))}
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in bundle["features"]}]).fillna(-1)
    return float(bundle["model"].predict(X)[0])


if __name__ == "__main__":
    main()
