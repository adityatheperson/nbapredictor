"""Inference: given 2 NBA teams, predict each player's PTS, team totals, winner.
Usage:
  python3 -m src.predict LAL BOS --home LAL
Works with trained model (models/player_pts_lgbm.pkl) or baseline fallback (PTS_AVG10).
"""
import argparse
import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features_rolling import ROLL_COLS, WINDOWS
from src.injuries import team_injuries, match_out_players, fetch_report
from src.team_model import team_total_for
from src.config import now

APP_DIR = Path("data/app")
MODEL_FILE = Path("models/player_pts_lgbm.pkl")

BENCH_CORRECTION = 0.93  # top-10 scorers account for ~93% of team points
DIFF_SIGMA = 13.5        # std of (total_A - total_B) for win prob
N_SIMS = 10_000
QUESTIONABLE_SCALE = 0.75  # expected-output discount for game-time decisions
REDIST_SHARE = 0.35        # share of a scratched player's scoring absorbed by teammates
# (teammates absorb most of the missing usage, but at worse efficiency, and the
# promoted bench players were already counted in the 7% bench allowance, so the
# net team impact of an absence is modestly negative — never positive, see cap)


def _resolve_teams():
    from nba_api.stats.static import teams as static_teams
    return {t["abbreviation"]: t for t in static_teams.get_teams()}


def get_roster(team_abbrev):
    """Live roster via CommonTeamRoster. Returns DataFrame with PLAYER_ID, PLAYER."""
    from nba_api.stats.static import teams as static_teams
    from nba_api.stats.endpoints import CommonTeamRoster
    all_teams = static_teams.get_teams()
    match = [t for t in all_teams if t["abbreviation"] == team_abbrev.upper()]
    if not match:
        raise ValueError(f"Unknown team abbrev: {team_abbrev}")
    tid = match[0]["id"]
    roster = CommonTeamRoster(team_id=tid, timeout=30).get_data_frames()[0]
    return roster


def get_snapshot_roster(team_abbrev, days=21):
    """Roster from our own recent game logs (works anywhere, no NBA API).
    Players whose latest game (within `days` of the newest data) was for `team_abbrev`."""
    recent = pd.read_parquet(APP_DIR / "player_recent.parquet",
                             columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE"])
    recent["GAME_DATE"] = pd.to_datetime(recent["GAME_DATE"])
    team_abbrev = team_abbrev.upper()
    latest = recent.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last")
    cutoff = recent["GAME_DATE"].max() - pd.Timedelta(days=days)
    g = latest[(latest["TEAM_ABBREVIATION"] == team_abbrev) & (latest["GAME_DATE"] >= cutoff)]
    return pd.DataFrame({"PLAYER_ID": g["PLAYER_ID"].values, "PLAYER": g["PLAYER_NAME"].values})


def resolve_rosters(team_a, team_b):
    """Snapshot rosters first (offline-safe); live NBA API only as fallback."""
    rosters = {}
    for t in (team_a, team_b):
        try:
            ros = get_snapshot_roster(t)
            if len(ros) < 8:
                raise ValueError(f"only {len(ros)} recent players")
            rosters[t] = ros
        except Exception:
            rosters[t] = get_roster(t)
    return rosters


def _player_history(player_id, hist):
    if hist is None or len(hist) == 0:
        return None
    g = hist[hist["PLAYER_ID"] == player_id].sort_values("GAME_DATE")
    return g if len(g) else None


def load_snapshot():
    """Runtime history from data/app/ (built by src.snapshot). Returns (recent, career, bios)."""
    recent = pd.read_parquet(APP_DIR / "player_recent.parquet")
    recent["GAME_DATE"] = pd.to_datetime(recent["GAME_DATE"])
    career = pd.read_parquet(APP_DIR / "player_career.parquet").set_index("PLAYER_ID")
    bios = pd.read_parquet(APP_DIR / "player_bios.parquet").drop_duplicates("PERSON_ID").set_index("PERSON_ID")
    return recent, career, bios


def build_feature_row(player_id, player_name, opp_abbrev, is_home, game_date, hist, bios, feature_names,
                      career=None):
    """Build one model-ready row from history (no leakage: only past games).
    hist holds each player's most recent games; career holds full-history averages."""
    g = _player_history(player_id, hist) if hist is not None else None
    row = {}
    if g is not None and len(g) >= 1:
        last = g.iloc[-1]
        for w in WINDOWS:
            window = g.tail(w)
            for c in ROLL_COLS:
                vals = window[c].dropna() if c in window.columns else pd.Series([], dtype=float)
                row[f"{c}_AVG{w}"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{c}_STD{w}"] = float(vals.std()) if len(vals) > 1 else 0.0
            row[f"GP_LAST{w}"] = int(min(len(window), w))
        cr = career.loc[player_id] if career is not None and player_id in career.index else None
        for c in ["PTS", "MIN_NUM", "REB", "AST"]:
            if cr is not None:
                row[f"{c}_CAREER_AVG"] = float(cr[f"{c}_CAREER_AVG"])
            else:
                row[f"{c}_CAREER_AVG"] = float(g[c].mean()) if c in g.columns else np.nan
        row["CAREER_GP"] = int(cr["CAREER_GP"]) if cr is not None else int(len(g))
        row["PTS_L1_vs_AVG10"] = float(g["PTS"].iloc[-1] - row.get("PTS_AVG10", np.nan))
        # context from last game + new matchup
        row["REST_DAYS"] = 2
        row["BACK_TO_BACK"] = 0
        row["GAMES_LAST_7D"] = int((g.tail(5).shape[0] > 3))
        # bio
        if bios is not None and player_id in bios.index:
            b = bios.loc[player_id]
            try:
                bdate = pd.to_datetime(b.get("BIRTHDATE"))
                row["AGE_AT_GAME"] = (pd.to_datetime(game_date) - bdate).days / 365.25
            except Exception:
                row["AGE_AT_GAME"] = 27.0
            row["HEIGHT_IN"] = _height_to_in(b.get("HEIGHT"))
            row["WEIGHT_NUM"] = _to_num(b.get("WEIGHT"), 220)
            row["DRAFT_NUMBER_NUM"] = _to_num(b.get("DRAFT_NUMBER"), 61)
            row["IS_UNDRAFTED"] = int(row["DRAFT_NUMBER_NUM"] == 61)
            row["SEASON_EXP_AT_FETCH"] = _to_num(b.get("SEASON_EXP"), 3)
        else:
            row["AGE_AT_GAME"] = 27.0
            row["HEIGHT_IN"] = np.nan
            row["WEIGHT_NUM"] = 220.0
            row["DRAFT_NUMBER_NUM"] = 61.0
            row["IS_UNDRAFTED"] = 1
            row["SEASON_EXP_AT_FETCH"] = 3.0
        # last-game derived flags baseline
        row["MIN_NUM"] = row.get("MIN_NUM_AVG5", 20.0)
    else:
        # Rookie / unknown: league-average priors
        for w in WINDOWS:
            for c in ROLL_COLS:
                row[f"{c}_AVG{w}"] = np.nan
                row[f"{c}_STD{w}"] = 0.0
            row[f"GP_LAST{w}"] = 0
        for c in ["PTS", "MIN_NUM", "REB", "AST"]:
            row[f"{c}_CAREER_AVG"] = np.nan
        row["CAREER_GP"] = 0
        row["PTS_L1_vs_AVG10"] = 0.0
        row["REST_DAYS"] = 2
        row["BACK_TO_BACK"] = 0
        row["GAMES_LAST_7D"] = 0
        row["AGE_AT_GAME"] = 22.0
        row["HEIGHT_IN"] = np.nan
        row["WEIGHT_NUM"] = 220.0
        row["DRAFT_NUMBER_NUM"] = 30.0
        row["IS_UNDRAFTED"] = 0
        row["SEASON_EXP_AT_FETCH"] = 0.0
        row["MIN_NUM"] = 15.0

    row["AGE_SQ"] = row["AGE_AT_GAME"] ** 2
    row["PTS_AVG10_x_AGE"] = (row.get("PTS_AVG10") or 0) * row["AGE_AT_GAME"]
    row["IS_HOME"] = int(is_home)
    row["GAME_MONTH"] = game_date.month
    row["GAME_WEEKDAY"] = game_date.weekday()
    row["SEASON_YEAR"] = game_date.year
    row["DOUBLE_DOUBLE"] = 0
    row["PTS_30"] = 0
    row["PTS_20"] = 0
    row["DID_NOT_PLAY"] = 0
    # derived per-min priors
    for c in ["EFG_PCT", "TS_PCT", "PTS_PER_MIN", "REB_PER_MIN", "AST_PER_MIN", "AST_TOV"]:
        row.setdefault(c, 0.0)
    for c in ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
              "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS", "FANTASY_PTS"]:
        row.setdefault(c, row.get(f"{c}_AVG10", 0.0) or 0.0)
    # order to model features
    return pd.Series({f: row.get(f, np.nan) for f in feature_names})


def _height_to_in(h):
    import re
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return np.nan
    m = re.match(r"(\d+)-(\d+)", str(h))
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    try:
        return float(h)
    except Exception:
        return np.nan


def _to_num(v, default):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return default
        return float(v)
    except Exception:
        return default


def load_model():
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["features"], True
    return None, None, False


def predict_matchup(team_a, team_b, home_team=None, game_date=None, rosters=None, hist=None, bios=None, career=None,
                    injuries="auto", extra_out=None, questionable_scale=QUESTIONABLE_SCALE,
                    redist_share=REDIST_SHARE):
    """injuries: 'auto' (ESPN report), None (ignore), or {TEAM: [player names]} treated as OUT.
    extra_out: {TEAM: [names]} merged on top (e.g. manual scratches from UI)."""
    team_a, team_b = team_a.upper(), team_b.upper()
    game_date = pd.to_datetime(game_date) if game_date else now().normalize()
    home_team = (home_team or team_a).upper()
    extra_out = extra_out or {}

    # --- resolve injuries ---
    auto_out, auto_q = {}, {}
    if injuries == "auto":
        try:
            report = fetch_report()
            for t in (team_a, team_b):
                injs = team_injuries(t, report)
                auto_out[t] = [i["player"] for i in injs if i["category"] == "out"]
                auto_q[t] = [i["player"] for i in injs if i["category"] == "questionable"]
        except Exception as e:
            print(f"Injury feed unavailable ({e}); proceeding without it.")
    elif isinstance(injuries, dict):
        for t in (team_a, team_b):
            auto_out[t] = list(injuries.get(t, []))
            auto_q[t] = []

    if hist is None or career is None or bios is None:
        snap_hist, snap_career, snap_bios = load_snapshot()
        hist = snap_hist if hist is None else hist
        career = snap_career if career is None else career
        bios = snap_bios if bios is None else bios

    if rosters is None:
        rosters = resolve_rosters(team_a, team_b)

    model, features, has_model = load_model()
    try:
        _mb = pickle.load(open("models/player_min_lgbm.pkl", "rb"))
        min_model, min_features = _mb["model"], _mb["features"]
    except Exception:
        min_model, min_features = None, None

    results = {}
    for team, opp in [(team_a, team_b), (team_b, team_a)]:
        ros = rosters[team]
        # normalize roster columns across API versions
        id_col = "PLAYER_ID" if "PLAYER_ID" in ros.columns else "PERSON_ID"
        name_col = "PLAYER" if "PLAYER" in ros.columns else "DISPLAY_FIRST_LAST"
        is_home = int(team == home_team)
        preds = []
        for _, r in ros.iterrows():
            pid = int(r[id_col])
            pname = str(r[name_col])
            if has_model:
                x = build_feature_row(pid, pname, opp, is_home, game_date, hist, bios, features, career)
                X = pd.DataFrame([x]).fillna(-1)
                pts = float(max(0.0, model.predict(X)[0]))
                sigma = float(x.get("PTS_STD10", 5.0) or 5.0)
                sigma = min(max(sigma, 4.0), 12.0)
                base_avg = float(x.get("PTS_AVG10", pts) or pts)
                pred_min, min_scale = None, 1.0
                if min_model is not None:
                    Xm = pd.DataFrame([{f: x.get(f, np.nan) for f in min_features}]).fillna(-1)
                    pred_min = float(max(0.0, min_model.predict(Xm)[0]))
                    # NOTE 2026-11-02: scaling PTS by pred/typical minutes was tested on
                    # 3 backtest nights and made things worse (4.93 vs ~4.6 MAE) — the
                    # minutes predictions are noisier than the scoring ones, and the
                    # constant-per-minute assumption doesn't hold. pred_min is kept
                    # for display only; scoring is NOT scaled.
            else:
                g = _player_history(pid, hist) if hist is not None else None
                pts = float(g.tail(10)["PTS"].mean()) if g is not None and len(g) else 8.0
                sigma = float(g.tail(10)["PTS"].std()) if g is not None and len(g) > 1 else 6.0
                sigma = min(max(sigma or 6.0, 4.0), 12.0)
                base_avg = pts
                pred_min, min_scale = None, 1.0
            preds.append({"player": pname, "player_id": pid, "pred_pts": pts,
                          "sigma": sigma, "base_avg10": base_avg,
                          "pred_min": round(pred_min, 1) if pred_min is not None else None})
        pdf = pd.DataFrame(preds).sort_values("pred_pts", ascending=False).reset_index(drop=True)
        # --- apply injuries: scratch OUTs, discount QUESTIONABLEs, redistribute ---
        roster_names = pdf["player"].tolist()
        out_names = match_out_players(roster_names,
                                      list(auto_out.get(team, [])) + list(extra_out.get(team, [])))
        q_names = match_out_players(roster_names, list(auto_q.get(team, []))) - out_names
        pdf["status"] = "OK"
        pdf.loc[pdf["player"].isin(out_names), "status"] = "OUT"
        pdf.loc[pdf["player"].isin(q_names), "status"] = "Q"
        no_injury_total = pdf["pred_pts"].head(10).sum() / BENCH_CORRECTION
        pdf.loc[pdf["status"] == "Q", "pred_pts"] *= questionable_scale
        removed_pts = float(pdf.loc[pdf["status"] == "OUT", "pred_pts"].sum())
        avail = pdf[pdf["status"] != "OUT"].sort_values("pred_pts", ascending=False)
        top10 = avail.head(10)["pred_pts"].sum()
        total = top10 / BENCH_CORRECTION + redist_share * removed_pts
        total = min(total, no_injury_total)  # absences can't improve a team
        results[team] = {"players": pdf, "total": float(total), "is_home": is_home,
                         "out": sorted(out_names), "questionable": sorted(q_names),
                         "removed_pts": removed_pts}

    # --- blend player-sum total with the team-context model (pace, opponent D, rest) ---
    for team, opp in [(team_a, team_b), (team_b, team_a)]:
        r = results[team]
        rest, b2b = 2, 0
        if hist is not None and "TEAM_ABBREVIATION" in hist.columns:
            past_dates = hist[hist["TEAM_ABBREVIATION"] == team]["GAME_DATE"]
            if len(past_dates):
                gd = pd.to_datetime(game_date)
                gd = gd.tz_localize(None) if getattr(gd, "tzinfo", None) else gd
                rest = int(max(0, (gd - past_dates.max()).days - 1))
                rest = int(min(rest, 14))
                b2b = int(rest == 0)
        try:
            tm = team_total_for(team, opp, game_date, r["is_home"], rest, b2b)
        except Exception:
            tm = None
        if tm is None:
            blended, tm_shown = r["total"], None
        else:
            tm_adj = tm + redist_share * r["removed_pts"]
            blended = 0.5 * r["total"] + 0.5 * tm_adj
            tm_shown = round(tm_adj, 1)
        r["playersum_total"] = round(r["total"], 1)
        r["teammodel_total"] = tm_shown
        r["total"] = float(blended)

    ta, tb = results[team_a]["total"], results[team_b]["total"]
    diff = ta - tb
    # analytic win prob (normal diff) + Monte Carlo with player variance
    p_a_analytic = 0.5 * (1 + math.erf(diff / (DIFF_SIGMA * math.sqrt(2))))
    rng = np.random.default_rng(42)
    sim_a, sim_b = np.zeros(N_SIMS), np.zeros(N_SIMS)
    for team, arr in [(team_a, sim_a), (team_b, sim_b)]:
        pdf = results[team]["players"]
        avail = pdf[pdf["status"] != "OUT"].sort_values("pred_pts", ascending=False).head(10)
        sims = np.zeros(N_SIMS)
        for _, pr in avail.iterrows():
            sims += np.maximum(0, rng.normal(pr["pred_pts"], pr["sigma"], N_SIMS))
        sims = sims / BENCH_CORRECTION + redist_share * results[team]["removed_pts"]
        # recenter sims on the blended total (keeps player-driven variance, fixes the mean)
        base = sims.mean()
        if base > 0:
            sims = sims * (results[team]["total"] / base)
        arr[:] = sims
    p_a_mc = float((sim_a > sim_b).mean())
    winner = team_a if p_a_mc >= 0.5 else team_b
    return {
        "team_a": team_a, "team_b": team_b, "home_team": home_team,
        "date": str(game_date.date()),
        "totals": {team_a: round(ta, 1), team_b: round(tb, 1)},
        "win_prob": {team_a: round(p_a_mc, 3), team_b: round(1 - p_a_mc, 3)},
        "win_prob_analytic": {team_a: round(p_a_analytic, 3), team_b: round(1 - p_a_analytic, 3)},
        "spread": round(diff, 1),
        "winner": winner,
        "sim_totals": {team_a: sim_a, team_b: sim_b},
        "players": {team_a: results[team_a]["players"], team_b: results[team_b]["players"]},
        "blend": {t: {"playersum": results[t]["playersum_total"],
                      "teammodel": results[t]["teammodel_total"]} for t in (team_a, team_b)},
        "out": {team_a: results[team_a]["out"], team_b: results[team_b]["out"]},
        "questionable": {team_a: results[team_a]["questionable"], team_b: results[team_b]["questionable"]},
        "used_model": has_model,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team_a")
    ap.add_argument("team_b")
    ap.add_argument("--home", default=None)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    out = predict_matchup(args.team_a, args.team_b, home_team=args.home, game_date=args.date)
    print(f"{out['team_a']} {out['totals'][out['team_a']]} vs {out['team_b']} {out['totals'][out['team_b']]} "
          f"on {out['date']} (home: {out['home_team']}) [{'MODEL' if out['used_model'] else 'BASELINE'}]")
    print(f"Winner: {out['winner']}  P({out['team_a']})={out['win_prob'][out['team_a']]}  spread={out['spread']}")
    for t in [out["team_a"], out["team_b"]]:
        if out["out"][t]:
            print(f"  {t} OUT: {', '.join(out['out'][t])}")
        if out["questionable"][t]:
            print(f"  {t} QUESTIONABLE: {', '.join(out['questionable'][t])}")
    for t in [out["team_a"], out["team_b"]]:
        print(f"\n--- {t} ---")
        print(out["players"][t].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
