"""Point-in-time backtest: predict every game on a past date using ONLY data from
before tip-off, then grade against the actual results. Works with the full local
history or, when that's absent (hosted app), with the data/app snapshot plus ESPN
(schedule + final scores; player-level grading needs the full logs).
Usage:
  python3 -m src.backtest --date 2026-10-30
  python3 -m src.backtest --date 2026-10-30 --out "BOS:Tatum,LAL:Doncic"
"""
import argparse
import re
import pandas as pd
from src.predict import predict_matchup
from src.config import TRAINING_FILE, BIOS_FILE, APP_DIR

ROT_MIN_THRESHOLD = 12.0   # prior-10 min/game that counts as "rotation"
OUT_STREAK = 3             # consecutive team games missed around D => reconstructed OUT
ROSTER_WINDOW_DAYS = 21


def parse_matchup_home_away(matchup):
    m = str(matchup)
    at = re.match(r"([A-Z]{2,3})\s*@\s*([A-Z]{2,3})", m)
    if at:
        return at.group(1), at.group(2)
    vs = re.match(r"([A-Z]{2,3})\s*vs\.\s*([A-Z]{2,3})", m)
    if vs:
        return vs.group(2), vs.group(1)
    return None, None


def season_year(d):
    return d.year if d.month >= 10 else d.year - 1


def games_on(logs, date):
    """[(game_id, away, home)] for one day, parsed from MATCHUP."""
    day = logs[logs["GAME_DATE"] == date]
    out = []
    for gid, grp in day.groupby("GAME_ID"):
        away = home = None
        for m in grp["MATCHUP"].dropna().unique():
            away, home = parse_matchup_home_away(m)
            if away and home:
                break
        if away and home:
            out.append((str(gid), away, home))
    return out


def roster_asof(hist, team, date, window_days=ROSTER_WINDOW_DAYS):
    """Players whose latest game strictly before `date` (within window) was for `team`."""
    cutoff = date - pd.Timedelta(days=window_days)
    h = hist[(hist["GAME_DATE"] < date) & (hist["GAME_DATE"] >= cutoff)]
    latest = h.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last")
    g = latest[latest["TEAM_ABBREVIATION"] == team]
    return pd.DataFrame({"PLAYER_ID": g["PLAYER_ID"].values, "PLAYER": g["PLAYER_NAME"].values})


def build_ctx(logs, date):
    """Precomputed lookups so absence checks are dict gets, not table scans.
    Rotation standing comes strictly from games before `date`; the absence span
    itself may use surrounding games (injuries span games — this reconstructs the
    injury truth for grading, while all predictive features stay pre-date)."""
    hh = logs.sort_values("GAME_DATE")
    min_lookup = hh.set_index(["GAME_ID", "PLAYER_ID"])["MIN_NUM"].fillna(0).to_dict()
    team_games = {}
    for team, grp in hh.groupby("TEAM_ABBREVIATION"):
        order = grp.groupby("GAME_ID")["GAME_DATE"].min().sort_values().index.tolist()
        team_games[team] = order
    pre = hh[hh["GAME_DATE"] < date]
    tail10 = pre.groupby(["TEAM_ABBREVIATION", "PLAYER_ID"]).tail(10)
    avgmin = tail10.groupby(["TEAM_ABBREVIATION", "PLAYER_ID"])["MIN_NUM"].mean().to_dict()
    names = hh.drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["PLAYER_NAME"].to_dict()
    return {"min": min_lookup, "games": team_games, "avgmin": avgmin, "names": names}


def reconstruct_outs(ctx, team, game_id, roster_pids, streak=OUT_STREAK):
    """Rotation players missing this game as part of a >=streak absence span.
    Candidates are restricted to the as-of roster (kills retired-player false
    positives). Evidence comes from games around the date (injuries span games);
    single or double misses are treated as rest/DNP-CD and left in — the model
    eats that error, which is the honest pre-game position."""
    games = ctx["games"].get(team, [])
    if game_id not in games:
        return []
    outs = []
    for (t, pid), mn in ctx["avgmin"].items():
        if t != team or mn < ROT_MIN_THRESHOLD or pid not in roster_pids:
            continue
            continue
        if float(ctx["min"].get((game_id, pid), 0)) > 0:
            continue
        i = games.index(game_id)
        miss_before = 0
        for gid in reversed(games[:i + 1]):
            if float(ctx["min"].get((gid, pid), 0)) > 0:
                break
            miss_before += 1
        miss_after = 0
        for gid in games[i + 1:]:
            if float(ctx["min"].get((gid, pid), 0)) > 0:
                break
            miss_after += 1
        if miss_before + miss_after >= streak:
            outs.append(str(ctx["names"].get(pid, pid)))
    return sorted(outs)


def load_sources():
    """History source: full local logs, or the committed data/app snapshot (hosted).
    Returns dict with logs (or None), mode, bios_path."""
    if TRAINING_FILE.exists():
        logs = pd.read_parquet(TRAINING_FILE)
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.normalize()
        return {"logs": logs, "mode": "full",
                "bios_path": BIOS_FILE if BIOS_FILE.exists() else APP_DIR / "player_bios.parquet"}
    recent = pd.read_parquet(APP_DIR / "player_recent.parquet")
    recent["GAME_DATE"] = pd.to_datetime(recent["GAME_DATE"]).dt.normalize()
    return {"logs": recent, "mode": "snapshot", "bios_path": APP_DIR / "player_bios.parquet"}


def backtest(date, manual_out=None, verbose=True):
    from src.today import get_todays_games, espn_finals
    date = pd.to_datetime(date).normalize()
    src = load_sources()
    logs, mode = src["logs"], src["mode"]
    hist = logs[logs["GAME_DATE"] < date].copy()
    if not len(hist):
        raise ValueError("No history before that date.")

    bios = pd.read_parquet(src["bios_path"]).drop_duplicates("PERSON_ID")
    sy = season_year(date)
    bio_exp = pd.to_numeric(bios.get("FROM_YEAR"), errors="coerce")
    bios = bios.copy()
    bios["SEASON_EXP"] = (sy - bio_exp).fillna(0).clip(lower=0)
    bios = bios.set_index("PERSON_ID").rename(columns={"SEASON_EXP": "SEASON_EXP_AT_FETCH"})

    if mode == "full":
        career = hist.groupby("PLAYER_ID").agg(
            CAREER_GP=("PTS", "size"),
            PTS_CAREER_AVG=("PTS", "mean"),
            MIN_NUM_CAREER_AVG=("MIN_NUM", "mean"),
            REB_CAREER_AVG=("REB", "mean"),
            AST_CAREER_AVG=("AST", "mean"),
        )
        games = games_on(logs, date)
        finals = {}
    else:
        career = pd.read_parquet(APP_DIR / "player_career.parquet").set_index("PLAYER_ID")
        games = [(g["game_id"], g["away"], g["home"]) for g in get_todays_games(date)]
        finals = espn_finals(date)

    note = None
    if mode == "snapshot":
        note = ("Hosted backtest: history limited to recent games and career averages "
                "include later seasons; team-level grading from final scores, no player rows.")
    if verbose:
        print(f"{date.date()}: {len(games)} games, history {len(hist):,} rows ({mode})")
    manual_out = manual_out or {}
    ctx = build_ctx(logs, date)
    game_rows, player_rows = [], []
    for gid, away, home in games:
        rosters = {away: roster_asof(hist, away, date), home: roster_asof(hist, home, date)}
        outs = {away: reconstruct_outs(ctx, away, gid, set(rosters[away]["PLAYER_ID"].tolist())),
                home: reconstruct_outs(ctx, home, gid, set(rosters[home]["PLAYER_ID"].tolist()))}
        for t, names in (manual_out or {}).items():
            if t in outs:
                outs[t] = sorted(set(outs[t]) | set(names))
        try:
            out = predict_matchup(away, home, home_team=home, game_date=date,
                                  rosters=rosters, hist=hist, bios=bios, career=career,
                                  injuries=None, extra_out=outs)
        except Exception as e:
            if verbose:
                print(f"  {away}@{home}: FAILED {str(e)[:150]}")
            continue
        grows = logs[logs["GAME_ID"] == gid]
        by_team = grows.groupby("TEAM_ABBREVIATION")["PTS"].agg(["sum", "size"])
        need = 1 if mode == "full" else 8  # snapshot only holds recent games; require coverage
        a_act = float(by_team.loc[away, "sum"]) if away in by_team.index and by_team.loc[away, "size"] >= need else None
        h_act = float(by_team.loc[home, "sum"]) if home in by_team.index and by_team.loc[home, "size"] >= need else None
        if (a_act is None or h_act is None) and (away, home) in finals:
            a_act, h_act = (float(v) for v in finals[(away, home)])
        final = a_act is not None and h_act is not None
        actual_winner = (away if final and a_act > h_act else home) if final else None
        row = {"Away": away, "Home": home,
               "Pred": f"{out['totals'][away]}-{out['totals'][home]}",
               "Actual": f"{a_act:.0f}-{h_act:.0f}" if final else "—",
               "Pred winner": out["winner"], "Actual winner": actual_winner or "—",
               "Call": ("✅" if out["winner"] == actual_winner else "❌") if final else "—",
               "Outs": "; ".join(f"{t}:{','.join(outs[t])}" for t in (away, home) if outs[t])}
        if final:
            row["Away err"] = round(abs(out["totals"][away] - a_act), 1)
            row["Home err"] = round(abs(out["totals"][home] - h_act), 1)
        game_rows.append(row)
        if final:
            actual_pts = grows.set_index("PLAYER_ID")["PTS"]
            for t in (away, home):
                for _, pr in out["players"][t].iterrows():
                    pid = int(pr["player_id"])
                    if pid in actual_pts.index:
                        player_rows.append({
                            "Game": f"{away}@{home}", "Team": t, "Player": pr["player"],
                            "Pred": round(float(pr["pred_pts"]), 1),
                            "Actual": float(actual_pts.loc[pid]),
                            "Err": round(abs(float(pr["pred_pts"]) - float(actual_pts.loc[pid])), 1),
                            "Status": pr.get("status", "OK")})

    games_df = pd.DataFrame(game_rows)
    players_df = pd.DataFrame(player_rows)
    agg = {}
    if len(players_df):
        ok = players_df[players_df["Status"] != "OUT"]
        agg["player_pts_mae"] = round(float(players_df["Err"].mean()), 2)
        agg["player_pts_mae_available"] = round(float(ok["Err"].mean()), 2) if len(ok) else None
        agg["n_players"] = len(players_df)
    finals = games_df[games_df["Call"].isin(["✅", "❌"])]
    if len(finals):
        agg["winner_accuracy"] = f"{(finals['Call'] == '✅').mean() * 100:.0f}% ({(finals['Call'] == '✅').sum()}/{len(finals)})"
        errs = pd.concat([finals.get("Away err"), finals.get("Home err")]).dropna()
        if len(errs):
            agg["team_total_mae"] = round(float(errs.mean()), 2)
    return {"date": str(date.date()), "games": games_df, "players": players_df, "agg": agg, "note": note}


def parse_manual(s):
    out = {}
    for chunk in (s or "").split(","):
        if ":" not in chunk:
            continue
        team, names = chunk.split(":", 1)
        out[team.strip().upper()] = [n.strip() for n in names.split(";") if n.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", default="", help='"BOS:Tatum;Brown,LAL:Doncic" known outs to scratch')
    args = ap.parse_args()
    res = backtest(args.date, parse_manual(args.out))
    print(res["games"].to_string(index=False))
    print("agg:", res["agg"])
    if res.get("note"):
        print("note:", res["note"])
    if len(res["players"]):
        print(res["players"].sort_values("Err", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
