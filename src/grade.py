"""Grade saved slate predictions against actual results.
Tries NBA box scores first (full player-level grading); falls back to ESPN final
scores (team-level only) when stats.nba.com is unreachable, e.g. on hosted servers.
Usage: python3 -m src.grade [--date YYYY-MM-DD]  (default: today)
"""
import argparse
import re
import pandas as pd
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import LeagueGameLog
from src.today import load_slate, fetch_espn_events, nba_abbrev
from src.config import now

ID_TO_ABBREV = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}


def season_for(date):
    d = pd.to_datetime(date)
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[-2:]}"
    return f"{d.year - 1}-{str(d.year)[-2:]}"


def parse_matchup(m):
    """'LAL @ BOS' -> ('LAL','BOS'); 'BOS vs. LAL' -> ('LAL','BOS')."""
    m = str(m)
    at = re.match(r"([A-Z]{2,3})\s*@\s*([A-Z]{2,3})", m)
    if at:
        return at.group(1), at.group(2)
    vs = re.match(r"([A-Z]{2,3})\s*vs\.\s*([A-Z]{2,3})", m)
    if vs:
        return vs.group(2), vs.group(1)
    return None, None


def actuals_for_date(date):
    """Returns (team_actuals, player_actuals, pair_to_gid) for one day.
    team_actuals: {(game_id, abbrev): pts}; player_actuals: {(game_id, player_id): (name, pts, min)}"""
    date = pd.to_datetime(date)
    dstr = date.strftime("%m/%d/%Y")
    season = season_for(date)
    team, players, pair_to_gid = {}, {}, {}

    def ingest(season_type):
        kwargs = dict(season=season, season_type_all_star=season_type,
                      date_from_nullable=dstr, date_to_nullable=dstr, timeout=60)
        tlog = LeagueGameLog(player_or_team_abbreviation="T", **kwargs).get_data_frames()[0]
        plog = LeagueGameLog(player_or_team_abbreviation="P", **kwargs).get_data_frames()[0]
        for _, r in tlog.iterrows():
            try:
                team[(str(r["GAME_ID"]), ID_TO_ABBREV[int(r["TEAM_ID"])])] = float(r["PTS"])
                a, h = parse_matchup(r.get("MATCHUP", ""))
                if a and h:
                    pair_to_gid[(a, h)] = str(r["GAME_ID"])
            except KeyError:
                continue
        for _, r in plog.iterrows():
            players[(str(r["GAME_ID"]), int(r["PLAYER_ID"]))] = (
                str(r["PLAYER_NAME"]), float(r["PTS"]), str(r.get("MIN", "")))

    ingest("Regular Season")
    try:
        ingest("Playoffs")
    except Exception:
        pass
    return team, players, pair_to_gid


def espn_finals(date):
    """Fallback when NBA Stats is unreachable: {(away, home): (away_pts, home_pts)} for Final games."""
    out = {}
    for e in fetch_espn_events(date):
        try:
            if e["status"]["type"]["description"] != "Final":
                continue
            comp = e["competitions"][0]
            abbrs = {c["homeAway"]: nba_abbrev(c["team"]["abbreviation"]) for c in comp["competitors"]}
            scores = {c["homeAway"]: int(c.get("score", 0)) for c in comp["competitors"]}
            out[(abbrs.get("away"), abbrs.get("home"))] = (scores.get("away"), scores.get("home"))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def grade(date=None):
    """Returns dict with game_rows, player_rows, aggregates. None if no saved slate."""
    date = pd.to_datetime(date).strftime("%Y-%m-%d") if date else now().strftime("%Y-%m-%d")
    slate = load_slate(date)
    if slate is None:
        return None
    note = None
    try:
        team_act, player_act, pair_to_gid = actuals_for_date(date)
    except Exception:
        team_act, player_act, pair_to_gid = {}, {}, {}
        note = ("Box-score feed unreachable — graded on final scores only "
                "(no player-level comparison).")
    espn = espn_finals(date) if not team_act else {}
    game_rows, player_rows = [], []
    for g in slate["games"]:
        away, home = g["away"], g["home"]
        gid = pair_to_gid.get((away, home)) or g.get("game_id")
        a_act = team_act.get((gid, away))
        h_act = team_act.get((gid, home))
        if a_act is None and (away, home) in espn:
            a_act, h_act = espn[(away, home)]
            a_act, h_act = float(a_act), float(h_act)
        final = a_act is not None and h_act is not None
        actual_winner = (away if a_act is not None and h_act is not None and a_act > h_act
                         else home if final else None)
        row = {"Away": away, "Home": home,
               "Pred": f"{g.get('totals', {}).get(away, '?')}-{g.get('totals', {}).get(home, '?')}",
               "Actual": f"{a_act:.0f}-{h_act:.0f}" if final else "—",
               "Pred winner": g.get("winner", "—"),
               "Actual winner": actual_winner or "—",
               "Call": ("✅" if g.get("winner") == actual_winner else "❌") if final and g.get("winner") else "—"}
        if final and g.get("totals"):
            row["Away err"] = round(abs(g["totals"][away] - a_act), 1)
            row["Home err"] = round(abs(g["totals"][home] - h_act), 1)
        game_rows.append(row)
        for t in (away, home):
            for pl in g.get("players", {}).get(t, []):
                key = (gid, int(pl["player_id"]))
                if key in player_act:
                    _, pts, mins = player_act[key]
                    player_rows.append({
                        "Game": f"{away}@{home}", "Team": t, "Player": pl["player"],
                        "Pred": pl["pred_pts"], "Actual": pts,
                        "Err": round(abs(pl["pred_pts"] - pts), 1), "Min": mins,
                    })
    games_df = pd.DataFrame(game_rows)
    players_df = pd.DataFrame(player_rows)
    agg = {}
    if len(players_df):
        agg["player_pts_mae"] = round(float(players_df["Err"].mean()), 2)
        agg["n_players"] = len(players_df)
    finals = games_df[games_df["Call"].isin(["✅", "❌"])]
    if len(finals):
        agg["winner_accuracy"] = f"{(finals['Call'] == '✅').mean() * 100:.0f}% ({(finals['Call'] == '✅').sum()}/{len(finals)})"
        errs = pd.concat([finals.get("Away err"), finals.get("Home err")]).dropna()
        if len(errs):
            agg["team_total_mae"] = round(float(errs.mean()), 2)
    return {"date": date, "games": games_df, "players": players_df, "agg": agg, "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    out = grade(args.date)
    if out is None:
        print("No saved slate for that date — predict the slate first (it auto-saves).")
        return
    print(f"=== {out['date']} ===")
    print(out["games"].to_string(index=False))
    print("agg:", out["agg"])
    if len(out["players"]):
        print(out["players"].sort_values("Err", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
