"""Grade saved slate predictions against actual box scores.
Usage: python3 -m src.grade [--date YYYY-MM-DD]  (default: today)
"""
import argparse
import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
from src.today import load_slate, ID_TO_ABBREV


def season_for(date):
    d = pd.to_datetime(date)
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[-2:]}"
    return f"{d.year - 1}-{str(d.year)[-2:]}"


def actuals_for_date(date):
    """Returns (team_actuals, player_actuals) for one day.
    team_actuals: {(game_id, abbrev): pts}; player_actuals: {(game_id, player_id): (name, pts, min)}"""
    date = pd.to_datetime(date)
    dstr = date.strftime("%m/%d/%Y")
    season = season_for(date)
    kwargs = dict(season=season, season_type_all_star="Regular Season",
                  date_from_nullable=dstr, date_to_nullable=dstr, timeout=60)
    tlog = LeagueGameLog(player_or_team_abbreviation="T", **kwargs).get_data_frames()[0]
    plog = LeagueGameLog(player_or_team_abbreviation="P", **kwargs).get_data_frames()[0]
    team, players = {}, {}
    for _, r in tlog.iterrows():
        try:
            team[(str(r["GAME_ID"]), ID_TO_ABBREV[int(r["TEAM_ID"])])] = float(r["PTS"])
        except KeyError:
            continue
    for _, r in plog.iterrows():
        players[(str(r["GAME_ID"]), int(r["PLAYER_ID"]))] = (
            str(r["PLAYER_NAME"]), float(r["PTS"]), str(r.get("MIN", "")))
    # playoffs that same day
    try:
        kwargs["season_type_all_star"] = "Playoffs"
        tlog2 = LeagueGameLog(player_or_team_abbreviation="T", **kwargs).get_data_frames()[0]
        plog2 = LeagueGameLog(player_or_team_abbreviation="P", **kwargs).get_data_frames()[0]
        for _, r in tlog2.iterrows():
            try:
                team[(str(r["GAME_ID"]), ID_TO_ABBREV[int(r["TEAM_ID"])])] = float(r["PTS"])
            except KeyError:
                continue
        for _, r in plog2.iterrows():
            players[(str(r["GAME_ID"]), int(r["PLAYER_ID"]))] = (
                str(r["PLAYER_NAME"]), float(r["PTS"]), str(r.get("MIN", "")))
    except Exception:
        pass
    return team, players


def game_id_for(slate_game, team_actuals_keys=None):
    return slate_game.get("game_id")


def grade(date=None):
    """Returns dict with game_rows, player_rows, aggregates. None if no saved slate."""
    date = pd.to_datetime(date).strftime("%Y-%m-%d") if date else pd.Timestamp.now().strftime("%Y-%m-%d")
    slate = load_slate(date)
    if slate is None:
        return None
    team_act, player_act = actuals_for_date(date)
    game_rows, player_rows = [], []
    for g in slate["games"]:
        gid, away, home = g.get("game_id"), g["away"], g["home"]
        a_act = team_act.get((gid, away))
        h_act = team_act.get((gid, home))
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
    return {"date": date, "games": games_df, "players": players_df, "agg": agg}


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
