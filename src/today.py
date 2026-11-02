"""Predict every game on a given day (default: today).
Usage: python3 -m src.today [--date YYYY-MM-DD]
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from src.predict import predict_matchup
from src.config import now

SLATE_DIR = Path("data/processed/slates")

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SCORE_CACHE = Path("data/raw/live_scores.json")
SCORE_TTL_SECONDS = 60

# ESPN scoreboard abbreviations that differ from NBA Stats abbreviations
ESPN_TO_NBA = {"NY": "NYK", "WSH": "WAS", "NO": "NOP", "UTAH": "UTA", "SA": "SAS", "GS": "GSW"}


def nba_abbrev(abbr):
    return ESPN_TO_NBA.get(abbr, abbr)


def fetch_espn_events(date=None):
    """Full ESPN scoreboard events for a day (schedule + live scores). Cached 60s."""
    date = pd.to_datetime(date).strftime("%Y%m%d") if date else now().strftime("%Y%m%d")
    if SCORE_CACHE.exists() and (time.time() - SCORE_CACHE.stat().st_mtime) < SCORE_TTL_SECONDS:
        try:
            cached = json.loads(SCORE_CACHE.read_text())
            if cached.get("date") == date:
                return cached["events"]
        except Exception:
            pass
    try:
        events = requests.get(f"{ESPN_SCOREBOARD}?dates={date}", timeout=20).json().get("events", [])
    except Exception as ex:
        print(f"Scoreboard unavailable ({ex})")
        return []
    SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_CACHE.write_text(json.dumps({"date": date, "events": events}))
    return events


def tip_text(event):
    """Human tip time, e.g. '8:30 PM ET'. Prefers ESPN's own label."""
    short = (event.get("status", {}).get("type", {}).get("shortDetail") or "").strip()
    if short and "ET" in short:
        return short
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        et = dt.astimezone(ZoneInfo("America/New_York"))
        return et.strftime("%-I:%M %p ET")
    except Exception:
        return ""


def fetch_live_scores(date=None):
    """Live scoreboard keyed by (away, home): {state, clock, period, scores}."""
    games = {}
    for e in fetch_espn_events(date):
        try:
            comp = e["competitions"][0]
            entry = {"state": e["status"]["type"]["description"],
                     "clock": e["status"].get("displayClock", ""),
                     "period": e["status"].get("period", 0),
                     "scores": {}}
            for c in comp["competitors"]:
                try:
                    entry["scores"][c["homeAway"]] = int(c.get("score", 0))
                except (TypeError, ValueError):
                    entry["scores"][c["homeAway"]] = 0
            abbrs = {c["homeAway"]: nba_abbrev(c["team"]["abbreviation"]) for c in comp["competitors"]}
            games[(abbrs.get("away"), abbrs.get("home"))] = entry
        except (KeyError, IndexError):
            continue
    return games


def get_todays_games(date=None):
    live = fetch_live_scores(date)
    games = []
    for e in fetch_espn_events(date):
        try:
            comp = e["competitions"][0]
            abbrs = {c["homeAway"]: nba_abbrev(c["team"]["abbreviation"]) for c in comp["competitors"]}
            away, home = abbrs.get("away"), abbrs.get("home")
            if not away or not home:
                continue
            lv = live.get((away, home), {})
            games.append({
                "game_id": str(e.get("id", f"{away}@{home}")),
                "home": home,
                "away": away,
                "status": tip_text(e),
                "arena": comp.get("venue", {}).get("fullName", ""),
                "live_state": lv.get("state", "Scheduled"),
                "clock": lv.get("clock", ""),
                "period": lv.get("period", 0),
                "live_away": lv.get("scores", {}).get("away"),
                "live_home": lv.get("scores", {}).get("home"),
            })
        except (KeyError, IndexError):
            continue
    return games


def predict_all_games(date=None, progress_cb=None, save=True):
    games = get_todays_games(date)
    results = []
    for i, g in enumerate(games):
        if progress_cb:
            progress_cb(i, len(games), g)
        try:
            out = predict_matchup(g["away"], g["home"], home_team=g["home"])
            results.append({"game": g, "prediction": out, "error": None})
        except Exception as e:
            results.append({"game": g, "prediction": None, "error": str(e)[:200]})
    if save:
        save_slate(date, results)
    return results


def save_slate(date, results):
    date = pd.to_datetime(date).strftime("%Y-%m-%d") if date else now().strftime("%Y-%m-%d")
    SLATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"date": date, "games": []}
    for r in results:
        g = r["game"]
        entry = {"away": g["away"], "home": g["home"], "game_id": g["game_id"],
                 "status": g["status"], "error": r["error"]}
        if r["prediction"] is not None:
            p = r["prediction"]
            entry.update({
                "totals": p["totals"], "winner": p["winner"], "win_prob": p["win_prob"],
                "out": p.get("out", {}), "questionable": p.get("questionable", {}),
                "players": {t: [{"player_id": int(row["player_id"]),
                                 "player": row["player"],
                                 "pred_pts": round(float(row["pred_pts"]), 2)}
                                for _, row in p["players"][t].iterrows()]
                            for t in (p["team_a"], p["team_b"])},
            })
        data["games"].append(entry)
    (SLATE_DIR / f"slate_{date}.json").write_text(json.dumps(data))
    return data


def load_slate(date):
    date = pd.to_datetime(date).strftime("%Y-%m-%d")
    path = SLATE_DIR / f"slate_{date}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summary_table(results):
    rows = []
    for r in results:
        g = r["game"]
        live = "vs"
        if g.get("live_away") is not None and g.get("live_state") != "Scheduled":
            live = f"{g['live_away']}-{g['live_home']} {game_clock(g)}".strip()
        base = {"Away": g["away"], "Home": g["home"], "Tip": g["status"],
                "Live": live}
        if r["prediction"] is None:
            rows.append({**base, "Away total": None, "Home total": None,
                         "Winner": None, "P(winner)": None, "Error": r["error"]})
            continue
        p = r["prediction"]
        row = {**base,
               "Away total": p["totals"][g["away"]],
               "Home total": p["totals"][g["home"]],
               "Winner": p["winner"],
               "P(winner)": p["win_prob"][p["winner"]]}
        if g.get("live_state") == "Final" and g.get("live_away") is not None:
            row["Away err"] = round(abs(p["totals"][g["away"]] - g["live_away"]), 1)
            row["Home err"] = round(abs(p["totals"][g["home"]] - g["live_home"]), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def game_clock(g):
    if g.get("live_state") == "Final":
        return "(Final)"
    if g.get("live_state") == "Scheduled":
        return ""
    q = f"Q{g['period']}" if g.get("period") else ""
    return f"({g.get('live_state', '')} {q} {g.get('clock', '')})".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    results = predict_all_games(args.date, progress_cb=lambda i, n, g: print(f"[{i+1}/{n}] {g['away']} @ {g['home']}..."))
    print(summary_table(results).to_string(index=False))


if __name__ == "__main__":
    main()
