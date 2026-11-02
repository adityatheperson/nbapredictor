"""Injury handling via ESPN's league-wide injury report (refreshed daily).
- fetch_report(): cached to data/raw/injuries_YYYY-MM-DD.json (refetch if >6h old)
- team_injuries(abbrev): [{player, status, category, date, comment}]
  category: 'out' (Out) or 'questionable' (Day-To-Day etc.)
- match_out_players(roster_df, injury_names): fuzzy name matching
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
CACHE_DIR = Path("data/raw")
CACHE_TTL_HOURS = 6

NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name):
    parts = re.sub(r"[.'’\-]", "", str(name).lower()).split()
    parts = [p for p in parts if p not in SUFFIXES]
    return " ".join(parts)


def classify(status):
    s = (status or "").lower()
    if "out" in s and "doubt" not in s:
        return "out"
    return "questionable"


def fetch_report(max_age_hours=CACHE_TTL_HOURS):
    from src.config import now
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = now().strftime("%Y-%m-%d")
    cache = CACHE_DIR / f"injuries_{today}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600:
        return json.loads(cache.read_text())
    r = requests.get(ESPN_URL, timeout=30)
    r.raise_for_status()
    report = r.json()
    cache.write_text(json.dumps(report))
    return report


def team_injuries(team_abbrev, report=None):
    report = report or fetch_report()
    team_abbrev = team_abbrev.upper()
    out = []
    for t in report.get("injuries", []):
        if NAME_TO_ABBREV.get(t.get("displayName", "")) != team_abbrev:
            continue
        for inj in t.get("injuries", []):
            a = inj.get("athlete", {})
            out.append({
                "player": a.get("displayName", ""),
                "status": inj.get("status", ""),
                "category": classify(inj.get("status", "")),
                "date": inj.get("date", ""),
                "comment": inj.get("shortComment", "") or inj.get("longComment", ""),
            })
    return out


def match_out_players(roster_names, injury_names):
    """Fuzzy-match injury names to roster names. Returns set of matched roster names."""
    norm_roster = {normalize_name(n): n for n in roster_names}
    last_only = {}
    for norm, orig in norm_roster.items():
        last = norm.split()[-1] if norm else ""
        last_only.setdefault(last, []).append(orig)
    matched = set()
    for inj in injury_names:
        n = normalize_name(inj)
        if n in norm_roster:
            matched.add(norm_roster[n])
            continue
        last = n.split()[-1] if n else ""
        cands = last_only.get(last, [])
        if len(cands) == 1:
            matched.add(cands[0])
    return matched
