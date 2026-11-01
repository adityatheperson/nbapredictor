"""Shared config: go as far back as NBA API allows, keep every stat column."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Expansive history: LeagueGameLog works back to at least 1970-71.
# Generate 1970-71 .. 2026-27 inclusive.
def season_strings(start_year=1970, end_year=2026):
    return [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year + 1)]

SEASONS = season_strings(1970, 2026)

SEASON_TYPES = ["Regular Season", "Playoffs"]

# No column subsetting: keep everything the API returns.
# These are documented for reference, but code keeps all columns dynamically.
PLAYER_LOG_FILE = DATA_RAW / "player_game_logs_all.parquet"
TEAM_LOG_FILE = DATA_RAW / "team_game_logs_all.parquet"
BIOS_FILE = DATA_RAW / "player_bios.parquet"
TRAINING_FILE = DATA_PROCESSED / "training_expansive.parquet"
