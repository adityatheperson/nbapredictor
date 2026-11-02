"""Compact runtime snapshot for the app, small enough to commit and host.
- player_recent.parquet: last 20 games per player (everything the rolling windows read)
- player_career.parquet: career averages and game counts per player
- player_bios.parquet:   bio table used for age/height/draft features
Run: python3 -m src.snapshot   (after build_dataset). Output goes to data/app/.
"""
import pandas as pd
from pathlib import Path
from src.features_rolling import ROLL_COLS
from src.config import BIOS_FILE

SRC = Path("data/processed/training_expansive.parquet")
APP_DIR = Path("data/app")
RECENT_N = 20
CAREER_COLS = ["PTS", "MIN_NUM", "REB", "AST"]


def main():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    cols = sorted(set(["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE"] + ROLL_COLS + CAREER_COLS))
    df = pd.read_parquet(SRC, columns=cols)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    career = df.groupby("PLAYER_ID").agg(
        CAREER_GP=("PTS", "size"),
        **{f"{c}_CAREER_AVG": (c, "mean") for c in CAREER_COLS},
    ).reset_index()
    recent = df.groupby("PLAYER_ID", group_keys=False).tail(RECENT_N)

    recent.to_parquet(APP_DIR / "player_recent.parquet", index=False)
    career.to_parquet(APP_DIR / "player_career.parquet", index=False)
    pd.read_parquet(BIOS_FILE).drop_duplicates("PERSON_ID").to_parquet(
        APP_DIR / "player_bios.parquet", index=False)
    print(f"recent: {len(recent):,} rows ({recent['PLAYER_ID'].nunique():,} players)")
    print(f"career: {len(career):,} players")
    print(f"written to {APP_DIR}/")


if __name__ == "__main__":
    main()
