"""Rolling pattern features: turns raw history into model-ready signals.
Run: python3 -m src.features_rolling
Reads data/processed/training_expansive.parquet, writes training_model_ready.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path("data/processed/training_expansive.parquet")
DST = Path("data/processed/training_model_ready.parquet")

ROLL_COLS = ["MIN_NUM", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
             "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
             "PLUS_MINUS", "EFG_PCT", "TS_PCT", "PTS_PER_MIN", "FANTASY_PTS"]
WINDOWS = [3, 5, 10, 20]

def add_rolling(g):
    g = g.sort_values("GAME_DATE")
    for w in WINDOWS:
        shifted = g[ROLL_COLS].shift(1)  # no leakage: only prior games
        roll = shifted.rolling(w, min_periods=1)
        mean = roll.mean()
        std = roll.std().fillna(0)
        for c in ROLL_COLS:
            g[f"{c}_AVG{w}"] = mean[c].values
            g[f"{c}_STD{w}"] = std[c].values
        # games played in window (injury/miss signal)
        g[f"GP_LAST{w}"] = shifted["PTS"].notna().rolling(w, min_periods=1).sum().values
    # career-to-date (expanding, shifted)
    exp = g[ROLL_COLS].shift(1).expanding(min_periods=1).mean()
    for c in ["PTS", "MIN_NUM", "REB", "AST"]:
        g[f"{c}_CAREER_AVG"] = exp[c].values
    # streak / form: last game vs avg10
    g["PTS_L1_vs_AVG10"] = g["PTS"].shift(1) - g["PTS_AVG10"]
    # age curve interaction
    if "AGE_AT_GAME" in g.columns:
        g["AGE_SQ"] = g["AGE_AT_GAME"] ** 2
        g["PTS_AVG10_x_AGE"] = g["PTS_AVG10"] * g["AGE_AT_GAME"]
    return g

def main():
    if not SRC.exists():
        print(f"No {SRC} yet — full pull still running. Skipping.")
        return
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df)} rows.")
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    df["CAREER_GP"] = df.groupby("PLAYER_ID").cumcount()
    # experience proxy if bios missing
    if "SEASON_EXP_AT_FETCH" not in df.columns:
        df["SEASON_EXP_AT_FETCH"] = np.nan
    parts = []
    for _, g in df.groupby("PLAYER_ID", sort=False):
        parts.append(add_rolling(g.copy()))
    df = pd.concat(parts, ignore_index=True).copy()
    df = df.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)
    # opponent rest / home already in base; add month one-hots downstream in train
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols -> {DST}")

if __name__ == "__main__":
    main()
