"""Build expansive training dataset: merge game logs + bios + context.
- Keeps EVERY raw stat column (no dropping).
- Adds off-court: age at game (exact), experience, draft pick/round/year, height_in, weight, position, school, country.
- Adds context: home/away, opponent abbrev, rest days, back-to-back, season stage.
- Adds derived efficiency (extra signal, raw kept): eFG%, TS%, AST_TOV, REB_RATE proxies, double-double, 30pt/20pt flags, minutes buckets.
- Sorts by player + date for rolling features downstream.
Run: python3 -m src.build_dataset
"""
import re
import numpy as np
import pandas as pd
from src.config import PLAYER_LOG_FILE, BIOS_FILE, TRAINING_FILE

def parse_height(h):
    # "6-7" -> 79
    if pd.isna(h):
        return np.nan
    m = re.match(r"(\d+)-(\d+)", str(h))
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    try:
        return float(h)
    except Exception:
        return np.nan

def parse_min(m):
    # "34:12" or 34.0 or 34
    if pd.isna(m):
        return np.nan
    s = str(m)
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return int(mm) + int(ss) / 60.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def main():
    print(f"Loading {PLAYER_LOG_FILE}...")
    logs = pd.read_parquet(PLAYER_LOG_FILE)
    print(f"Logs: {len(logs)} rows, cols={len(logs.columns)}")
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["MIN_NUM"] = logs["MIN"].apply(parse_min)

    # --- Context: home/away, opponent ---
    # MATCHUP examples: "LAL vs. BOS" (home) / "LAL @ BOS" (away)
    logs["IS_HOME"] = logs["MATCHUP"].str.contains("vs\\.", na=False).astype(int)
    logs["OPPONENT"] = logs["MATCHUP"].str.extract(r"(?:vs\.|@)\s*([A-Z]{2,3})")

    # --- Rest days + back-to-back per player ---
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"])
    logs["PREV_GAME_DATE"] = logs.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    logs["REST_DAYS"] = (logs["GAME_DATE"] - logs["PREV_GAME_DATE"]).dt.days - 1
    logs["REST_DAYS"] = logs["REST_DAYS"].fillna(7).clip(lower=0, upper=14)
    logs["BACK_TO_BACK"] = (logs["REST_DAYS"] == 0).astype(int)
    logs["GAMES_LAST_7D"] = logs.groupby("PLAYER_ID")["GAME_DATE"].transform(
        lambda s: s.rolling("7D", on=None).count() if False else np.nan
    ) if False else np.nan  # placeholder, computed properly below

    # Proper trailing count using expanding per player with date diff
    def trailing_counts(g):
        dates = g["GAME_DATE"].values
        out = []
        for i, d in enumerate(dates):
            out.append(int(((dates[:i] >= d - np.timedelta64(7, "D")).sum())))
        return pd.Series(out, index=g.index)
    logs["GAMES_LAST_7D"] = logs.groupby("PLAYER_ID", group_keys=False).apply(trailing_counts, include_groups=False)

    # --- Derived efficiency (additive, raw never dropped) ---
    fga = logs["FGA"].fillna(0); fgm = logs["FGM"].fillna(0)
    fg3a = logs["FG3A"].fillna(0); fta = logs["FTA"].fillna(0); ftm = logs["FTM"].fillna(0)
    pts = logs["PTS"].fillna(0); tov = logs["TOV"].fillna(0)
    ast = logs["AST"].fillna(0); reb = logs["REB"].fillna(0)
    logs["EFG_PCT"] = np.where(fga > 0, (fgm + 0.5 * logs["FG3M"].fillna(0)) / fga, np.nan)
    logs["TS_PCT"] = np.where((fga + 0.44 * fta) > 0, pts / (2 * (fga + 0.44 * fta)), np.nan)
    logs["AST_TOV"] = np.where(tov > 0, ast / tov, ast)
    logs["PTS_PER_MIN"] = np.where(logs["MIN_NUM"] > 0, pts / logs["MIN_NUM"], 0.0)
    logs["REB_PER_MIN"] = np.where(logs["MIN_NUM"] > 0, reb / logs["MIN_NUM"], 0.0)
    logs["AST_PER_MIN"] = np.where(logs["MIN_NUM"] > 0, ast / logs["MIN_NUM"], 0.0)
    logs["DOUBLE_DOUBLE"] = (((logs[["PTS", "REB", "AST", "STL", "BLK"]].fillna(0) >= 10).sum(axis=1)) >= 2).astype(int)
    logs["PTS_30"] = (pts >= 30).astype(int)
    logs["PTS_20"] = (pts >= 20).astype(int)
    logs["DID_NOT_PLAY"] = ((logs["MIN_NUM"].fillna(0) == 0) & (pts == 0)).astype(int)

    # Season stage / month / weekday
    logs["GAME_MONTH"] = logs["GAME_DATE"].dt.month
    logs["GAME_WEEKDAY"] = logs["GAME_DATE"].dt.day_of_week
    logs["SEASON_YEAR"] = logs["GAME_DATE"].dt.year

    # --- Merge bios (off-court) ---
    try:
        bios = pd.read_parquet(BIOS_FILE)
        print(f"Bios: {len(bios)} rows, cols: {list(bios.columns)}")
        keep = ["PERSON_ID", "BIRTHDATE", "HEIGHT", "WEIGHT", "POSITION",
                "JERSEY", "SCHOOL", "COUNTRY", "DRAFT_YEAR", "DRAFT_ROUND",
                "DRAFT_NUMBER", "SEASON_EXP", "DISPLAY_FIRST_LAST"]
        keep = [c for c in keep if c in bios.columns]
        bios = bios[keep].drop_duplicates(subset=["PERSON_ID"]).rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON_EXP": "SEASON_EXP_AT_FETCH"})
        logs = logs.merge(bios, on="PLAYER_ID", how="left")

        # Exact age at game
        logs["BIRTHDATE"] = pd.to_datetime(logs["BIRTHDATE"], errors="coerce")
        logs["AGE_AT_GAME"] = (logs["GAME_DATE"] - logs["BIRTHDATE"]).dt.days / 365.25
        logs["HEIGHT_IN"] = logs["HEIGHT"].apply(parse_height) if "HEIGHT" in logs.columns else np.nan
        logs["WEIGHT_NUM"] = pd.to_numeric(logs["WEIGHT"], errors="coerce") if "WEIGHT" in logs.columns else np.nan
        # Draft pick numeric (undrafted -> 61)
        logs["DRAFT_NUMBER_NUM"] = pd.to_numeric(logs["DRAFT_NUMBER"], errors="coerce").fillna(61)
        logs["IS_UNDRAFTED"] = (logs["DRAFT_NUMBER_NUM"] == 61).astype(int)
        logs["DRAFT_ROUND_NUM"] = pd.to_numeric(logs.get("DRAFT_ROUND"), errors="coerce")
    except FileNotFoundError:
        print("WARNING: bios not found, skipping off-court merge. Run ingest_bios first.")
        logs["AGE_AT_GAME"] = np.nan

    # Final sort
    logs = logs.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)
    TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(TRAINING_FILE, index=False)
    print(f"Saved {len(logs)} rows x {len(logs.columns)} cols -> {TRAINING_FILE}")
    print("Columns:", list(logs.columns))

if __name__ == "__main__":
    main()
