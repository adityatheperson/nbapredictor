"""Fetch EVERY player game log back to 1970-71, Regular Season + Playoffs.
Keeps all columns returned by the API (no stat left out).
Run: python3 -m src.ingest_player_logs
"""
import time
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import LeagueGameLog
from src.config import SEASONS, SEASON_TYPES, PLAYER_LOG_FILE, DATA_RAW

def fetch_one(season, season_type, retries=4, sleep=1.0):
    for attempt in range(retries):
        try:
            log = LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                timeout=60,
            )
            df = log.get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            return df
        except Exception as e:
            print(f"  retry {attempt+1}/{retries} {season} {season_type}: {str(e)[:150]}")
            time.sleep(sleep * (attempt + 1) * 2)
    return pd.DataFrame()

def main(resume=True):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    done = set()
    existing = None
    if resume and PLAYER_LOG_FILE.exists():
        existing = pd.read_parquet(PLAYER_LOG_FILE)
        if "SEASON" in existing.columns and "SEASON_TYPE" in existing.columns:
            done = set(zip(existing["SEASON"], existing["SEASON_TYPE"]))
            print(f"Resuming: {len(done)} season/types already done, {len(existing)} rows.")
    all_frames = [existing] if existing is not None and len(existing) else []
    tasks = [(s, t) for s in SEASONS for t in SEASON_TYPES if (s, t) not in done]
    print(f"Fetching {len(tasks)} season/types...")
    for season, stype in tqdm(tasks):
        df = fetch_one(season, stype)
        print(f"{season} {stype}: {len(df)} rows")
        if len(df):
            all_frames.append(df)
            # checkpoint every season
            pd.concat(all_frames, ignore_index=True).to_parquet(PLAYER_LOG_FILE, index=False)
        time.sleep(0.6)  # rate-limit courtesy
    if all_frames:
        full = pd.concat(all_frames, ignore_index=True)
        full.to_parquet(PLAYER_LOG_FILE, index=False)
        print(f"Saved {len(full)} rows x {len(full.columns)} cols -> {PLAYER_LOG_FILE}")
        print("Columns:", list(full.columns))
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()
