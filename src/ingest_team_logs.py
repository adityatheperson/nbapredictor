"""Fetch team game logs for the same expansive range (opponent/context features).
Run: python3 -m src.ingest_team_logs
"""
import time
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import LeagueGameLog
from src.config import SEASONS, SEASON_TYPES, TEAM_LOG_FILE, DATA_RAW

def fetch_one(season, season_type, retries=4):
    for attempt in range(retries):
        try:
            log = LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
                timeout=60,
            )
            df = log.get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            return df
        except Exception as e:
            print(f"  retry {attempt+1} {season} {season_type}: {str(e)[:150]}")
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()

def main(resume=True):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    done = set()
    existing = None
    if resume and TEAM_LOG_FILE.exists():
        existing = pd.read_parquet(TEAM_LOG_FILE)
        if "SEASON" in existing.columns and "SEASON_TYPE" in existing.columns:
            done = set(zip(existing["SEASON"], existing["SEASON_TYPE"]))
            print(f"Resuming: {len(done)} done, {len(existing)} rows.")
    frames = [existing] if existing is not None and len(existing) else []
    tasks = [(s, t) for s in SEASONS for t in SEASON_TYPES if (s, t) not in done]
    for season, stype in tqdm(tasks):
        df = fetch_one(season, stype)
        print(f"{season} {stype}: {len(df)} rows")
        if len(df):
            frames.append(df)
            pd.concat(frames, ignore_index=True).to_parquet(TEAM_LOG_FILE, index=False)
        time.sleep(0.6)
    if frames:
        full = pd.concat(frames, ignore_index=True)
        full.to_parquet(TEAM_LOG_FILE, index=False)
        print(f"Saved {len(full)} rows -> {TEAM_LOG_FILE}")

if __name__ == "__main__":
    main()
