"""Daily updater: fetch last 7 days of player+team logs, dedupe, append, rebuild.
Run: python3 -m src.update_daily
"""
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import LeagueGameLog
from src.config import PLAYER_LOG_FILE, TEAM_LOG_FILE

def fetch_range(date_from, date_to, team_abbrev):
    log = LeagueGameLog(
        season="2026-27",
        season_type_all_star="Regular Season",
        player_or_team_abbreviation=team_abbrev,
        date_from_nullable=date_from,
        date_to_nullable=date_to,
        timeout=60,
    )
    return log.get_data_frames()[0]

def upsert(path, new_df, keys):
    keys = list(keys)
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[k for k in keys if k in combined.columns], keep="last")
        combined.to_parquet(path, index=False)
        print(f"{path.name}: {len(old)} -> {len(combined)} (+{len(combined)-len(old)})")
    else:
        new_df.to_parquet(path, index=False)
        print(f"{path.name}: created {len(new_df)} rows")

def main():
    to = datetime.now().date()
    frm = to - timedelta(days=7)
    dfrom, dto = frm.strftime("%m/%d/%Y"), to.strftime("%m/%d/%Y")
    print(f"Updating {dfrom}..{dto}")
    for abbrev, path, keys in [("P", PLAYER_LOG_FILE, ["GAME_ID", "PLAYER_ID"]),
                               ("T", TEAM_LOG_FILE, ["GAME_ID", "TEAM_ID"])]:
        try:
            df = fetch_range(dfrom, dto, abbrev)
            df["SEASON"] = "2026-27"
            df["SEASON_TYPE"] = "Regular Season"
            print(f"Fetched {abbrev}: {len(df)} rows")
            if len(df):
                upsert(path, df, keys)
        except Exception as e:
            print(f"FAILED {abbrev}: {e}")
    print("Now run: python3 -m src.build_dataset && python3 -m src.features_rolling")

if __name__ == "__main__":
    main()
