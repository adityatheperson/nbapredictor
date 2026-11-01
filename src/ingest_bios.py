"""Fetch player bios (off-court data: age source, height, weight, draft, school, country, position).
Uses CommonAllPlayers (all historical) + CommonPlayerInfo per player.
Run: python3 -m src.ingest_bios
"""
import time
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import CommonAllPlayers, CommonPlayerInfo
from src.config import BIOS_FILE, DATA_RAW

def main(limit=None, resume=True):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print("Fetching CommonAllPlayers...")
    all_players = CommonAllPlayers(timeout=60).get_data_frames()[0]
    print(f"Found {len(all_players)} all-time player entries, cols: {list(all_players.columns)}")
    # Dedupe to unique PERSON_ID
    ids = all_players[["PERSON_ID", "DISPLAY_FIRST_LAST"]].drop_duplicates()

    done_ids = set()
    existing = None
    if resume and BIOS_FILE.exists():
        existing = pd.read_parquet(BIOS_FILE)
        if "PERSON_ID" in existing.columns:
            done_ids = set(existing["PERSON_ID"].tolist())
            print(f"Resuming: {len(done_ids)} bios already fetched.")

    ids = ids[~ids["PERSON_ID"].isin(done_ids)]
    if limit:
        ids = ids.head(limit)
    print(f"Fetching {len(ids)} player bios...")
    rows = []
    if existing is not None and len(existing):
        rows.append(existing)
    for _, r in tqdm(list(ids.iterrows()), total=len(ids)):
        pid = int(r["PERSON_ID"])
        for attempt in range(3):
            try:
                info = CommonPlayerInfo(player_id=pid, timeout=30).get_data_frames()[0]
                if len(info):
                    rows.append(info)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {pid} {r['DISPLAY_FIRST_LAST']}: {str(e)[:120]}")
                time.sleep(1.0 * (attempt + 1))
        time.sleep(0.4)
        # checkpoint every 200
        if len(rows) % 200 == 0 and len(rows) > 0:
            pd.concat(rows, ignore_index=True).to_parquet(BIOS_FILE, index=False)
    if rows:
        full = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["PERSON_ID"])
        full.to_parquet(BIOS_FILE, index=False)
        print(f"Saved {len(full)} bios -> {BIOS_FILE}")
        print("Columns:", list(full.columns))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None,
                    help="Fetch at most N more bios (for chunked runs with fresh processes).")
    args = ap.parse_args()
    main(limit=args.max)
