"""Pipeline progress tracker. Run anytime: python3 -m src.progress
Shows an ASCII progress bar per stage. Live-watch with:
  while true; do clear; python3 -m src.progress; sleep 30; done
"""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import PLAYER_LOG_FILE, TEAM_LOG_FILE, BIOS_FILE, TRAINING_FILE

READY_FILE = TRAINING_FILE.parent / "training_model_ready.parquet"
MODEL_FILE = Path("models/player_pts_lgbm.pkl")
METRICS_FILE = Path("models/metrics.json")
TOTAL_SEASON_TYPES = 114  # 57 seasons x Regular + Playoffs


def bar(frac, width=30):
    frac = max(0.0, min(1.0, frac))
    fill = int(frac * width)
    return f"[{'#' * fill}{'-' * (width - fill)}] {frac * 100:5.1f}%"


def combos_done(path):
    try:
        df = pd.read_parquet(path, columns=["SEASON", "SEASON_TYPE"])
        return df.drop_duplicates().shape[0], len(df)
    except Exception:
        return 0, 0


def get_stages():
    """Returns [(label, frac_0_1, detail_str), ...] for UI reuse."""
    stages = []
    n_combo, n_rows = combos_done(PLAYER_LOG_FILE)
    stages.append(("Player game logs", n_combo / TOTAL_SEASON_TYPES, f"{n_rows:,} games"))
    n_combo_t, n_rows_t = combos_done(TEAM_LOG_FILE)
    stages.append(("Team game logs", n_combo_t / TOTAL_SEASON_TYPES, f"{n_rows_t:,} games"))
    try:
        bout = PLAYER_LOG_FILE
        n_players_target = (pd.read_parquet(bout, columns=["PLAYER_ID"])
                            ["PLAYER_ID"].nunique() if bout.exists() else 5196)
        n_bios = pq.read_metadata(BIOS_FILE).num_rows if BIOS_FILE.exists() else 0
    except Exception:
        n_players_target, n_bios = 5196, 0
    stages.append(("Player bios", n_bios / n_players_target if n_players_target else 0,
                   f"{n_bios:,}/{n_players_target:,} players"))
    for label, path in [("Training table", TRAINING_FILE), ("Feature table", READY_FILE)]:
        try:
            n = pq.read_metadata(path).num_rows
            stages.append((label, 1.0, f"{n:,} rows"))
        except Exception:
            stages.append((label, 0.0, "not built yet"))
    if MODEL_FILE.exists():
        try:
            m = json.loads(METRICS_FILE.read_text())
            stages.append(("Model", 1.0, f"PTS MAE {m.get('player_pts_mae')}"))
        except Exception:
            stages.append(("Model", 1.0, "trained"))
    else:
        stages.append(("Model", 0.0, "not trained yet"))
    return stages


def main():
    print("NBA Predictor — pipeline progress")
    print("=" * 60)
    for i, (label, frac, detail) in enumerate(get_stages(), start=1):
        print(f"{i}. {label:<16} {bar(frac)}  ({detail})")
    print("=" * 60)


if __name__ == "__main__":
    main()
