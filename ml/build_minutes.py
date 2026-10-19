"""Build per-match and per-season minutes tables from all raw match files."""
from pathlib import Path
import pandas as pd
from ml.minutes import match_minutes

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

def build_minutes() -> pd.DataFrame:
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    files = sorted(RAW_DIR.glob("*.pkl"))
    frames = []
    for i, path in enumerate(files):
        frames.append(match_minutes(pd.read_pickle(path)))
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)} matches.")
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    per_match = build_minutes()
    per_match.to_csv(OUT_DIR / "match_minutes.csv", index=False)

    season = (
        per_match.groupby("player_id")
        .agg(
            player=("player", "first"),
            team=("team", "nunique"),
            appearances=("match_id", "nunique"),
            minutes=("minutes", "sum"),
        )
        .reset_index()
    )
    season.to_csv(OUT_DIR / "season_minutes.csv", index=False)

    print(f"\nMatch-player rows: {len(per_match)}")
    print(f"Players total: {len(season)}")
    for threshold in (450, 900):
        print(f"Players with >= {threshold} minutes: {(season['minutes'] >= threshold).sum()}")
    print(f"Players who appeared for 2+ teams: {(season['team'] > 1).sum()}")
    print("\nTop 5 by minutes:")
    print(season.sort_values("minutes", ascending=False).head(5).to_string(index=False))