"""Build one row per player: season per-90 features, minutes >= threshold."""
from pathlib import Path

import pandas as pd

from ml.features import match_features

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
MIN_MINUTES = 450
NOT_RATES = ["player_id", "sum_x", "sum_y", "n_loc"]


def build_totals() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.pkl"))
    frames = []
    for i, path in enumerate(files, 1):
        frames.append(match_features(pd.read_pickle(path)))
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)} matches")
    return pd.concat(frames, ignore_index=True).groupby("player_id").sum().reset_index()


if __name__ == "__main__":
    totals = build_totals()

    mm = pd.read_csv(OUT_DIR / "match_minutes.csv")
    info = mm.groupby("player_id").agg(
        player=("player", "first"),
        appearances=("match_id", "nunique"),
        minutes=("minutes", "sum"),
    )
    # Club where the player spent the most minutes
    main_team = (
        mm.groupby(["player_id", "team"])["minutes"].sum().reset_index()
        .sort_values("minutes", ascending=False)
        .drop_duplicates("player_id")
        .set_index("player_id")["team"]
    )
    info["team"] = main_team
    df = info.reset_index().merge(totals, on="player_id", how="left").fillna(0)

    print(f"\nLeague-wide pass completion: {df['passes_completed'].sum() / df['passes'].sum():.1%}")
    print("Top 5 by total non-penalty xG:")
    print(df.sort_values("xg", ascending=False)[["player", "team", "xg"]].head(5).round(1).to_string(index=False))
    print("Top 5 by assists:")
    print(df.sort_values("assists", ascending=False)[["player", "team", "assists"]].head(5).to_string(index=False))

    df = df[(df["minutes"] >= MIN_MINUTES) & (df["n_loc"] > 0)].copy()
    count_cols = [c for c in totals.columns if c not in NOT_RATES]
    for c in count_cols:
        df[f"{c}_per90"] = df[c] / df["minutes"] * 90
    df["avg_x"] = df["sum_x"] / df["n_loc"]
    df["avg_y"] = df["sum_y"] / df["n_loc"]

    keep = ["player_id", "player", "team", "appearances", "minutes"]
    keep += [f"{c}_per90" for c in count_cols] + ["avg_x", "avg_y"]
    df = df[keep].sort_values("minutes", ascending=False)
    df.to_csv(OUT_DIR / "player_features.csv", index=False)

    print(f"\nPlayers kept (>= {MIN_MINUTES} min): {len(df)}")
    print(f"Feature columns: {len(keep) - 5}")
    print(df[["player", "minutes", "passes_per90", "progressive_carries_per90", "tackles_per90", "avg_x"]].head(3).round(2).to_string(index=False))