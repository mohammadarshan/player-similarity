"""Assign each player a main position and a position group from event data."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def position_group(pos: str) -> str:
    if pos == "Goalkeeper":
        return "GK"
    if "Back" in pos:  # includes Wing Backs
        return "DEF"
    if "Wing" in pos or "Forward" in pos or "Striker" in pos:
        return "ATT"
    if "Midfield" in pos:
        return "MID"
    return "OTHER"


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("*.pkl"))
    parts = []
    for i, path in enumerate(files, 1):
        ev = pd.read_pickle(path)[["player_id", "position"]].dropna()
        ev["player_id"] = ev["player_id"].astype(int)
        parts.append(ev.groupby(["player_id", "position"]).size().rename("n").reset_index())
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)} matches")

    counts = pd.concat(parts).groupby(["player_id", "position"])["n"].sum().reset_index()
    main = (
        counts.sort_values("n", ascending=False)
        .drop_duplicates("player_id")
        .rename(columns={"position": "main_position"})[["player_id", "main_position"]]
    )
    main["position_group"] = main["main_position"].map(position_group)
    main.to_csv(OUT_DIR / "player_positions.csv", index=False)

    feats = pd.read_csv(OUT_DIR / "player_features.csv")
    merged = feats.merge(main, on="player_id", how="left")
    print("\nPosition groups among the kept players:")
    print(merged["position_group"].value_counts(dropna=False).to_string())
    spot = merged[merged["player_id"].isin([5503, 5203, 20005, 4691])]
    print("\nSpot check (Messi, Busquets, Alderweireld, Bravo):")
    print(spot[["player", "main_position", "position_group"]].to_string(index=False))