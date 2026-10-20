"""Find the most similar players using standardized per-90 features."""
import sys
from pathlib import Path
import pandas as pd
import re
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/player_features.csv")
ID_COLS = ["player_id", "player", "team", "appearances", "minutes"]

def find_player(df: pd.DataFrame, query: str) -> str:
    """Return the row index of exactly one player, or exit with the candidates."""
    if query.isdigit():
        hit = df[df["player_id"] == int(query)]
    else:
        hit = df[df["player"].str.contains(query, case=False, na=False, regex=False)]
        if len(hit) > 1:
            whole_word = df[
                df["player"].str.contains(rf"\b{re.escape(query)}\b", case=False, na=False)
            ]
            if len(whole_word) == 1:
                hit = whole_word

    if hit.empty:
        raise SystemExit(f"No player found matching '{query}'")
    if len(hit) > 1:
        print(f"'{query}' matches {len(hit)} players. Re-run with a player_id:\n")
        print(hit[["player_id", "player", "team"]].head(15).to_string(index=False))
        raise SystemExit()
    return hit.index[0]

def similar_players(query: str, top_n: int = 10):
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    scaled = StandardScaler().fit_transform(df[feature_cols])

    idx = find_player(df, query)
    dist = euclidean_distances(scaled[[idx]], scaled)[0]
    result = df[["player", "team", "minutes"]].copy()
    result["distance"] = dist
    result = result.drop(idx).sort_values("distance").head(top_n)
    return df.loc[idx, "player"], result

if __name__ == "__main__":
    chosen, result = similar_players(sys.argv[1])
    print(f"\nPlayers most similar to: {chosen}")
    print(result.round(2).to_string(index=False))