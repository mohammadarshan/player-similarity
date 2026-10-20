"""Find the most similar players using standardized per-90 features."""
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/player_features.csv")
ID_COLS = ["player_id", "player", "team", "appearances", "minutes"]\

def similar_players(name_part: str, top_n: int = 10):
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    scaled = StandardScaler().fit_transform(df[feature_cols])

    matches = df[df["player"].str.contains(name_part, case=False, na=False)]
    if matches.empty:
        raise SystemExit(f"No player found matching '{name_part}'")
    idx = matches.index[0] # df has a default 0..n-1 index, so this is also the row in `scaled`

    dist = euclidean_distances(scaled[[idx]], scaled)[0]
    result = df[["player", "team", "minutes"]].copy()
    result["distance"] = dist
    result = result.drop(idx).sort_values("distance").head(top_n)
    return df.loc[idx, "player"], matches["player"].tolist(), result

if __name__ == "__main__":
    query = sys.argv[1]
    chosen, all_matches, result = similar_players(query)
    if len(all_matches) > 1:
        print(f"{len(all_matches)} players match '{query}': {all_matches}")
    print(f"\nPlayers most similar to: {chosen}")
    print(result.round(2).to_string(index=False))

