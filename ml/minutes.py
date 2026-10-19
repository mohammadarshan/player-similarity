"""Compute minutes played per player for one match from StatsBomb events."""
import pandas as pd
from pathlib import Path
from ml.teams import clean_team_names

def match_minutes(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values("index")
    match_end = int(events["minute"].max())
    spans = {}

    # Starters are on the pitch from minute 0
    for _, ev in events[events["type"] == "Starting XI"].iterrows():
        for p in ev["tactics"]["lineup"]:
            spans[p["player"]["id"]] = {
                "player": p["player"]["name"],
                "team": ev["team"],
                "start": 0,
                "end": match_end,
            }

    # Substitutions cut one span short and open a new one
    for _, ev in events[events["type"] == "Substitution"].iterrows():
        minute = int(ev["minute"])
        off_id = int(ev["player_id"])
        on_id = int(ev["substitution_replacement_id"])
        if off_id in spans:
            spans[off_id]["end"] = minute
        spans[on_id] = {
            "player": ev["substitution_replacement"],
            "team": ev["team"],
            "start": minute,
            "end": match_end,
        }

    df = pd.DataFrame.from_dict(spans, orient="index")
    df.index.name = "player_id"
    df = df.reset_index()
    df["minutes"] = df["end"] - df["start"]
    df["match_id"] = int(events["match_id"].iloc[0])
    df["team"] = clean_team_names(df["team"])   
    return df[["match_id", "player_id", "player", "team", "minutes"]]

if __name__ == "__main__":
    path = sorted(Path("data/raw").glob("*.pkl"))[0]
    result = match_minutes(pd.read_pickle(path))
    print(result.sort_values(["team", "minutes"]).to_string(index=False))