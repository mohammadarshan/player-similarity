"""Count per-player events for one match (StatsBomb events)."""
from pathlib import Path

import numpy as np
import pandas as pd

SET_PIECES = ["Throw-in", "Free Kick", "Goal Kick", "Corner", "Kick Off"]
SHOT_ON_TARGET = ["Goal", "Saved", "Saved to Post"]
TACKLE_WON = ["Won", "Success In Play", "Success Out"]
PROGRESS_YARDS = 10  # a progressive action moves the ball this much closer to goal

# Columns that may be missing from a match with few events
NEEDED = [
    "location", "pass_type", "pass_outcome", "pass_end_location",
    "pass_shot_assist", "pass_goal_assist", "carry_end_location",
    "dribble_outcome", "shot_type", "shot_outcome", "shot_statsbomb_xg",
    "duel_type", "duel_outcome", "ball_recovery_recovery_failure",
]


def _xy(series: pd.Series):
    arr = np.array(series.tolist(), dtype=float).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def _dist_to_goal(x, y):
    return np.sqrt((120 - x) ** 2 + (40 - y) ** 2)


def _gain(events: pd.DataFrame, start_col: str, end_col: str):
    """Yards each event moves the ball closer to the opponent's goal."""
    sx, sy = _xy(events[start_col])
    ex, ey = _xy(events[end_col])
    return _dist_to_goal(sx, sy) - _dist_to_goal(ex, ey)


def match_features(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[ev["player_id"].notna()].copy()
    ev["player_id"] = ev["player_id"].astype(int)
    for col in NEEDED:
        if col not in ev.columns:
            ev[col] = np.nan

    all_passes = ev[ev["type"] == "Pass"]
    passes = all_passes[~all_passes["pass_type"].isin(SET_PIECES)]
    done = passes[passes["pass_outcome"].isna()]
    prog_passes = done[_gain(done, "location", "pass_end_location") >= PROGRESS_YARDS]

    carries = ev[ev["type"] == "Carry"]
    prog_carries = carries[_gain(carries, "location", "carry_end_location") >= PROGRESS_YARDS]

    dribbles = ev[ev["type"] == "Dribble"]
    shots = ev[(ev["type"] == "Shot") & (ev["shot_type"] != "Penalty")]
    tackles = ev[(ev["type"] == "Duel") & (ev["duel_type"] == "Tackle")]
    recoveries = ev[
        (ev["type"] == "Ball Recovery") & ~ev["ball_recovery_recovery_failure"].eq(True)
    ]

    counts = {
        "passes": passes,
        "passes_completed": done,
        "progressive_passes": prog_passes,
        "key_passes": all_passes[all_passes["pass_shot_assist"].eq(True)],
        "assists": all_passes[all_passes["pass_goal_assist"].eq(True)],
        "carries": carries,
        "progressive_carries": prog_carries,
        "dribbles": dribbles,
        "dribbles_completed": dribbles[dribbles["dribble_outcome"] == "Complete"],
        "shots": shots,
        "shots_on_target": shots[shots["shot_outcome"].isin(SHOT_ON_TARGET)],
        "pressures": ev[ev["type"] == "Pressure"],
        "tackles": tackles,
        "tackles_won": tackles[tackles["duel_outcome"].isin(TACKLE_WON)],
        "interceptions": ev[ev["type"] == "Interception"],
        "ball_recoveries": recoveries,
    }
    parts = [d.groupby("player_id").size().rename(name) for name, d in counts.items()]
    parts.append(shots.groupby("player_id")["shot_statsbomb_xg"].sum().rename("xg"))

    # Average pitch position: keep sums and a count so seasons can be averaged correctly
    located = ev[ev["location"].notna()]
    lx, ly = _xy(located["location"])
    pos = pd.DataFrame({"player_id": located["player_id"].to_numpy(), "x": lx, "y": ly})
    pos = pos.groupby("player_id").agg(
        sum_x=("x", "sum"), sum_y=("y", "sum"), n_loc=("x", "size")
    )
    parts.append(pos)

    out = pd.concat(parts, axis=1).fillna(0)
    out.index.name = "player_id"
    return out.reset_index()


if __name__ == "__main__":
    path = sorted(Path("data/raw").glob("*.pkl"))[0]
    feats = match_features(pd.read_pickle(path))
    print("shape:", feats.shape)
    view = feats[feats["player_id"].isin([5503, 5203, 4691])].set_index("player_id")
    view["avg_x"] = view["sum_x"] / view["n_loc"]
    print(view.T.round(2).to_string())