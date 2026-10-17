"""Download StatsBomb events for the four complete 2015/16 leagues
and cache one file per match in data/raw."""
import argparse
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore", message="credentials were not supplied.")

LEAGUES = ["Premier League", "LaLiga", "Serie A", "Ligue 1"]
SEASONS = "2015/2016"
RAW_DIR = Path("data/raw")

def get_league_seasons() -> pd.DataFrame:
    """Return the competition rows for our four leagues in 2015/16."""
    comps = sb.competitions()
    mask = comps["competition_name"].isin(LEAGUES) & (comps["season_name"] == SEASONS)
    return comps[mask]

def download_events(limit = None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for _, comp in get_league_seasons().iterrows():
        matches = sb.matches(
            competition_id=int(comp.competition_id), season_id=int(comp.season_id)
        )
        print(f"{comp.competition_name}: {len(matches)} matches")
        for match_id in matches.match_id:
            if limit is not None and downloaded >= limit:
                print(f"Stopped at the limit of {limit} matches.")
                return
            path = RAW_DIR / f"{match_id}.pkl"
            if path.exists():
                continue
            sb.events(match_id=int(match_id)).to_pickle(path)
            downloaded += 1
            if downloaded % 25 == 0:
                print(f" {downloaded} new matches downloaded.")
        print(f"Done. {downloaded} new matches downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max new matches to download")
    args = parser.parse_args()
    download_events(limit=args.limit)