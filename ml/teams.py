"""Team name cleanup: StatsBomb uses two names for some clubs."""
import pandas as pd

TEAM_ALIASES = {
    "Caen": "Stade Malherbe Caen",
    "Marseille": "Olympique de Marseille",
}

def clean_team_names(teams: pd.Series) -> pd.Series:
    return teams.replace(TEAM_ALIASES)