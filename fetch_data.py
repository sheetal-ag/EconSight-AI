"""Download state unemployment and participation series from FRED into data/*.csv.

Usage:  python fetch_data.py        (needs FRED_API_KEY in your .env file)
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from tools import STATES


def fetch(fred, pause=0.15):
    """Return (unemployment, participation), one column per state, built from known series ids."""
    def pull(series_id, name):
        time.sleep(pause)  # stay well under FRED's request limit
        return fred.get_series(series_id).rename(name)

    unemp = pd.concat([pull(f"{abbr}UR", n) for n, (abbr, _) in STATES.items()], axis=1)
    part = pd.concat([pull(f"LBSSA{fips}", n) for n, (_, fips) in STATES.items()], axis=1)
    return unemp.dropna(), part.dropna()


def main(out="data"):
    load_dotenv()
    key = os.getenv("FRED_API_KEY")
    if not key:
        sys.exit("FRED_API_KEY is not set. Add it to your .env file.")
    unemp, part = fetch(Fred(api_key=key))
    assert unemp.shape[1] == part.shape[1] == len(STATES), "Expected one column per state"
    Path(out).mkdir(exist_ok=True)
    unemp.to_csv(Path(out) / "unemployment.csv")
    part.to_csv(Path(out) / "participation.csv")
    (Path(out) / "metadata.json").write_text(json.dumps({
        "retrieved": date.today().isoformat(),
        "unemployment_series": "<state abbreviation>UR", "participation_series": "LBSSA<state fips>",
        "last_observation": unemp.index.max().strftime("%Y-%m-%d")}, indent=2))
    print("Saved", unemp.shape, part.shape, "to", out)


if __name__ == "__main__":
    main()
