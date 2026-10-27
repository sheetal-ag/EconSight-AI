#Data layer and the tools the AI analyst may call.


import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2020-01-01", "2021-12-01"   # project scope
BASELINE = pd.Timestamp("2020-02-01")     # last month before COVID hit
RECOVERY_TOLERANCE = 1.0                  # "recovered" = within 1 point of baseline
MAX_STATES_PER_CALL = 6

# name -> (postal abbreviation, FIPS code). FRED ids: <abbr>UR and LBSSA<fips>
STATES = {
    "Alabama": ("AL", "01"), "Alaska": ("AK", "02"), "Arizona": ("AZ", "04"), "Arkansas": ("AR", "05"),
    "California": ("CA", "06"), "Colorado": ("CO", "08"), "Connecticut": ("CT", "09"), "Delaware": ("DE", "10"),
    "Florida": ("FL", "12"), "Georgia": ("GA", "13"), "Hawaii": ("HI", "15"), "Idaho": ("ID", "16"),
    "Illinois": ("IL", "17"), "Indiana": ("IN", "18"), "Iowa": ("IA", "19"), "Kansas": ("KS", "20"),
    "Kentucky": ("KY", "21"), "Louisiana": ("LA", "22"), "Maine": ("ME", "23"), "Maryland": ("MD", "24"),
    "Massachusetts": ("MA", "25"), "Michigan": ("MI", "26"), "Minnesota": ("MN", "27"), "Mississippi": ("MS", "28"),
    "Missouri": ("MO", "29"), "Montana": ("MT", "30"), "Nebraska": ("NE", "31"), "Nevada": ("NV", "32"),
    "New Hampshire": ("NH", "33"), "New Jersey": ("NJ", "34"), "New Mexico": ("NM", "35"), "New York": ("NY", "36"),
    "North Carolina": ("NC", "37"), "North Dakota": ("ND", "38"), "Ohio": ("OH", "39"), "Oklahoma": ("OK", "40"),
    "Oregon": ("OR", "41"), "Pennsylvania": ("PA", "42"), "Rhode Island": ("RI", "44"),
    "South Carolina": ("SC", "45"), "South Dakota": ("SD", "46"), "Tennessee": ("TN", "47"), "Texas": ("TX", "48"),
    "Utah": ("UT", "49"), "Vermont": ("VT", "50"), "Virginia": ("VA", "51"), "Washington": ("WA", "53"),
    "West Virginia": ("WV", "54"), "Wisconsin": ("WI", "55"), "Wyoming": ("WY", "56"),
}
STATE_ABBR = {name: abbr for name, (abbr, _) in STATES.items()}


def load_data(folder="data"):
    folder = Path(folder)
    unemp = pd.read_csv(folder / "unemployment.csv", index_col=0, parse_dates=True)
    part = pd.read_csv(folder / "participation.csv", index_col=0, parse_dates=True)
    return unemp.loc[START:END], part.loc[START:END]


def load_metadata(folder="data"):
    path = Path(folder) / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def label(ts):
    return pd.Timestamp(ts).strftime("%b %Y")


class LaborData:
    def __init__(self, unemp, part):
        self.frames = {"unemployment": unemp, "participation": part}
        self.unemp, self.part = unemp, part
        self.states = list(unemp.columns)
        self.first, self.last = unemp.index.min(), unemp.index.max()

    # ---- validation helpers --------------------------------------------
    def _frame(self, metric):
        if metric not in self.frames:
            raise ValueError(f"metric must be one of {list(self.frames)}")
        return self.frames[metric]

    def _check_states(self, states):
        bad = [s for s in states if s not in self.states]
        if bad:
            raise ValueError(f"Unknown state(s): {bad}. Use full state names, e.g. 'New York'.")

    def _month(self, date):
        try:
            ts = pd.Timestamp(date).to_period("M").to_timestamp()
        except (ValueError, TypeError):
            raise ValueError(f"'{date}' is not a valid month. Use YYYY-MM.")
        if not (self.first <= ts <= self.last):
            raise ValueError(f"No data for {date}. Data covers {label(self.first)} to {label(self.last)}.")
        return ts

    # ---- per-state summary (used by charts, KPIs and tools) ---------------
    @cached_property
    def summary(self):
        u, p = self.unemp, self.part
        rows = []
        for s in self.states:
            series = u[s]
            peak_month = series.idxmax()
            back = series.loc[peak_month:]
            back = back[back <= series[BASELINE] + RECOVERY_TOLERANCE]
            months = np.nan if back.empty else (back.index[0].to_period("M") - peak_month.to_period("M")).n
            part_after = p[s].loc[BASELINE:]
            rows.append({
                "state": s,
                "baseline_unemployment": series[BASELINE],
                "peak_unemployment": series.max(),
                "peak_month": label(peak_month),
                "months_to_recover": months,
                "latest_unemployment": series.iloc[-1],
                "baseline_participation": p[s][BASELINE],
                "lowest_participation": part_after.min(),
                "lowest_participation_month": label(part_after.idxmin()),
                "latest_participation": p[s].iloc[-1],
            })
        t = pd.DataFrame(rows).set_index("state")
        t["unemployment_increase"] = t["peak_unemployment"] - t["baseline_unemployment"]
        t["participation_drop"] = t["lowest_participation"] - t["baseline_participation"]
        t["participation_change_latest"] = t["latest_participation"] - t["baseline_participation"]
        return t

    def highlights(self):
        t = self.summary
        rec = t.dropna(subset=["months_to_recover"])
        top = t["peak_unemployment"].idxmax()
        out = {
            "period": f"{label(self.first)} to {label(self.last)}",
            "recovery_definition": f"unemployment back within {RECOVERY_TOLERANCE} point of its {label(BASELINE)} level",
            "highest_peak": {"state": top, "rate": float(round(t.loc[top, "peak_unemployment"], 1)),
                             "month": t.loc[top, "peak_month"]},
            "not_recovered_by_end": sorted(t.index[t["months_to_recover"].isna()].tolist()),
            "participation_still_below_baseline_count": int((t["participation_change_latest"] < -0.5).sum()),
        }
        if not rec.empty:
            fast, slow = rec["months_to_recover"].idxmin(), rec["months_to_recover"].idxmax()
            out["fastest_recovery"] = {"state": fast, "months": int(rec.loc[fast, "months_to_recover"])}
            out["slowest_recovery"] = {"state": slow, "months": int(rec.loc[slow, "months_to_recover"])}
        drop = t["participation_drop"].idxmin()
        out["biggest_participation_drop"] = {"state": drop, "points": float(round(t.loc[drop, "participation_drop"], 1))}
        return out

    def relationship_frame(self):
        t = self.summary
        return t[["unemployment_increase", "participation_drop"]].rename(
            columns={"participation_drop": "participation_change"})

    # ---- tools exposed to the LLM ---------------------------------------
    def rank_states(self, metric, date, top_n=10, order="highest"):
        if order not in ("highest", "lowest"):
            raise ValueError("order must be 'highest' or 'lowest'")
        df, ts = self._frame(metric), self._month(date)
        vals = df.loc[ts].sort_values(ascending=(order == "lowest")).head(max(1, min(int(top_n), 50)))
        return {"metric": metric, "month": label(ts), "order": order, "unit": "percent",
                "results": [{"state": s, "value": round(float(v), 1)} for s, v in vals.items()]}

    def get_series(self, states, metric, start=None, end=None):
        self._check_states(states)
        if len(states) > MAX_STATES_PER_CALL:
            raise ValueError(f"Ask for at most {MAX_STATES_PER_CALL} states at a time.")
        df = self._frame(metric)
        a = self._month(start) if start else self.first
        b = self._month(end) if end else self.last
        if a > b:
            raise ValueError("start must not be after end")
        sub = df.loc[a:b, states]
        return {"metric": metric, "unit": "percent",
                "data": {s: {d.strftime("%Y-%m"): round(float(v), 1) for d, v in sub[s].items()} for s in states}}

    def largest_change(self, metric, start, end, top_n=10, order="increase"):
        if order not in ("increase", "decrease"):
            raise ValueError("order must be 'increase' or 'decrease'")
        df, a, b = self._frame(metric), self._month(start), self._month(end)
        if a >= b:
            raise ValueError("start must be before end")
        change = (df.loc[b] - df.loc[a]).sort_values(ascending=(order == "decrease"))
        change = change.head(max(1, min(int(top_n), 50)))
        return {"metric": metric, "from": label(a), "to": label(b), "unit": "percentage points", "order": order,
                "results": [{"state": s, "start_value": round(float(df.loc[a, s]), 1),
                             "end_value": round(float(df.loc[b, s]), 1), "change": round(float(c), 1)}
                            for s, c in change.items()]}

    def covid_summary(self, state):
        self._check_states([state])
        row = self.summary.loc[state]
        out = {}
        for k, v in row.items():
            if k == "months_to_recover":
                out[k] = None if pd.isna(v) else int(v)
            else:
                out[k] = round(float(v), 1) if isinstance(v, (int, float, np.floating)) else v
        out["definitions"] = (f"baseline = {label(BASELINE)}; recovered = unemployment within "
                              f"{RECOVERY_TOLERANCE} point of baseline, counted in months from the peak")
        return {"state": state, **out}

    def relationship(self):
        f = self.relationship_frame()
        x, y = f["unemployment_increase"], f["participation_change"]
        slope, _ = np.polyfit(x, y, 1)
        return {"states": int(len(f)), "correlation": round(float(x.corr(y)), 2),
                "slope_points": round(float(slope), 2),
                "x": "rise in unemployment rate from Feb 2020 to its peak (points)",
                "y": "change in participation rate from Feb 2020 to its lowest point (points)",
                "note": "This is an association across states, not proof that one caused the other."}


# JSON schemas the model sees
_METRIC = {"type": "string", "enum": ["unemployment", "participation"]}
TOOLS = [
    {"name": "rank_states",
     "description": "Rank US states by unemployment rate or labor force participation rate in one month.",
     "input_schema": {"type": "object", "properties": {
         "metric": _METRIC,
         "date": {"type": "string", "description": "Month as YYYY-MM, between 2020-01 and 2021-12"},
         "top_n": {"type": "integer", "description": "How many states to return (default 10)"},
         "order": {"type": "string", "enum": ["highest", "lowest"]}},
         "required": ["metric", "date"]}},
    {"name": "largest_change",
     "description": ("Which states changed most between two months (end value minus start value, in percentage "
                     "points). Use for 'how did X change during 2020' questions, e.g. start 2020-01, end 2020-12."),
     "input_schema": {"type": "object", "properties": {
         "metric": _METRIC,
         "start": {"type": "string", "description": "YYYY-MM"},
         "end": {"type": "string", "description": "YYYY-MM, after start"},
         "top_n": {"type": "integer"},
         "order": {"type": "string", "enum": ["increase", "decrease"]}},
         "required": ["metric", "start", "end"]}},
    {"name": "get_series",
     "description": "Monthly values for up to 6 states. Use to compare states or describe a trend.",
     "input_schema": {"type": "object", "properties": {
         "states": {"type": "array", "items": {"type": "string"}, "description": "Full state names"},
         "metric": _METRIC,
         "start": {"type": "string", "description": "YYYY-MM (optional)"},
         "end": {"type": "string", "description": "YYYY-MM (optional)"}},
         "required": ["states", "metric"]}},
    {"name": "covid_summary",
     "description": ("One state's pre-COVID baseline, unemployment peak and month, months to recover, "
                     "lowest participation and latest values."),
     "input_schema": {"type": "object", "properties": {"state": {"type": "string", "description": "Full state name"}},
                      "required": ["state"]}},
    {"name": "relationship",
     "description": ("How the rise in unemployment relates to the fall in participation across all states "
                     "(correlation and slope). Use for questions about the relationship between the two."),
     "input_schema": {"type": "object", "properties": {}}},
]

_ALLOWED = {t["name"] for t in TOOLS}


def run_tool(data, name, args):
    """Run a tool the model asked for and return a JSON string (errors included)."""
    try:
        if name not in _ALLOWED:
            raise ValueError(f"Unknown tool {name}")
        out = getattr(data, name)(**(args or {}))
    except (ValueError, KeyError, TypeError) as e:
        out = {"error": str(e)}
    return json.dumps(out)
