"""
Stadium Operations Analytics — Synthetic Data Generator
========================================================
Generates a realistic match-day dataset of ~100 fixtures at a 55,000-capacity
Premier League stadium. Models attendance, concession revenue, staffing, and
an operations pressure score.

Author : Portfolio project (Business Analytics MSc)
Usage  : python data_generation/generate_data.py
Output : data/match_day_data.csv, dashboard/data/match_day_data.json
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── constants ────────────────────────────────────────────────────────────────
STADIUM_CAPACITY = 55_000
NUM_MATCHES = 100
SEASON_START = datetime(2024, 8, 10)
SEASON_END = datetime(2025, 5, 25)

# ── opponent pools ───────────────────────────────────────────────────────────
TOP6 = [
    "Manchester City", "Arsenal", "Liverpool",
    "Chelsea", "Tottenham", "Newcastle United",
]
MID_TABLE = [
    "Aston Villa", "Brighton", "West Ham", "Brentford",
    "Crystal Palace", "Fulham", "Bournemouth", "Wolves",
    "Nottingham Forest", "Everton",
]
RELEGATION_ZONE = [
    "Burnley", "Sheffield United", "Luton Town",
    "Ipswich Town", "Leicester City",
]

COMPETITIONS = ["Premier League", "FA Cup", "League Cup", "Friendly"]
KICKOFF_TIMES = ["12:30", "15:00", "17:30", "20:00"]


# ── helper: generate fixture list ────────────────────────────────────────────
def _generate_fixtures(n: int) -> pd.DataFrame:
    """Build a calendar of n fixtures with realistic EPL-style scheduling."""

    # spread match dates roughly evenly across the season window
    total_days = (SEASON_END - SEASON_START).days
    dates = sorted(
        [SEASON_START + timedelta(days=int(d))
         for d in np.linspace(0, total_days, n + 2)[1:-1]]
    )
    # nudge each date by -1/0/+1 day to add jitter
    dates = [d + timedelta(days=int(np.random.choice([-1, 0, 1]))) for d in dates]

    records = []
    for i, date in enumerate(dates):
        dow = date.strftime("%A")  # Monday … Sunday

        # ── opponent tier ────────────────────────────────────────────────
        tier_roll = np.random.random()
        if tier_roll < 0.20:
            tier = "top6"
            opponent = np.random.choice(TOP6)
        elif tier_roll < 0.70:
            tier = "mid"
            opponent = np.random.choice(MID_TABLE)
        else:
            tier = "relegation"
            opponent = np.random.choice(RELEGATION_ZONE)

        # ── competition ──────────────────────────────────────────────────
        comp_roll = np.random.random()
        if comp_roll < 0.72:
            competition = "Premier League"
        elif comp_roll < 0.84:
            competition = "FA Cup"
        elif comp_roll < 0.94:
            competition = "League Cup"
        else:
            competition = "Friendly"

        # ── kickoff time (conditional on day of week) ────────────────────
        if dow in ("Tuesday", "Wednesday", "Thursday"):
            # midweek → overwhelmingly evening
            ko = np.random.choice(
                KICKOFF_TIMES, p=[0.02, 0.05, 0.08, 0.85]
            )
        elif dow == "Saturday":
            ko = np.random.choice(
                KICKOFF_TIMES, p=[0.20, 0.40, 0.25, 0.15]
            )
        elif dow == "Sunday":
            ko = np.random.choice(
                KICKOFF_TIMES, p=[0.10, 0.35, 0.35, 0.20]
            )
        else:
            # Monday / Friday → mostly evening
            ko = np.random.choice(
                KICKOFF_TIMES, p=[0.05, 0.10, 0.20, 0.65]
            )

        day_type = "Weekend" if dow in ("Saturday", "Sunday") else "Midweek"

        records.append({
            "match_id": i + 1,
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": dow,
            "day_type": day_type,
            "opponent": opponent,
            "opponent_tier": tier,
            "competition": competition,
            "kickoff_time": ko,
        })

    return pd.DataFrame(records)


# ── helper: attendance model ─────────────────────────────────────────────────
def _model_attendance(df: pd.DataFrame) -> pd.Series:
    """
    Attendance drawn from a shifted beta distribution, then modified by
    opponent tier, kickoff time, day type, and competition.
    """
    n = len(df)

    # base attendance: beta(5, 2) ∈ [0,1] → scaled to [35000, 55000]
    base = np.random.beta(5, 2, size=n) * 20_000 + 35_000

    # ── tier multiplier ──────────────────────────────────────────────────
    tier_mult = df["opponent_tier"].map({
        "top6": 1.10,
        "mid": 1.00,
        "relegation": 0.93,
    }).values

    # ── kickoff / day multiplier ─────────────────────────────────────────
    def _ko_day_mult(row):
        ko = row["kickoff_time"]
        day = row["day_type"]
        if day == "Weekend" and ko == "15:00":
            return 1.04
        if day == "Weekend" and ko in ("17:30", "20:00"):
            return 1.02
        if day == "Midweek" and ko == "20:00":
            return 0.96
        if ko == "12:30":
            return 0.95
        return 1.00

    ko_mult = df.apply(_ko_day_mult, axis=1).values

    # ── competition multiplier ───────────────────────────────────────────
    comp_mult = df["competition"].map({
        "Premier League": 1.00,
        "FA Cup": 1.03,
        "League Cup": 0.92,
        "Friendly": 0.82,
    }).values

    attendance = base * tier_mult * ko_mult * comp_mult

    # add normal noise (σ ≈ 2000)
    attendance += np.random.normal(0, 2000, size=n)

    # clip to realistic bounds
    attendance = np.clip(attendance, 28_000, STADIUM_CAPACITY).astype(int)

    return pd.Series(attendance, index=df.index)


# ── helper: concession revenue model ────────────────────────────────────────
def _model_concessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-head spend by outlet type, modified by kickoff time and opponent tier.
    """
    n = len(df)
    attendance = df["attendance"].values

    # base per-head spend (£)
    bar_base = 4.50
    food_base = 3.80
    retail_base = 2.20

    # ── kickoff-time multipliers ─────────────────────────────────────────
    ko_map = {
        "12:30": {"bar": 0.75, "food": 1.25, "retail": 1.00},
        "15:00": {"bar": 1.00, "food": 1.00, "retail": 1.00},
        "17:30": {"bar": 1.15, "food": 1.05, "retail": 1.00},
        "20:00": {"bar": 1.35, "food": 0.90, "retail": 0.95},
    }
    bar_ko = df["kickoff_time"].map(lambda x: ko_map[x]["bar"]).values
    food_ko = df["kickoff_time"].map(lambda x: ko_map[x]["food"]).values
    retail_ko = df["kickoff_time"].map(lambda x: ko_map[x]["retail"]).values

    # ── opponent-tier effect on retail (big match → more merch) ──────────
    retail_tier = df["opponent_tier"].map({
        "top6": 1.30, "mid": 1.00, "relegation": 0.90,
    }).values

    # revenue = attendance × per-head × multipliers + noise
    noise_pct = 0.08  # 8 % standard deviation

    bar_rev = (attendance * bar_base * bar_ko
               * (1 + np.random.normal(0, noise_pct, n)))
    food_rev = (attendance * food_base * food_ko
                * (1 + np.random.normal(0, noise_pct, n)))
    retail_rev = (attendance * retail_base * retail_ko * retail_tier
                  * (1 + np.random.normal(0, noise_pct, n)))

    # ensure non-negative
    bar_rev = np.maximum(bar_rev, 0)
    food_rev = np.maximum(food_rev, 0)
    retail_rev = np.maximum(retail_rev, 0)

    return pd.DataFrame({
        "concession_bar_revenue": np.round(bar_rev, 2),
        "concession_food_revenue": np.round(food_rev, 2),
        "concession_retail_revenue": np.round(retail_rev, 2),
        "concession_total_revenue": np.round(bar_rev + food_rev + retail_rev, 2),
    }, index=df.index)


# ── helper: staffing model ──────────────────────────────────────────────────
def _model_staffing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Staff scheduled scales with attendance, with a step increase once
    attendance exceeds 85% capacity. Overtime hours increase non-linearly
    at high attendance.
    """
    n = len(df)
    attendance = df["attendance"].values
    capacity_pct = attendance / STADIUM_CAPACITY

    # base ratio: 1 staff per 110 attendees
    # above 85 % threshold: 1 per 95
    THRESHOLD = 0.85
    base_ratio_low = 110
    base_ratio_high = 95

    staff = np.where(
        capacity_pct < THRESHOLD,
        attendance / base_ratio_low,
        attendance / base_ratio_high,
    )

    # add ±5 % noise and round
    staff = staff * (1 + np.random.normal(0, 0.05, n))
    staff = np.round(staff).astype(int)
    staff = np.maximum(staff, 250)  # minimum crew

    # overtime hours
    ot = np.where(
        capacity_pct < 0.75,
        np.random.uniform(0, 20, n),
        np.where(
            capacity_pct < 0.90,
            np.random.uniform(15, 60, n),
            np.where(
                capacity_pct < 0.95,
                np.random.uniform(50, 130, n),
                np.random.uniform(80, 180, n),
            )
        )
    )
    ot = np.round(ot, 1)

    return pd.DataFrame({
        "staff_scheduled": staff,
        "overtime_hours": ot,
    }, index=df.index)


# ── helper: operations pressure score ────────────────────────────────────────
def _operations_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Composite 0–100 index:
      - attendance %           (weight 0.40)
      - attendees per staff    (weight 0.35)
      - concession rev / staff (weight 0.25)
    Each component is min-max normalised before weighting.
    """
    att_pct = df["attendance"] / STADIUM_CAPACITY
    att_per_staff = df["attendance"] / df["staff_scheduled"]
    rev_per_staff = df["concession_total_revenue"] / df["staff_scheduled"]

    def _minmax(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    score = (
        0.40 * _minmax(att_pct)
        + 0.35 * _minmax(att_per_staff)
        + 0.25 * _minmax(rev_per_staff)
    ) * 100

    return score.round(1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("🏟  Stadium Operations — Data Generator")
    print("=" * 50)

    # 1. fixtures
    df = _generate_fixtures(NUM_MATCHES)
    print(f"✓ Generated {len(df)} fixtures")

    # 2. attendance
    df["attendance"] = _model_attendance(df)
    df["attendance_pct"] = (df["attendance"] / STADIUM_CAPACITY * 100).round(1)
    print(f"✓ Modelled attendance  (mean {df['attendance'].mean():,.0f})")

    # 3. concessions
    conc = _model_concessions(df)
    df = pd.concat([df, conc], axis=1)
    print(f"✓ Modelled concessions (season total £{df['concession_total_revenue'].sum():,.0f})")

    # 4. staffing
    staff = _model_staffing(df)
    df = pd.concat([df, staff], axis=1)
    print(f"✓ Modelled staffing    (mean {df['staff_scheduled'].mean():.0f} per match)")

    # 5. operations pressure
    df["ops_pressure_score"] = _operations_pressure(df)
    print(f"✓ Computed ops pressure (mean {df['ops_pressure_score'].mean():.1f}/100)")

    # 6. derived convenience columns
    df["per_head_bar"] = (df["concession_bar_revenue"] / df["attendance"]).round(2)
    df["per_head_food"] = (df["concession_food_revenue"] / df["attendance"]).round(2)
    df["per_head_retail"] = (df["concession_retail_revenue"] / df["attendance"]).round(2)
    df["per_head_total"] = (df["concession_total_revenue"] / df["attendance"]).round(2)
    df["attendees_per_staff"] = (df["attendance"] / df["staff_scheduled"]).round(1)

    # ── write outputs ────────────────────────────────────────────────────
    # ensure directories exist
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "data")
    dash_data_dir = os.path.join(base, "dashboard", "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(dash_data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "match_day_data.csv")
    json_path = os.path.join(dash_data_dir, "match_day_data.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\n📁 CSV  → {csv_path}")
    print(f"📁 JSON → {json_path}")
    print(f"\n🎉 Done — {len(df)} matches generated.")

    # quick sanity check
    print("\n── Sample rows ──")
    print(df[["date", "opponent", "opponent_tier", "kickoff_time",
              "attendance", "concession_total_revenue",
              "staff_scheduled", "ops_pressure_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
