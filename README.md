# 🏟️ Stadium Operations Analytics

**How crowd size drives operational flow inside a Premier League football stadium**

A data analytics portfolio project that models and quantifies the relationship between match-day attendance and three critical operational levers: **staffing requirements**, **concession revenue**, and **operations pressure**. Built for a 55,000-capacity stadium over ~100 synthetic fixtures.

---

## 📌 Problem Statement

Stadium operations managers must make staffing, stock, and logistics decisions *before* each match based on expected attendance. But not all 45,000-attendance matches are the same — a Saturday 20:00 kickoff against a top-6 rival creates a very different operational profile than a Tuesday night League Cup match against a relegation-zone team.

This project explores:
1. **Where is the staffing inflection point?** At what attendance level do staffing requirements jump non-linearly?
2. **How does kickoff time change concession demand?** Do evening games drive different spending patterns than early kickoffs?
3. **Which match profiles stress operations the most?** Can we identify the "perfect storm" combinations of opponent tier, kickoff time, and day of week?
4. **Can we build a simple staffing recommendation tool?** Given expected attendance and kickoff time, what's the right staff count?

---

## 🗂️ Project Structure

```
Stadium Operations/
├── README.md                           ← You are here
├── requirements.txt                    ← Python dependencies
├── data_generation/
│   └── generate_data.py                ← Synthetic dataset generator
├── analysis/
│   └── analyse.py                      ← Statistical analysis & model
├── data/
│   └── match_day_data.csv              ← Generated dataset (100 matches)
├── dashboard/
│   ├── index.html                      ← Interactive web dashboard
│   └── data/
│       ├── match_day_data.json         ← Dataset for the dashboard
│       └── staffing_model.json         ← Regression model coefficients
└── outputs/
    ├── analysis_report.md              ← Detailed findings
    └── charts/                         ← Static chart images (PNG)
        ├── staffing_inflection.png
        ├── concession_mix.png
        ├── pressure_profiles.png
        └── staffing_model_residuals.png
```

---

## ⚠️ Why Synthetic Data?

This project uses **entirely synthetic data**. No real club data was accessed or used. This is a deliberate methodological choice:

- **No access to real operational data**: Stadium concession volumes, staffing rosters, and operational metrics are commercially sensitive and not publicly available.
- **Grounded in domain knowledge**: Assumptions are informed by publicly available EPL attendance figures, stadium capacity data, and the author's experience working event hospitality shifts at Premier League venues.
- **Transparent modelling**: Every assumption is explicitly documented in the data generation script and listed below.
- **Focus on methodology**: The value of this project is in the analytical framework — the approach would apply directly to real data if available.

### Key Assumptions

| Parameter | Assumed Value | Rationale |
|-----------|:------------:|-----------|
| Stadium capacity | 55,000 | Typical large EPL venue |
| Base per-head spend (bar) | £4.50 | Industry benchmark for matchday hospitality |
| Base per-head spend (food) | £3.80 | Includes hot food, coffee, snacks |
| Base per-head spend (retail) | £2.20 | Merchandise, programmes |
| Staff-to-attendee ratio (normal) | 1 : 110 | Below 85% capacity threshold |
| Staff-to-attendee ratio (high load) | 1 : 95 | Above 85% capacity threshold |
| Evening kickoff bar multiplier | ×1.35 | Industry observation: evening = more drinks |
| Early kickoff food multiplier | ×1.25 | "Coffee & pie" effect for 12:30 starts |

---

## 🔬 Methodology

### 1. Data Generation
A Python script generates 100 match-day records with correlated, noisy variables:
- **Attendance** is drawn from a beta distribution and modified by opponent tier (top-6 ≈ +10%), kickoff time, day type, and competition.
- **Concessions** model three outlet types with per-head spend adjusted by kickoff-time multipliers and opponent-tier effects.
- **Staffing** scales with attendance via a step-function: a standard ratio of 1:110 below 85% capacity, tightening to 1:95 above. Overtime hours increase non-linearly at high-attendance matches.
- **Operations Pressure Score** is a composite 0–100 index combining attendance %, attendees-per-staff ratio, and concession volume per staff member.

All variables include random noise (typically 5–8% σ) to prevent perfectly linear relationships.

### 2. Analysis
- **Piecewise-linear regression** (scipy) identifies the staffing inflection point.
- **Grouped aggregations** reveal concession mix shifts by kickoff time.
- **Profile ranking** identifies the highest-pressure match combinations.
- **Linear regression** (scikit-learn) builds a staffing recommendation model using attendance and kickoff time as predictors (R² = 0.86, MAE = 25 staff).

### 3. Dashboard
A self-contained HTML/CSS/JS dashboard with three interactive views, built with Chart.js. No build step, no server — just open `dashboard/index.html` in a browser.

---

## 📊 Key Findings

*Written as if presenting to a stadium operations manager:*

### 1. Staff numbers need to step up sharply once attendance crosses ~46,650 (85% capacity)

Below this threshold, one staff member per ~113 attendees is sufficient. Above it, the ratio tightens to roughly 1:89 — requiring a significantly higher staffing density. This confirms that crowd management, security screening, and service bottlenecks increase non-linearly at high-attendance matches. Planning should treat the ~85% line as a trigger for activating reserve staffing pools.

### 2. Evening kickoffs shift concession spend toward the bar by +35%

A 20:00 Saturday kickoff generates 35% higher bar revenue per head than a 15:00 start, while food/coffee spending drops by ~10%. Conversely, 12:30 early kickoffs drive a 25% uplift in food counters (the "coffee and breakfast roll" effect). **Operational implication**: stock and staff rosters for bar vs. food outlets should be adjusted based on kickoff time, not just total attendance.

### 3. Weekend late-afternoon matches against mid-table opponents are the most operationally stretched

The highest operations pressure score (60.9/100) belongs to **mid-table opponents at 17:30 weekend kickoffs** — not top-6 evening matches as intuition might suggest. This is because these matches draw high-but-not-maximum attendance while staffing hasn't yet stepped up to the "high capacity" tier, creating a gap between demand and coverage.

### 4. A simple model can recommend staff counts with ±25 accuracy

Using just expected attendance and kickoff time, a linear regression model predicts the required staff count with an R² of 0.86 and MAE of ~25 staff. The model is embedded in the dashboard as an interactive tool — enter your expected attendance and kickoff time to get a recommendation.

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+ (the scripts use `uv` for zero-install dependency management, or install from `requirements.txt`)
- A modern web browser (Chrome, Firefox, Safari) for the dashboard

### Quick Start

```bash
# 1. Generate the synthetic dataset
uv run --with pandas --with numpy --with scipy --with scikit-learn --with matplotlib --with seaborn \
    python data_generation/generate_data.py

# 2. Run the analysis
uv run --with pandas --with numpy --with scipy --with scikit-learn --with matplotlib --with seaborn \
    python analysis/analyse.py

# 3. Open the dashboard
open dashboard/index.html
```

### Alternative (with pip)

```bash
pip install -r requirements.txt
python data_generation/generate_data.py
python analysis/analyse.py
open dashboard/index.html
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Generation | Python, pandas, NumPy |
| Analysis | scipy (piecewise fit), scikit-learn (linear regression), matplotlib/seaborn |
| Dashboard | Vanilla HTML/CSS/JS, Chart.js (CDN) |
| Design | Glassmorphic dark theme, Inter typeface, responsive layout |

---

## 📝 Author Notes

This is a portfolio project for an MSc in Business Analytics. It demonstrates:
- Domain modelling with synthetic but plausible data
- End-to-end analytical pipeline (generation → analysis → visualisation)
- Interactive data communication for a non-technical audience
- Practical business recommendations grounded in quantitative analysis

The analytical framework is designed to be directly applicable to real stadium operational data if made available.
