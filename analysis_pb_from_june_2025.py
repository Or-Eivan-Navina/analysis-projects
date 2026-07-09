# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: venv_test
#     language: python
#     name: python3
# ---

# %%
from navina_da_utils.utils.data_loading import load_from_snowflake, load_from_postgres
import pandas as pd
from datetime import datetime, date, timedelta

# %% [markdown]
# # Patient Base Scheduling Analysis
#
# Comparing cadence-based vs freshness-based scheduling strategies for Patient Base runs.

# %%
"""
Section 1 — Load & validate encounter data
===========================================
Reads the encounter CSV, parses dates, and validates the schema/assumptions
the rest of the analysis relies on. Returns a clean dataframe.

Assumptions checked:
  - required columns present
  - RUNS_TYPE in {DAILY, SAME_DAY}
  - DAILY <-> RUNS_BEFORE_DOS == 2 ; SAME_DAY <-> RUNS_BEFORE_DOS == 1
  - DOS parseable as date
  - eligibility months parseable as YYYY-MM
  - no DAILY row with 0-day booking lead (agreed data rule)
"""

import pandas as pd


REQUIRED_COLS = [
    "PATIENT_KEY",
    "MIN_ELIGIBILITY_MONTH",
    "MAX_ELIGIBILITY_MONTH",
    "DOS",
    "DAYS_SCHEDULED_BEFORE_DOS",
    "RUNS_BEFORE_DOS",
    "RUNS_TYPE",
]


def load_encounters(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # drop the unnamed index column if present
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]

    # --- column presence ---
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- parse dates ---
    df["DOS"] = pd.to_datetime(df["DOS"], errors="coerce")
    # eligibility months are YYYY-MM -> anchor to first day of that month
    df["MIN_ELIG"] = pd.to_datetime(df["MIN_ELIGIBILITY_MONTH"], format="%Y-%m", errors="coerce")
    df["MAX_ELIG"] = pd.to_datetime(df["MAX_ELIGIBILITY_MONTH"], format="%Y-%m", errors="coerce")

    # --- type normalisation ---
    df["RUNS_TYPE"] = df["RUNS_TYPE"].str.strip().str.upper()
    df["DAYS_SCHEDULED_BEFORE_DOS"] = pd.to_numeric(df["DAYS_SCHEDULED_BEFORE_DOS"], errors="coerce")
    df["RUNS_BEFORE_DOS"] = pd.to_numeric(df["RUNS_BEFORE_DOS"], errors="coerce")

    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    problems = []

    # DOS parse
    n_bad_dos = df["DOS"].isna().sum()
    if n_bad_dos:
        problems.append(f"{n_bad_dos} rows with unparseable DOS")

    # eligibility parse
    n_bad_elig = df["MIN_ELIG"].isna().sum() + df["MAX_ELIG"].isna().sum()
    if n_bad_elig:
        problems.append(f"{n_bad_elig} rows with unparseable eligibility months")

    # RUNS_TYPE domain
    bad_types = set(df["RUNS_TYPE"].dropna().unique()) - {"DAILY", "SAME_DAY"}
    if bad_types:
        problems.append(f"unexpected RUNS_TYPE values: {bad_types}")

    # DAILY <-> 2, SAME_DAY <-> 1
    daily_bad = df[(df["RUNS_TYPE"] == "DAILY") & (df["RUNS_BEFORE_DOS"] != 2)]
    sameday_bad = df[(df["RUNS_TYPE"] == "SAME_DAY") & (df["RUNS_BEFORE_DOS"] != 1)]
    if len(daily_bad):
        problems.append(f"{len(daily_bad)} DAILY rows where RUNS_BEFORE_DOS != 2")
    if len(sameday_bad):
        problems.append(f"{len(sameday_bad)} SAME_DAY rows where RUNS_BEFORE_DOS != 1")

    # DAILY with 0-day booking lead should not occur (agreed data rule)
    daily_zero_lead = df[(df["RUNS_TYPE"] == "DAILY") & (df["DAYS_SCHEDULED_BEFORE_DOS"] == 0)]
    if len(daily_zero_lead):
        problems.append(f"{len(daily_zero_lead)} DAILY rows with 0-day booking lead (unexpected)")

    if problems:
        raise ValueError("Validation failed:\n  - " + "\n  - ".join(problems))


def summarize(df: pd.DataFrame) -> None:
    print(f"rows:                {len(df)}")
    print(f"unique patients:     {df['PATIENT_KEY'].nunique()}")
    print(f"DOS range:           {df['DOS'].min().date()} -> {df['DOS'].max().date()}")
    print(f"RUNS_TYPE counts:    {df['RUNS_TYPE'].value_counts().to_dict()}")
    print(f"elig month range:    {df['MIN_ELIG'].min().date()} -> {df['MAX_ELIG'].max().date()}")
    enc_per_pt = df.groupby('PATIENT_KEY').size()
    print(f"encounters/patient:  min {enc_per_pt.min()}, max {enc_per_pt.max()}, mean {enc_per_pt.mean():.2f}")


if __name__ == "__main__":
    PATH = "./analysis_input_data/encounter_level_per_patients_for_analysis_with_min_one_encounter_after_june_2025_20260701.parquet"
    df = load_encounters(PATH)
    summarize(df)
    print("\nSection 1 OK — data loaded and validated.")

# %%
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# %%
"""
Section 2 — Filter to the study population
==========================================
Two filters:
  (a) keep encounters whose DOS falls in the study window (May 2025 - May 2026)
  (b) restrict to patients eligible for the FULL window
      (MIN_ELIG <= Jan 2025 AND MAX_ELIG >= study_end)

Returns the encounters df + the set of kept patients.
"""

import pandas as pd


# scoring window — where PB runs are COUNTED and quality is measured
SCORING_START = pd.Timestamp("2025-06-01")
SCORING_END   = pd.Timestamp("2026-05-31")

# history window — DOS pulled this far back to seed real freshness at Jun 1
HISTORY_START = pd.Timestamp("2025-01-01")
HISTORY_END   = pd.Timestamp("2026-05-31")   # = SCORING_END


# "full min 12-month eligible" thresholds (month anchors)
ELIG_MIN_MAX = pd.Timestamp("2025-06-01")   # MIN_ELIG must be <= this
ELIG_MAX_MIN = pd.Timestamp("2026-05-01")   # MAX_ELIG must be >= this


def filter_population(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n0 = len(df)
    pt0 = df["PATIENT_KEY"].nunique()

    # (a) keep encounters whose DOS falls in the HISTORY window (not just scoring)
    in_window = df["DOS"].between(HISTORY_START, HISTORY_END)
    df = df[in_window].copy()

    # (b) full-12-month eligibility (patient-level property)
    full_elig = (df["MIN_ELIG"]<=ELIG_MIN_MAX) & (df["MAX_ELIG"] >= ELIG_MAX_MIN)
    df = df[full_elig].copy()

    if verbose:
        print("=== Section 2 — population filter ===")
        print(f"start:                 {n0} rows / {pt0} patients")
        print(f"after DOS-in-window:   {in_window.sum()} rows")
        print(f"after full-eligibility:{len(df)} rows / {df['PATIENT_KEY'].nunique()} patients")
        dropped_pt = pt0 - df["PATIENT_KEY"].nunique()
        print(f"patients dropped:      {dropped_pt} "
              f"({dropped_pt / pt0 * 100:.1f}% of original)")

    return df


def get_patients(df: pd.DataFrame) -> list:
    """The v1 patient set, after filtering."""
    return sorted(df["PATIENT_KEY"].unique().tolist())


if __name__ == "__main__":

    df = filter_population(df)
    print(f"\nv1 patients: {len(get_patients(df))}")
    print("Section 2 OK — population filtered.")

# %%
"""
Section 3 — Synthesize Risk portrait timestamps (VECTORIZED)
============================================================
Synthesize from the WIDE DOS pull (Jan 2025 onward), gated by eligibility:
a portrait is only created where DOS >= MIN_ELIG (Risk can't run on a patient
before they were eligible). Each portrait is tagged pre-window vs in-window.

  SAME_DAY -> {DOS}
  DAILY    -> {DOS - min(7, DAYS_SCHEDULED_BEFORE_DOS), DOS - 1}

Output:
  risk_events   : long df, one row per portrait, with provenance + is_prewindow
  risk_timeline : dict PATIENT_KEY -> sorted deduped list[Timestamp] (ALL portraits,
                  incl. pre-window, for seeding freshness)
"""

import pandas as pd
import numpy as np

SCORING_START = pd.Timestamp("2025-06-01")   # portraits before this seed freshness only


def synth_risk_portraits(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    d = df[["PATIENT_KEY", "DOS", "RUNS_TYPE", "DAYS_SCHEDULED_BEFORE_DOS", "MIN_ELIG"]].copy()
    d["DAYS_SCHEDULED_BEFORE_DOS"] = d["DAYS_SCHEDULED_BEFORE_DOS"].fillna(0).astype(int)

    # --- eligibility-floor gate: only synth where DOS >= MIN_ELIG ---
    n_before = len(d)
    d = d[d["DOS"] >= d["MIN_ELIG"]].copy()
    n_gated = n_before - len(d)

    daily   = d["RUNS_TYPE"] == "DAILY"
    sameday = d["RUNS_TYPE"] == "SAME_DAY"

    # DAILY portrait 1: DOS - min(7, lead)
    dly = d[daily]
    first_offset = np.minimum(7, dly["DAYS_SCHEDULED_BEFORE_DOS"].to_numpy())
    p1 = pd.DataFrame({
        "PATIENT_KEY": dly["PATIENT_KEY"].to_numpy(),
        "portrait_date": dly["DOS"].to_numpy() - first_offset.astype("timedelta64[D]"),
        "runs_type": "DAILY", "dos": dly["DOS"].to_numpy(),
        "days_scheduled_before_dos": dly["DAYS_SCHEDULED_BEFORE_DOS"].to_numpy(),
        "offset_days": first_offset, "portrait_seq": 1,
        "clip_applied": dly["DAYS_SCHEDULED_BEFORE_DOS"].to_numpy() < 7,
    })
    # DAILY portrait 2: DOS - 1
    p2 = pd.DataFrame({
        "PATIENT_KEY": dly["PATIENT_KEY"].to_numpy(),
        "portrait_date": dly["DOS"].to_numpy() - np.timedelta64(1, "D"),
        "runs_type": "DAILY", "dos": dly["DOS"].to_numpy(),
        "days_scheduled_before_dos": dly["DAYS_SCHEDULED_BEFORE_DOS"].to_numpy(),
        "offset_days": 1, "portrait_seq": 2, "clip_applied": False,
    })
    # SAME_DAY portrait: DOS
    sd = d[sameday]
    p3 = pd.DataFrame({
        "PATIENT_KEY": sd["PATIENT_KEY"].to_numpy(),
        "portrait_date": sd["DOS"].to_numpy(),
        "runs_type": "SAME_DAY", "dos": sd["DOS"].to_numpy(),
        "days_scheduled_before_dos": sd["DAYS_SCHEDULED_BEFORE_DOS"].to_numpy(),
        "offset_days": 0, "portrait_seq": 1, "clip_applied": False,
    })

    risk_events = pd.concat([p1, p2, p3], ignore_index=True)
    risk_events["source"] = "risk"
    # --- tag pre-window vs in-window ---
    risk_events["is_prewindow"] = risk_events["portrait_date"] < SCORING_START
    risk_events = risk_events.sort_values(["PATIENT_KEY", "portrait_date"]).reset_index(drop=True)

    if verbose:
        daily_n, sameday_n = int(daily.sum()), int(sameday.sum())
        print("=== Section 3 — risk portrait synthesis ===")
        print(f"rows gated out (DOS < MIN_ELIG):  {n_gated}")
        print(f"DAILY encounters:    {daily_n}  -> {daily_n*2} portraits")
        print(f"SAME_DAY encounters: {sameday_n} -> {sameday_n} portraits")
        print(f"total portraits:     {len(risk_events)}")
        print(f"  pre-window:  {int(risk_events['is_prewindow'].sum())}")
        print(f"  in-window:   {int((~risk_events['is_prewindow']).sum())}")
        print(f"clip applied (DAILY <7d lead): {int(risk_events['clip_applied'].sum())}")
        bad_after  = (risk_events["portrait_date"] > risk_events["dos"]).sum()
        print(f"portraits after DOS (should be 0): {bad_after}")

    return risk_events


def build_timeline(risk_events: pd.DataFrame) -> dict:
    """PATIENT_KEY -> sorted deduped portrait Timestamps (ALL portraits, incl pre-window)."""
    return (
        risk_events.groupby("PATIENT_KEY")["portrait_date"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )


risk_events = synth_risk_portraits(df)
risk_timeline = build_timeline(risk_events)
print(f"\npatients with >=1 portrait: {len(risk_timeline)}")
print("Section 3 OK.")

# %%
risk_timeline['15089154_1080']


# %%
# --- Section 3 addition: one representative portrait per encounter (QUALITY metric only) ---
# latest portrait per encounter: DOS-1 for DAILY, DOS for SAME_DAY.
# Prevents DAILY's two-portrait cluster (DOS-7, DOS-1) from injecting a fake ~6-day gap.
# NOTE: engines still cancel runs using the FULL risk_timeline (all real portraits).
def build_encounter_reps(df):
    d = df[["PATIENT_KEY", "DOS", "RUNS_TYPE", "MIN_ELIG"]].copy()
    d = d[d["DOS"] >= d["MIN_ELIG"]].copy()                        # same eligibility gate as synthesis
    d["rep_date"] = d["DOS"] - pd.to_timedelta((d["RUNS_TYPE"] == "DAILY").astype(int), unit="D")
    return (d.groupby("PATIENT_KEY")["rep_date"]
            .apply(lambda s: sorted(set(s)))
            .to_dict())

encounter_reps = build_encounter_reps(df)     # PATIENT_KEY -> sorted list of one-per-encounter reps
print(f"encounter_reps built for {len(encounter_reps)} patients")

# %%
import pandas as pd
from math import ceil

SIM_START     = pd.Timestamp("2025-06-01")
SIM_END       = pd.Timestamp("2026-06-01")      # exclusive; 365-day scoring clock
HISTORY_START = pd.Timestamp("2025-01-01")      # warm-up / data floor
SCORING_START = SIM_START
WINDOW_DAYS   = (SIM_END - SIM_START).days      # 365

CADENCE_INTERVAL  = {12: 31, 4: 92, 3: 122}
METHOD1_CELLS     = {12: [14, 30], 4: [14, 30, 90], 3: [14, 30, 90, 120]}
METHOD2_COOLDOWNS = [31, 92, 122]



# %%
"""
Section 4 — Simulation clock & Method-1 slot generator (MIN_ELIG-anchored)
==========================================================================
Patient Base is modeled as starting at each patient's eligibility (MIN_ELIG)
and stepping FORWARD by the cadence interval. So slots are PER-PATIENT:
  slot_k = anchor + k * interval,  anchor = MIN_ELIG,  k = 0,1,2,...
Slots before HISTORY_START are not simulated (no Risk data / not eligible).
Runs are COUNTED only in the scoring window [SCORING_START, SIM_END).

Consequence: MAX (scoring-slot count) varies per patient by eligibility phase
(e.g. cadence 12 -> 11 or 12). MAX is therefore per-patient; Section 7 reports
the mean MAX per cell. Cells are still grouped/compared by cadence.
"""


def patient_slots_method1(cadence: int, min_elig: pd.Timestamp) -> list:
    """
    MIN_ELIG-anchored forward slots, floored at HISTORY_START, up to SIM_END.
    Returns the full list (warm-up + scoring) in date order.
    """
    iv = CADENCE_INTERVAL[cadence]
    anchor = min_elig
    floor = max(HISTORY_START, anchor)           # first allowed simulated slot
    if anchor >= floor:
        first = anchor
    else:
        k = ceil((floor - anchor).days / iv)
        first = anchor + pd.Timedelta(days=iv * k)
    slots, d = [], first
    while d < SIM_END:
        slots.append(d)
        d += pd.Timedelta(days=iv)
    return slots


# Method 2 nominal MAX (continuous; handled fully in Section 6)
def build_max_table_method2(verbose=True):
    rows = [{"method": 2, "cadence": None, "cooldown": D, "max_runs_nominal": round(WINDOW_DAYS / D)}
            for D in METHOD2_COOLDOWNS]
    mt = pd.DataFrame(rows)
    if verbose:
        print("=== Section 4 — clock & generators ===")
        print(f"scoring clock: {SIM_START.date()} -> {SIM_END.date()} ({WINDOW_DAYS}d) | "
              f"warm-up floor {HISTORY_START.date()}")
        print("Method 1 slots are per-patient (MIN_ELIG-anchored); MAX computed in Section 5.")
        print("\nMethod 2 nominal MAX:")
        print(mt.to_string(index=False))
    return mt


m2_max_nominal = build_max_table_method2()
print("\nSection 4 OK.")

# %%
"""
Section 4b — Adjusted MAX
=========================
Population = patients with >= 1 encounter on/after SCORING_START (guaranteed >= 1
in-window Risk portrait). On cells where cadence == 365/cooldown, that portrait
reliably cancels exactly one run, so the theoretical MAX drops by 1.

  - Method 2: ALL cooldowns (MAX = 365/cooldown by definition)  -> MAX - 1
  - Method 1: ONLY diagonal twins where cadence == 365/cooldown  -> MAX - 1  (12/30, 4/90, 3/120)
  - Method 1 off-diagonal cells: unchanged (nominal cadence)
Applied at the PER-PATIENT level (each affected patient's max_runs = MAX-1).
"""

CADENCE_TWIN_COOLDOWN = {26: 14, 12: 30, 4: 90, 3: 120}     # cadence -> its 365/D cooldown

def method1_max(cadence, cooldown):
    return cadence - 1 if CADENCE_TWIN_COOLDOWN.get(cadence) == cooldown else cadence

def method2_max(cooldown):
    return round(WINDOW_DAYS / cooldown) - 1


# %%
import random

sampled_keys = random.sample(list(risk_timeline.keys()), 3)

test_timeline = {key: risk_timeline[key] for key in sampled_keys}

# test_timeline = {'2218388_16':risk_timeline['2218388_16']}



# %%
"""
Section 5b — Coverage-quality measures (corrected)
==================================================
One representative portrait per encounter (from encounter_reps). Reports:
  clean_*    : Risk reps only (baseline, before Patient Base) -- same across cells for a patient
  combined_* : Risk reps + fired PB runs (this cell)
Both window-anchored at both ends; measured strictly within [SIM_START, SIM_END).
Quality metric only -- engines cancel runs using ALL real portraits.
"""

import numpy as np

def _gaps(dates, start=SIM_START, end=SIM_END):
    pts = sorted(set(d for d in dates if start <= d < end))
    anchors = [start] + pts + [end]
    g = [(anchors[i + 1] - anchors[i]).days for i in range(len(anchors) - 1)]
    return float(np.mean(g)), int(max(g))

def quality_measures(rep_dates, pb_runs):
    clean_mean, clean_stale = _gaps(rep_dates)
    comb_mean, comb_stale = _gaps(list(rep_dates) + list(pb_runs))
    return clean_mean, clean_stale, comb_mean, comb_stale


# %%
"""
Section 5 — Method 1 engine  [NO-PB-WARMUP variant]
===================================================
Patient Base does NOT run during warm-up, so its clock starts at SCORING_START
(June 1): slots are June-1-anchored (same for every patient), stepping forward
by the cadence interval. Risk portraits (incl. pre-June) still seed freshness,
so a fresh Risk portrait can still skip the first in-window run.
MAX = nominal cadence (12/4/3).
"""
import pandas as pd
import numpy as np
from bisect import insort, bisect_right


# ---- quality helper (defined before the engine uses it) ----
def _gaps(dates, start=SIM_START, end=SIM_END):
    pts = sorted(set(d for d in dates if start <= d < end))
    anchors = [start] + pts + [end]
    g = [(anchors[i + 1] - anchors[i]).days for i in range(len(anchors) - 1)]
    return float(np.mean(g)), int(max(g))

def quality_measures(rep_dates, pb_runs):
    clean_mean, clean_stale = _gaps(rep_dates)
    comb_mean, comb_stale = _gaps(list(rep_dates) + list(pb_runs))
    return clean_mean, clean_stale, comb_mean, comb_stale


# ---- June-1-anchored slots (NO warm-up; same for all patients) ----
def scoring_slots_method1(cadence):
    iv = CADENCE_INTERVAL[cadence]
    out, d = [], SCORING_START
    while d < SIM_END:
        out.append(d)
        d += pd.Timedelta(days=iv)
    return out


# ---- freshness check ----
def _fresh_exists(sorted_dates, S, D):
    lo = bisect_right(sorted_dates, S - pd.Timedelta(days=D))
    hi = bisect_right(sorted_dates, S)
    return hi > lo


# ---- per-patient walk (slots start at June 1; no PB runs before then) ----
def run_method1_patient(slots, D, all_portrait_dates, pid=None, cadence=None, audit=None):
    history = sorted(all_portrait_dates)          # Risk portraits (incl pre-window) seed freshness
    fired = []
    for S in slots:
        if S < SCORING_START:                     # safety: June-anchored slots never trip this
            continue
        is_fresh = _fresh_exists(history, S, D)
        if is_fresh:
            decision = "skip"
        else:
            decision = "fire"
            insort(history, S)                    # in-window PB runs join freshness
            fired.append(S)
        if audit is not None:
            covering = history[bisect_right(history, S) - 1] if is_fresh else None
            audit.append({"PATIENT_KEY": pid, "cadence": cadence, "cooldown": D,
                          "slot_date": S, "phase": "scoring", "decision": decision,
                          "counted": (decision == "fire"),
                          "covering_portrait": covering})
    return fired


# ---- driver over all patients x cells ----
def compute_method1(risk_timeline, encounter_reps, method1_cells,
                    trace_n=50, verbose=True):
    patients = list(risk_timeline.keys())
    trace_set = set(patients[:trace_n]) if trace_n is not None else set(patients)

    event_rows, actual_rows, audit_rows = [], [], []

    for cadence, cooldowns in method1_cells.items():
        slots = scoring_slots_method1(cadence)                # June-1-anchored, same for all patients
        for D in cooldowns:
            max_runs = cadence                                # nominal MAX (12/4/3)
            for pid in patients:
                all_dates = risk_timeline[pid]

                aud = audit_rows if pid in trace_set else None
                fired = run_method1_patient(slots, D, all_dates,
                                            pid=pid, cadence=cadence, audit=aud)
                actual = len(fired)

                reps = encounter_reps.get(pid, [])
                clean_mean, clean_stale, comb_mean, comb_stale = quality_measures(reps, fired)

                actual_rows.append({"PATIENT_KEY": pid, "cadence": cadence, "cooldown": D,
                                    "max_runs": max_runs, "actual_runs": actual,
                                    "saved_runs": max_runs - actual,
                                    "clean_mean_gap": clean_mean, "clean_max_staleness": clean_stale,
                                    "mean_gap": comb_mean, "max_staleness": comb_stale})
                for run_date in fired:
                    event_rows.append({"PATIENT_KEY": pid, "method": 1, "cadence": cadence,
                                       "cooldown": D, "run_date": run_date})

    m1_events = pd.DataFrame(event_rows)
    m1_actual = pd.DataFrame(actual_rows)
    m1_audit  = pd.DataFrame(audit_rows)

    if verbose:
        print("=== Section 5 — Method 1 engine [NO-PB-WARMUP] ===")
        print(f"patients: {len(patients)} | traced: {len(trace_set)}")
        summary = (m1_actual.groupby(["cadence", "cooldown"])
                   .agg(max_runs=("max_runs", "first"),
                        mean_actual=("actual_runs", "mean"),
                        mean_saved=("saved_runs", "mean"),
                        clean_gap=("clean_mean_gap", "mean"),
                        comb_gap=("mean_gap", "mean"),
                        clean_stale=("clean_max_staleness", "mean"),
                        comb_stale=("max_staleness", "mean")).round(2))
        print(summary.to_string())
        print(f"\nPB run rows: {len(m1_events)} | audit rows: {len(m1_audit)}")

    return m1_events, m1_actual, m1_audit


# ---- run (produces the output) ----
m1_events, m1_actual, m1_audit = compute_method1(
    risk_timeline, encounter_reps, METHOD1_CELLS, trace_n=50)
print("\nSection 5 OK.")

# %%
risk_events[risk_events['PATIENT_KEY']=='428864_16']

# %%
risk_timeline['277565_42']

# %%
m1_audit.sort_values(['PATIENT_KEY','cadence','cooldown'])

# %%
m1_events.to_parquet('./analysis_output_data/m1_events_no_pb_in_warmpup_20260702.parquet')
m1_actual.to_parquet('./analysis_output_data/m1_actual_no_pb_in_warmpup_20260702.parquet')
m1_audit.to_parquet('./analysis_output_data/m1_audit_no_pb_in_warmpup_20260702.parquet')

# %%
m1_events = pd.read_parquet('./analysis_output_data/m1_events_no_pb_in_warmpup_20260702.parquet')
m1_actual = pd.read_parquet('./analysis_output_data/m1_actual_no_pb_in_warmpup_20260702.parquet')
m1_audit = pd.read_parquet('./analysis_output_data/m1_audit_no_pb_in_warmpup_20260702.parquet')

# %%
risk_timeline['00001141_41']

# %%
m1_audit[m1_audit['PATIENT_KEY']=='00001141_41']

# %%
m1_actual.groupby(['cadence','cooldown'])[['actual_runs','mean_gap','max_staleness']].mean()

# %%
m1_actual.groupby(['cadence','cooldown'])[['actual_runs','mean_gap','max_staleness']].mean()

# %%
"""
Section 6 — Method 2 engine  [NO-PB-WARMUP variant]
===================================================
Continuous walk starts at SCORING_START (June 1), not the warm-up start, so no
PB run is generated before June. Risk portraits (incl. pre-June) still seed
freshness, so the first in-window check defers to a fresh Risk portrait.
MAX = nominal 365/D (26/12/4/3). saved_runs left unclamped (can be negative).
"""
import pandas as pd
from bisect import insort, bisect_right


def run_method2_patient(D, all_portrait_dates, warmup_start, pid=None, audit=None):
    history = sorted(all_portrait_dates)          # Risk portraits (incl pre-window) seed freshness
    fired = []
    t = SCORING_START                             # start walk at June 1 (ignore warmup_start)
    while t < SIM_END:
        lo = bisect_right(history, t - pd.Timedelta(days=D))
        hi = bisect_right(history, t)
        is_fresh = hi > lo
        if is_fresh:
            covering = history[hi - 1]
            if audit is not None:
                audit.append({"PATIENT_KEY": pid, "cooldown": D, "event_date": t,
                              "phase": "scoring", "decision": "skip", "counted": False,
                              "covering_portrait": covering})
            t = covering + pd.Timedelta(days=D)
        else:
            fired.append(t)
            insort(history, t)
            if audit is not None:
                audit.append({"PATIENT_KEY": pid, "cooldown": D, "event_date": t,
                              "phase": "scoring", "decision": "fire", "counted": True,
                              "covering_portrait": None})
            t = t + pd.Timedelta(days=D)
    return fired


def compute_method2(risk_timeline, encounter_reps, patient_min_elig, cooldowns,
                    trace_n=50, verbose=True):
    patients = list(risk_timeline.keys())
    trace_set = set(patients[:trace_n]) if trace_n is not None else set(patients)

    event_rows, actual_rows, audit_rows = [], [], []

    for D in cooldowns:
        max_runs = round(WINDOW_DAYS / D)                         # nominal MAX (26/12/4/3)
        for pid in patients:
            warmup    = max(patient_min_elig[pid], HISTORY_START)  # kept for signature; unused in walk
            all_dates = risk_timeline[pid]

            aud = audit_rows if pid in trace_set else None
            fired = run_method2_patient(D, all_dates, warmup, pid=pid, audit=aud)
            actual = len(fired)

            reps = encounter_reps.get(pid, [])
            clean_mean, clean_stale, comb_mean, comb_stale = quality_measures(reps, fired)

            actual_rows.append({"PATIENT_KEY": pid, "cooldown": D,
                                "max_runs": max_runs, "actual_runs": actual,
                                "saved_runs": max_runs - actual,
                                "clean_mean_gap": clean_mean, "clean_max_staleness": clean_stale,
                                "mean_gap": comb_mean, "max_staleness": comb_stale})
            for run_date in fired:
                event_rows.append({"PATIENT_KEY": pid, "method": 2,
                                   "cooldown": D, "run_date": run_date})

    m2_events = pd.DataFrame(event_rows)
    m2_actual = pd.DataFrame(actual_rows)
    m2_audit  = pd.DataFrame(audit_rows)

    if verbose:
        print("=== Section 6 — Method 2 engine [NO-PB-WARMUP] ===")
        print(f"patients: {len(patients)} | traced: {len(trace_set)}")
        summary = (m2_actual.groupby("cooldown")
                   .agg(max_runs=("max_runs", "first"),
                        mean_actual=("actual_runs", "mean"),
                        mean_saved=("saved_runs", "mean"),
                        pct_overrun=("saved_runs", lambda s: round((s < 0).mean() * 100, 1)),
                        clean_gap=("clean_mean_gap", "mean"),
                        comb_gap=("mean_gap", "mean"),
                        clean_stale=("clean_max_staleness", "mean"),
                        comb_stale=("max_staleness", "mean")).round(2))
        print(summary.to_string())
        print(f"\nPB run rows: {len(m2_events)} | audit rows: {len(m2_audit)}")

    return m2_events, m2_actual, m2_audit

patient_min_elig = df.groupby("PATIENT_KEY")["MIN_ELIG"].min().to_dict()


# ---- run (produces the output) ----
m2_events, m2_actual, m2_audit = compute_method2(
    risk_timeline, encounter_reps, patient_min_elig, METHOD2_COOLDOWNS, trace_n=50)
print("\nSection 6 OK.")

# %%
risk_timeline['00001141_41']

# %%
m2_audit[m2_audit['PATIENT_KEY']=='00001141_41']

# %%
m2_events.to_parquet('./analysis_output_data/m2_events_no_pb_in_warmpup_20260702.parquet')
m2_actual.to_parquet('./analysis_output_data/m2_actual_no_pb_in_warmpup_20260702.parquet')
m2_audit.to_parquet('./analysis_output_data/m2_audit_no_pb_in_warmpup_20260702.parquet')

# %%
"""
Section 7 — Savings aggregation (per cell)
==========================================
Stack Method 1 and Method 2 per-patient results, then aggregate to one row per cell.

Reports BOTH definitions of saved %:
  saved_pct_pooled = sum(saved_runs) / sum(max_runs)      <- headline (total saved / total possible)
  saved_pct_mean   = mean(saved_runs / max_runs)          <- average patient

Plus mean actual, mean saved (#), and the quality measures (mean_gap, max_staleness).
For Method 2, saved can be negative (over-run); pct_overrun reports how often.
"""

import pandas as pd
import numpy as np


def aggregate_cell(g):
    n = len(g)
    sum_max    = g["max_runs"].sum()
    sum_actual = g["actual_runs"].sum()
    sum_saved  = g["saved_runs"].sum()
    
    return pd.Series({
        "n_patients":       n,
        "max_runs":         g["max_runs"].iloc[0],          # fixed within a cell
        "mean_actual":      g["actual_runs"].mean(),
        "mean_saved":       g["saved_runs"].mean(),
        "saved_pct_pooled": sum_saved / sum_max if sum_max else np.nan,
        "saved_pct_mean":   (g["saved_runs"] / g["max_runs"]).mean(),
        "pct_overrun":      (g["saved_runs"] < 0).mean() * 100,   # share of patients over theoretical max
        "mean_gap":         g["mean_gap"].mean(),
        "clean_mean_gap":      g["clean_mean_gap"].mean(),
        "max_staleness":    g["max_staleness"].mean(),
        "clean_max_staleness": g["clean_max_staleness"].mean(),
    })


def build_savings_table(m1_actual, m2_actual, verbose=True):
    m1 = m1_actual.copy()
    m1["method"] = 1
    m2 = m2_actual.copy()
    m2["method"] = 2
    m2["cadence"] = (365 / m2["cooldown"]).round().astype(int)   # 14->26, 30->12, 90->4, 120->3

    cols = ["method", "cadence", "cooldown",
            "max_runs", "actual_runs", "saved_runs", "mean_gap","clean_mean_gap", "max_staleness",'clean_max_staleness']
    stacked = pd.concat([m1[cols], m2[cols]], ignore_index=True)

    agg = (stacked.groupby(["method", "cadence", "cooldown"])
           .apply(aggregate_cell)
           .reset_index())

    # tidy types / rounding
    agg["n_patients"] = agg["n_patients"].astype(int)
    agg["max_runs"]   = agg["max_runs"].astype(int)
    for c in ["mean_actual", "mean_saved", "mean_gap", "max_staleness"]:
        agg[c] = agg[c].round(2)
    for c in ["saved_pct_pooled", "saved_pct_mean"]:
        agg[c] = (agg[c] * 100).round(1)     # as %
    agg["pct_overrun"] = agg["pct_overrun"].round(1)

    agg = agg.sort_values(["cadence", "method", "cooldown"],
                          ascending=[False, True, True]).reset_index(drop=True)

    if verbose:
        print("=== Section 7 — savings per cell ===")
        show = ["method", "cadence", "cooldown", "n_patients", "max_runs",
                "mean_actual", "mean_saved", "saved_pct_pooled", "saved_pct_mean",
                "pct_overrun", "mean_gap","clean_mean_gap","max_staleness",'clean_max_staleness']
        print(agg[show].to_string(index=False))

    return agg


savings_table = build_savings_table(m1_actual, m2_actual)
print("\nSection 7 OK.")

# %%
savings_table.to_csv('./res.csv')

# %%
risk_timeline["00000895_41"]

# %%
m1_audit[m1_audit['PATIENT_KEY']=='00000895_41']

# %%
df[df['PATIENT_KEY']=='00001046_41']

# %%
risk_events[risk_events['PATIENT_KEY']=='00001046_41']

# %%
m1_audit[(m1_audit['PATIENT_KEY']=='00001046_41')&(m1_audit['cadence']==4)&(m1_audit['cooldown']==90)]

# %%
m2_actual

# %%
"""
Section 8 — Comparison table (grouped by cadence)
=================================================
Reshape the per-cell savings_table into the deliverable view: one block per
cadence group, Method 1 cooldown variants beside the single Method 2 cell,
with the diagonal twin (M1 cooldown == M2 cooldown) marked. Cost columns
(max, actual, saved, saved%) sit beside quality columns (mean_gap, max_staleness)
so the run-cost-vs-freshness trade-off is visible within each group.

headline_pct: "pooled" (sum saved / sum max) or "mean" (avg per-patient %).
"""

import pandas as pd

# the Method 2 cooldown that matches each cadence (the diagonal twin key)
CADENCE_TWIN_COOLDOWN = {26: 14, 12: 30, 4: 90, 3: 120}


def build_comparison(savings_table, headline_pct="pooled", verbose=True):
    t = savings_table.copy()
    pct_col = "saved_pct_pooled" if headline_pct == "pooled" else "saved_pct_mean"

    t["is_diagonal_twin"] = t.apply(
        lambda r: r["method"] == 1 and r["cooldown"] == CADENCE_TWIN_COOLDOWN.get(r["cadence"]),
        axis=1)
    t["method_label"] = t["method"].map({1: "M1 (cadence)", 2: "M2 (continuous)"})

    t = t.sort_values(["cadence", "method", "cooldown"],
                      ascending=[False, True, True]).reset_index(drop=True)

    show = ["cadence", "method_label", "cooldown", "is_diagonal_twin", "max_runs",
            "mean_actual", "mean_saved", pct_col, "pct_overrun", "mean_gap", "max_staleness"]
    disp = t[show].rename(columns={pct_col: "saved_pct", "is_diagonal_twin": "twin"})

    if verbose:
        print(f"=== Section 8 — comparison by cadence (headline %={headline_pct}) ===\n")
        for cad in sorted(disp["cadence"].unique(), reverse=True):
            grp = disp[disp["cadence"] == cad].drop(columns=["cadence"]).copy()
            grp["twin"] = grp["twin"].map({True: "<-- twin", False: ""})
            print(f"--- Cadence {cad} (~{cad} runs/yr) ---")
            print(grp.to_string(index=False))
            print()

    return t   # full table (with twin flag + labels) for Section 9 / export


comparison_table = build_comparison(savings_table, headline_pct="pooled")
print("Section 8 OK.")

# %%
m1_actual[]

# %%
import pandas as pd
import numpy as np

VISIT_SHARE, NOVISIT_SHARE = 0.82, 0.18

# interval per cell = the no-visit patient's gap AND max staleness
#   Method 1: cadence interval (31/92/122)   Method 2: cooldown D
CAD_IV = {12: 31, 4: 92, 3: 122}

def _novisit_interval(method, cadence, cooldown):
    return CAD_IV[cadence] if method == 1 else cooldown

wp = savings_table.copy()

# ---- COST: no-visit patients run the full theoretical max, save 0 ----
wp["wp_actual"]    = VISIT_SHARE * wp["mean_actual"] + NOVISIT_SHARE * wp["max_runs"]
wp["wp_saved"]     = wp["max_runs"] - wp["wp_actual"]
wp["wp_saved_pct"] = wp["wp_saved"] / wp["max_runs"] * 100

# ---- QUALITY: no-visit gap == max staleness == interval ----
wp["_nv"] = wp.apply(lambda r: _novisit_interval(r["method"], r["cadence"], r["cooldown"]), axis=1)
wp["wp_mean_gap"]      = VISIT_SHARE * wp["mean_gap"]      + NOVISIT_SHARE * wp["_nv"]
wp["wp_max_staleness"] = VISIT_SHARE * wp["max_staleness"] + NOVISIT_SHARE * wp["_nv"]

# ---- tidy ----
for c in ["wp_actual", "wp_saved", "wp_mean_gap", "wp_max_staleness"]:
    wp[c] = wp[c].round(2)
wp["wp_saved_pct"] = wp["wp_saved_pct"].round(1)
wp = wp.drop(columns=["_nv"])
wp["wp_mean_runs"] = 8.4     # current whole-pop runs/patient/yr, no PB

# 1. PB addition, absolute (you already have this as wp_actual — the added runs)
wp["pb_add_abs"] = wp["wp_actual"]

# 2. PB addition as a fraction of current run volume
wp["pb_add_frac"] = wp["wp_actual"] / wp["wp_mean_runs"] * 100      # %

# 3. PB *max* addition (theoretical, if Risk saved nothing) as a fraction
wp["pb_max_frac"] = wp["max_runs"] / wp["wp_mean_runs"] * 100       # %


print(wp[["method", "cadence", "cooldown", "max_runs","wp_mean_runs",
          "mean_actual", "wp_actual","pb_add_abs", "pb_add_frac","pb_max_frac","wp_saved", "wp_saved_pct",
          "mean_gap", "wp_mean_gap", "max_staleness", "wp_max_staleness"]].to_string(index=False))

# %%
wp.to_csv('./final_res_table_for_deck.csv')

# %%
import matplotlib.pyplot as plt
import numpy as np

def cost_bars(df_method, method_num, title):
    d = df_method.sort_values(["cadence", "cooldown"], ascending=[False, True]).copy()
    d = d[d['cadence']!=26]
    labels = [f"cad {int(c)}\n{int(cd)}d" for c, cd in zip(d["cadence"], d["cooldown"])]
    x = np.arange(len(d)); w = 0.6

    fig, ax = plt.subplots(figsize=(max(7, len(d)*1.1), 5.5))
    # faded = theoretical max addition; solid = actual addition
    ax.bar(x, d["pb_max_frac"], w, color="#c9d9ee", label="max PB runs addition[%] (no Risk runs)")
    ax.bar(x, d["pb_add_frac"], w, color="#2a78d6", label="actual PB runs addition[%]")

    for xi, (act, mx, absn) in enumerate(zip(d["pb_add_frac"], d["pb_max_frac"], d["pb_add_abs"])):
        ax.text(xi, act + 1, f"{act:.0f}%", ha="center", va="bottom", fontsize=10, color="#1a5296")
        ax.text(xi, mx + 1, f"{mx:.0f}%", ha="center", va="bottom", fontsize=9, color="#8aa")
        ax.text(xi, 1.5, f"+{absn:.1f}", ha="center", va="bottom", fontsize=11, color="white")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("PB run addition (% of current 8.4 runs/yr)")
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    ax.grid(True, axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, max(d["pb_max_frac"]) * 1.2)
    plt.tight_layout(); plt.show()

cost_bars(wp[wp.method == 1], 1, "Method 1(Anchor Dates Based) — PB run addition[%] per variant")
cost_bars(wp[wp.method == 2], 2, "Method 2(Freshness Based ) — PB run addition[%] per variant")

# %%
import matplotlib.pyplot as plt
import numpy as np

def tradeoff_bars(df_method, title, x_is_cooldown_only=False):
    d = df_method.sort_values(["cadence", "cooldown"], ascending=[False, True]).copy()
    d = d[d['cadence']!=26]

    if x_is_cooldown_only:   # Method 2: label by cooldown only
        labels = [f"{int(cd)}d" for cd in d["cooldown"]]
    else:                    # Method 1: cadence + cooldown
        labels = [f"cad {int(c)}\n{int(cd)}d" for c, cd in zip(d["cadence"], d["cooldown"])]

    x = np.arange(len(d)); w = 0.38
    fig, ax1 = plt.subplots(figsize=(max(7, len(d)*1.3), 5.5))
    ax2 = ax1.twinx()

    # left axis: % PB addition (cost)
    b1 = ax1.bar(x - w/2, d["pb_add_frac"], w, color="#2a78d6", label="PB addition (% of current runs)")
    # right axis: whole-pop max staleness (quality risk)
    b2 = ax2.bar(x + w/2, d["wp_max_staleness"], w, color="#eb6834", label="worst-case staleness (days)")

    for xi, v in zip(x, d["pb_add_frac"]):
        ax1.text(xi - w/2, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color="#1a5296")
    for xi, v in zip(x, d["wp_max_staleness"]):
        ax2.text(xi + w/2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#a5461f")

    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("PB run addition (% of current 8.4 runs/yr)", color="#2a78d6")
    ax2.set_ylabel("worst-case staleness (days)", color="#eb6834")
    ax1.tick_params(axis="y", labelcolor="#2a78d6")
    ax2.tick_params(axis="y", labelcolor="#eb6834")
    ax1.set_ylim(0, max(d["pb_add_frac"]) * 1.25)
    ax2.set_ylim(0, max(d["wp_max_staleness"]) * 1.25)
    ax1.set_title(title, fontsize=12)
    ax1.grid(True, axis="y", color="#e1e0d9", lw=0.6); ax1.set_axisbelow(True)

    # combined legend
    ax1.legend(handles=[b1, b2], labels=["PB addition (%)", "worst-case staleness (days)"],
               loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout(); plt.show()

tradeoff_bars(wp[wp.method == 1], "Method 1 — cost vs staleness per variant (whole population)")
tradeoff_bars(wp[wp.method == 2], "Method 2 — cost vs staleness per variant (whole population)",
              x_is_cooldown_only=True)

# %%
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from adjustText import adjust_text

CAD_COLOR = {12: "#2a78d6", 4: "#1baf7a", 3: "#eb6834"}
VISIT, NOVISIT = 0.82, 0.18
WP_BASE_STALE = VISIT * savings_table["clean_max_staleness"].iloc[0] + NOVISIT * 365

def scatter_method(df_method, method_num, method_name):
    d = df_method[~((df_method["method"] == 2) & (df_method["cooldown"] == 14))].copy()

    fig, ax = plt.subplots(figsize=(8.5, 6))

    texts = []
    for _, r in d.iterrows():
        ax.scatter(r["pb_add_frac"], r["wp_max_staleness"],       # x = % PB addition
                   c=CAD_COLOR[r["cadence"]], s=140,
                   edgecolors="white", linewidths=0.9, zorder=3)
        texts.append(ax.text(r["pb_add_frac"], r["wp_max_staleness"],
                             f'cad {int(r["cadence"])} / {int(r["cooldown"])}d',
                             fontsize=8.5, color="#555"))

    ax.axhline(WP_BASE_STALE, ls="--", color="#898781", lw=1.4, zorder=1)
    ax.annotate(f"No Patient Base ({WP_BASE_STALE:.0f}d)",
                (ax.get_xlim()[1], WP_BASE_STALE), textcoords="offset points",
                xytext=(-6, 5), ha="right", fontsize=9, color="#898781")

    adjust_text(texts, ax=ax, only_move={"text": "xy"},
                arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6))

    ax.set_xlabel("PB run Addition (% of current mean runs per patient/yr)")
    ax.set_ylabel("Worst-case Portrait Staleness (days)")
    ax.set_title(f"Method {method_num} ({method_name}) — Cost vs Worst-case portrait staleness",
                 fontsize=12)
    ax.set_xlim(0, max(d["pb_add_frac"]) * 1.2)
    ax.set_ylim(0, 240)                                          # >150, fits baseline ~228
    ax.grid(True, color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)

    cad_handles = [mlines.Line2D([], [], color=c, marker="o", ls="", ms=9, label=f"cadence {cad}")
                   for cad, c in CAD_COLOR.items() if cad in d["cadence"].values]
    ax.legend(handles=cad_handles, loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(0, 150)
    plt.tight_layout(); plt.show()

scatter_method(wp[wp.method == 1], 1, "Anchor Dates Based")
scatter_method(wp[wp.method == 2], 2, "Freshness Based")

# %%
# Updated: added timestamp to output
from datetime import datetime
print(f"Analysis last run: {datetime.now()}")
