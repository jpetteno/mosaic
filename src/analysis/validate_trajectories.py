"""
MOBILITY THEORY VALIDATOR
=========================
Tests simulated trajectories against established mobility science literature.

Standard metrics (POI-level):
  1. Distance distribution          -- Gonzalez et al. (2008)
  2. Radius of gyration (RoG)       -- Gonzalez et al. (2008), Pappalardo et al. (2013)
  3. Revisit rate                   -- Pappalardo et al. (2015)
  4. Top-N POI concentration        -- Pappalardo et al. (2015)
  5. Zipf's law (POI popularity)    -- Noulas et al. (2012)
  6. Temporal activity patterns     -- Song et al. (2010)
  7. Activity entropy               -- Song et al. (2010)
  8. Inter-visit time distribution  -- Barabasi (2010)

Grid-level metrics (cell-level, additive -- requires --use_grid):
  G1. Cell Zipf's law               -- grid-level analogue of Noulas et al. (2012)
  G2. Cell entropy                  -- grid-level analogue of Song et al. (2010)
  G3. Cell revisit rate             -- grid-level analogue of Pappalardo et al. (2015)
  G4. Cell flow power law           -- gravity model analogue, Simini et al. (2012)

Usage:
  # Standard (POI-level only):
  python validate_trajectories.py \
      --activities results/data/activity_log.csv \
      --travels    results/data/travel_log.csv \
      --output     validation_output

  # With grid metrics (requires sim.py --use_grid to have been run first):
  python validate_trajectories.py \
      --activities results/data/activity_log.csv \
      --travels    results/data/travel_log.csv \
      --output     validation_output \
      --use_grid \
      --grid_dir   results/grid_data

  (--grid_dir defaults to ../grid_data relative to --activities if omitted)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import entropy

# -- Plot style ----------------------------------------------------------------
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

SEP  = "=" * 70
THIN = "-" * 70

ACTIVITY_COLORS = {
    'HOME':    '#1f77b4',
    'WORK':    '#ff7f0e',
    'FOOD':    '#2ca02c',
    'ERRAND':  '#d62728',
    'LEISURE': '#9467bd',
    'TRAVEL':  '#7f7f7f',
}


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_data(activities_path: str, travels_path: str):
    acts  = pd.read_csv(activities_path)
    travs = pd.read_csv(travels_path)

    acts['start_time']   = pd.to_datetime(acts['start_time'],  errors='coerce')
    acts['end_time']     = pd.to_datetime(acts['end_time'],    errors='coerce')
    acts['duration_min'] = pd.to_numeric(acts['duration_min'], errors='coerce')

    travs['duration_min'] = pd.to_numeric(travs['duration_min'], errors='coerce')
    travs['distance_km']  = pd.to_numeric(travs['distance_km'],  errors='coerce')

    acts  = acts[acts['duration_min']  > 0].copy()
    travs = travs[travs['distance_km'] > 0].copy()

    print(f"  Loaded {len(acts):,} activity records  ({acts['sim_id'].nunique()} agents)")
    print(f"  Loaded {len(travs):,} travel records")
    return acts, travs


def load_grid_data(grid_dir: str):
    """
    Load grid-snapped CSVs produced by sim.py --use_grid.
    Raises FileNotFoundError with a clear message if they are missing.
    """
    paths = {
        'acts_grid':  os.path.join(grid_dir, "data", "activity_log_grid.csv"),
        'travs_grid': os.path.join(grid_dir, "data", "travel_log_grid.csv"),
        'vc_df':      os.path.join(grid_dir, "data", "cell_visit_counts.csv"),
        'flow_df':    os.path.join(grid_dir, "data", "flow_matrix.csv"),
    }
    for key, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Grid data file not found: {p}\n"
                f"Run sim.py with --use_grid first to generate grid_data/."
            )

    acts_grid  = pd.read_csv(paths['acts_grid'])
    travs_grid = pd.read_csv(paths['travs_grid'])
    vc_df      = pd.read_csv(paths['vc_df'])
    flow_df    = pd.read_csv(paths['flow_df'])

    acts_grid['start_time']   = pd.to_datetime(acts_grid['start_time'], errors='coerce')
    acts_grid['duration_min'] = pd.to_numeric(acts_grid['duration_min'], errors='coerce')

    print(f"  [Grid] {len(acts_grid):,} activity records  "
          f"({acts_grid['sim_id'].nunique()} agents, "
          f"{vc_df['cell_id'].nunique()} unique cells)")
    print(f"  [Grid] {len(travs_grid):,} travel records  "
          f"({len(flow_df):,} O-D cell pairs)")

    return acts_grid, travs_grid, vc_df, flow_df


# =============================================================================
# METRIC 1 -- Distance Distribution (Gonzalez 2008)
# =============================================================================

def validate_distance_distribution(travs: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  1. DISTANCE DISTRIBUTION  --  Gonzalez et al. (2008)")
    print(SEP)

    d = travs['distance_km'].dropna().values
    d = d[d > 0]

    pct_2km  = (d <= 2.0).sum()  / len(d) * 100
    pct_5km  = (d <= 5.0).sum()  / len(d) * 100
    pct_10km = (d <= 10.0).sum() / len(d) * 100

    tail = d[d >= 1.0]
    try:
        fit_result = stats.powerlaw.fit(tail, floc=0)
        alpha = fit_result[0]
    except Exception:
        alpha = float('nan')

    passes = pct_10km >= 70.0

    print(f"  Mean distance   : {d.mean():.2f} km")
    print(f"  Median distance : {np.median(d):.2f} km")
    print(f"  <= 2 km          : {pct_2km:.1f}%")
    print(f"  <= 5 km          : {pct_5km:.1f}%")
    print(f"  <= 10 km         : {pct_10km:.1f}%  (literature target: >=80%)")
    print(f"  Power-law alpha : {alpha:.3f}        (literature: ~1.75)")
    print(f"  Status          : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Distance Distribution', reference='Gonzalez 2008',
        mean_km=float(d.mean()), median_km=float(np.median(d)),
        pct_under_2km=float(pct_2km), pct_under_5km=float(pct_5km),
        pct_under_10km=float(pct_10km), powerlaw_alpha=float(alpha),
        literature_threshold=80.0, passes=passes, distances=d
    )


# =============================================================================
# METRIC 2 -- Radius of Gyration (Gonzalez 2008)
# =============================================================================

def validate_radius_of_gyration(acts: pd.DataFrame, travs: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  2. RADIUS OF GYRATION  --  Gonzalez et al. (2008)")
    print(SEP)

    rogs = []
    unique_pois_per_agent = []

    for sim_id, agent_travs in travs.groupby('sim_id'):
        d = agent_travs['distance_km'].dropna().values
        if len(d) < 2:
            continue
        rog = np.sqrt(np.mean(d ** 2))
        rogs.append(rog)

        agent_acts = acts[acts['sim_id'] == sim_id]
        unique_pois_per_agent.append(agent_acts['location_id'].nunique())

    mean_rog = np.mean(rogs)
    passes = 3.0 <= mean_rog <= 25.0

    print(f"  Mean RoG         : {mean_rog:.2f} km")
    print(f"  Median RoG       : {np.median(rogs):.2f} km")
    print(f"  Std RoG          : {np.std(rogs):.2f} km")
    print(f"  Literature range : 5-20 km (typical urban)")
    print(f"  Status           : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Radius of Gyration', reference='Gonzalez 2008',
        mean_rog_km=float(mean_rog), median_rog_km=float(np.median(rogs)),
        std_rog_km=float(np.std(rogs)),
        literature_range=(5.0, 20.0), passes=passes,
        rogs=rogs, unique_pois_per_agent=unique_pois_per_agent
    )


# =============================================================================
# METRIC 3 -- Revisit Rate (Pappalardo 2015)
# =============================================================================

def validate_revisit_rate(acts: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  3. REVISIT RATE  --  Pappalardo et al. (2015)")
    print(SEP)

    non_home = acts[acts['activity_type'] != 'HOME'].copy()
    revisit_rates = []

    for sim_id, agent_acts in non_home.groupby('sim_id'):
        agent_acts = agent_acts.sort_values('start_time')
        seen, revisits, total = set(), 0, 0
        for loc in agent_acts['location_id'].dropna():
            if loc in seen:
                revisits += 1
            seen.add(loc)
            total += 1
        if total > 0:
            revisit_rates.append(revisits / total)

    mean_revisit = np.mean(revisit_rates)
    passes = mean_revisit >= 0.60

    print(f"  Mean revisit rate  : {mean_revisit*100:.1f}%")
    print(f"  Median revisit rate: {np.median(revisit_rates)*100:.1f}%")
    print(f"  Literature target  : ~70% (Pappalardo 2015)")
    print(f"  Status             : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Revisit Rate', reference='Pappalardo 2015',
        mean_revisit_rate=float(mean_revisit),
        median_revisit_rate=float(np.median(revisit_rates)),
        literature_target=0.70, passes=passes,
        revisit_rates=revisit_rates
    )


# =============================================================================
# METRIC 4 -- Top-N POI Concentration (Pappalardo 2015)
# =============================================================================

def validate_topn_concentration(acts: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  4. TOP-N POI CONCENTRATION  --  Pappalardo et al. (2015)")
    print(SEP)

    non_home = acts[acts['activity_type'] != 'HOME'].dropna(subset=['location_id'])
    per_agent = []

    for sim_id, agent_acts in non_home.groupby('sim_id'):
        counts = agent_acts['location_id'].value_counts()
        total  = len(agent_acts)
        if total == 0:
            continue
        top1 = counts.iloc[0]  / total * 100 if len(counts) >= 1 else 0
        top3 = counts.iloc[:3].sum() / total * 100 if len(counts) >= 3 else counts.sum() / total * 100
        top5 = counts.iloc[:5].sum() / total * 100 if len(counts) >= 5 else counts.sum() / total * 100
        per_agent.append(dict(sim_id=sim_id, top1=top1, top3=top3, top5=top5,
                              unique_pois=len(counts), total_visits=total))

    df = pd.DataFrame(per_agent)
    mean_top3 = df['top3'].mean()
    passes = 55.0 <= mean_top3 <= 85.0

    print(f"  Mean Top-1 concentration : {df['top1'].mean():.1f}% +/- {df['top1'].std():.1f}%")
    print(f"  Mean Top-3 concentration : {mean_top3:.1f}% +/- {df['top3'].std():.1f}%")
    print(f"  Mean Top-5 concentration : {df['top5'].mean():.1f}% +/- {df['top5'].std():.1f}%")
    print(f"  Mean unique POIs/agent   : {df['unique_pois'].mean():.1f}")
    print(f"  Literature target        : Top-3 captures 60-80% (Pappalardo 2015)")
    print(f"  Status                   : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Top-N Concentration', reference='Pappalardo 2015',
        mean_top1=float(df['top1'].mean()), mean_top3=float(mean_top3),
        mean_top5=float(df['top5'].mean()),
        std_top3=float(df['top3'].std()),
        literature_range=(60.0, 80.0), passes=passes,
        per_agent=df
    )


# =============================================================================
# METRIC 5 -- Zipf's Law / POI Popularity (Noulas 2012)
# =============================================================================

def validate_zipf_law(acts: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  5. ZIPF'S LAW (POI POPULARITY)  --  Noulas et al. (2012)")
    print(SEP)

    non_home = acts[acts['activity_type'] != 'HOME'].dropna(subset=['location_id'])
    counts = non_home['location_id'].value_counts().values.astype(float)
    ranks  = np.arange(1, len(counts) + 1)

    slope, intercept, r, _, _ = stats.linregress(np.log(ranks), np.log(counts))
    passes = -1.5 <= slope <= -0.5

    print(f"  Unique POIs visited : {len(counts):,}")
    print(f"  Power-law slope     : {slope:.3f}  (ideal Zipf: -1.0)")
    print(f"  R^2                  : {r**2:.3f}")
    print(f"  Status              : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric="Zipf's Law", reference='Noulas 2012',
        n_unique_pois=len(counts), slope=float(slope),
        r_squared=float(r**2), literature_slope=-1.0,
        passes=passes, ranks=ranks, counts=counts
    )


# =============================================================================
# METRIC 6 -- Temporal Patterns (Song 2010)
# =============================================================================

def validate_temporal_patterns(acts: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  6. TEMPORAL PATTERNS  --  Song et al. (2010)")
    print(SEP)

    non_home = acts[acts['activity_type'] != 'HOME'].copy()
    non_home['hour'] = non_home['start_time'].dt.hour

    expected_peaks = {
        'WORK':    (7, 10),
        'FOOD':    (11, 20),
        'ERRAND':  (8, 19),
        'LEISURE': (16, 23),
    }

    results = {}
    for act_type, (peak_start, peak_end) in expected_peaks.items():
        subset = non_home[non_home['activity_type'] == act_type]
        if len(subset) == 0:
            continue
        in_peak    = ((subset['hour'] >= peak_start) & (subset['hour'] < peak_end)).sum()
        pct_in_peak = in_peak / len(subset) * 100
        passes     = pct_in_peak >= 60.0
        print(f"  {act_type:<8} peak window {peak_start:02d}:00-{peak_end:02d}:00 : "
              f"{pct_in_peak:.1f}% in window  {'[OK]' if passes else '[FAIL]'}")
        results[act_type] = dict(
            peak_window=(peak_start, peak_end),
            pct_in_peak=float(pct_in_peak), passes=passes
        )

    return dict(metric='Temporal Patterns', reference='Song 2010',
                activity_results=results, non_home_acts=non_home)


# =============================================================================
# METRIC 7 -- Activity Entropy (Song 2010)
# =============================================================================

def validate_activity_entropy(acts: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  7. ACTIVITY ENTROPY  --  Song et al. (2010)")
    print(SEP)

    non_home = acts[acts['activity_type'] != 'HOME'].dropna(subset=['location_id'])
    entropies = []

    for sim_id, agent_acts in non_home.groupby('sim_id'):
        counts = agent_acts['location_id'].value_counts().values.astype(float)
        probs  = counts / counts.sum()
        h = entropy(probs, base=2)
        entropies.append(h)

    mean_ent  = np.mean(entropies)
    max_ent   = np.log2(non_home['location_id'].nunique())
    norm_ent  = mean_ent / max_ent if max_ent > 0 else 0
    passes    = 0.3 <= norm_ent <= 0.9

    print(f"  Mean entropy (bits)   : {mean_ent:.3f}")
    print(f"  Max possible (bits)   : {max_ent:.3f}")
    print(f"  Normalised entropy    : {norm_ent:.3f}  (target: 0.3-0.9)")
    print(f"  Status                : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Activity Entropy', reference='Song 2010',
        mean_entropy_bits=float(mean_ent), max_entropy_bits=float(max_ent),
        normalised_entropy=float(norm_ent), passes=passes, entropies=entropies
    )


# =============================================================================
# METRIC 8 -- Inter-visit Time Distribution (Barabasi 2010)
# =============================================================================

def validate_inter_visit_times(acts: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  8. INTER-VISIT TIME DISTRIBUTION  --  Barabasi (2010)")
    print(SEP)

    non_home = acts[acts['activity_type'] != 'HOME'].dropna(subset=['location_id']).copy()
    non_home = non_home.sort_values(['sim_id', 'location_id', 'start_time'])

    ivts = []
    for (sim_id, loc_id), visits in non_home.groupby(['sim_id', 'location_id']):
        times = visits['start_time'].dropna().sort_values()
        if len(times) < 2:
            continue
        diffs = times.diff().dropna().dt.total_seconds() / 3600
        ivts.extend(diffs[diffs > 0].tolist())

    ivts = np.array(ivts)
    if len(ivts) > 10:
        tail = ivts[ivts >= 1.0]
        try:
            fit = stats.powerlaw.fit(tail, floc=0)
            alpha = fit[0]
        except Exception:
            alpha = float('nan')
        slope, intercept, r, _, _ = stats.linregress(np.log(tail), np.log(np.ones(len(tail))))
        passes = r ** 2 >= 0.5 or not np.isnan(alpha)
    else:
        alpha, r, passes = float('nan'), float('nan'), False

    print(f"  Total inter-visit gaps  : {len(ivts):,}")
    print(f"  Mean gap                : {ivts.mean():.1f} h")
    print(f"  Median gap              : {np.median(ivts):.1f} h")
    print(f"  Power-law alpha         : {alpha:.3f}  (literature: heavy tail expected)")
    print(f"  Status                  : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Inter-visit Times', reference='Barabasi 2010',
        n_gaps=len(ivts), mean_gap_h=float(ivts.mean()),
        median_gap_h=float(np.median(ivts)),
        powerlaw_alpha=float(alpha), passes=passes, ivts=ivts
    )


# =============================================================================
# METRIC G1 -- Cell Zipf's Law
# =============================================================================

def validate_cell_zipf(vc_df: pd.DataFrame) -> dict:
    """
    Tests whether cell visit counts follow a Zipf power-law distribution,
    analogous to POI-level Metric 5 but at 500m cell resolution.

    Cells aggregate multiple POIs, so the slope is expected to be steeper
    than the POI-level slope (fewer dominant cells capture more visits).
    Literature range widened to -2.0 to -0.8 to reflect this.

    Comparison note: if cell_slope << poi_slope, agent visitation is
    strongly spatially clustered.  If they are similar, agents spread
    visits across many distinct cells.
    """
    print(f"\n{SEP}")
    print("  G1. CELL ZIPF'S LAW  (grid-level analogue of Noulas 2012)")
    print(SEP)

    counts = vc_df.sort_values('visit_count', ascending=False)['visit_count'].values.astype(float)
    ranks  = np.arange(1, len(counts) + 1)

    slope, intercept, r, _, _ = stats.linregress(np.log(ranks), np.log(counts))
    passes = -2.0 <= slope <= -0.8

    print(f"  Unique cells visited : {len(counts):,}")
    print(f"  Power-law slope      : {slope:.3f}  (ideal Zipf: -1.0; expected steeper for cells)")
    print(f"  R^2                   : {r**2:.3f}")
    print(f"  Status               : {'[OK] PASS' if passes else '[FAIL] FAIL'}  (range: -2.0 to -0.8)")

    return dict(
        metric="Cell Zipf's Law", reference='Grid-level / Noulas 2012',
        n_unique_cells=int(len(counts)), slope=float(slope),
        r_squared=float(r**2), literature_slope_range=(-2.0, -0.8),
        passes=passes, ranks=ranks, counts=counts
    )


# =============================================================================
# METRIC G2 -- Cell Entropy
# =============================================================================

def validate_cell_entropy(acts_grid: pd.DataFrame) -> dict:
    """
    Per-agent entropy of cell visitation, analogous to POI-level Metric 7.

    Cell entropy is expected to be lower than POI entropy because cells
    aggregate POIs -- agents appear more spatially predictable at coarser
    resolution.  Normalised entropy target relaxed to 0.2-0.85.

    Comparison note: (poi_entropy - cell_entropy) quantifies intra-cell
    diversity -- how much agents vary which specific POI they visit within
    the same 500m neighbourhood.
    """
    print(f"\n{SEP}")
    print("  G2. CELL ENTROPY  (grid-level analogue of Song 2010)")
    print(SEP)

    non_home = acts_grid[
        (acts_grid['activity_type'] != 'HOME') &
        (acts_grid['cell_id'] != 'invalid')
    ].dropna(subset=['cell_id'])

    entropies = []
    for sim_id, agent_acts in non_home.groupby('sim_id'):
        counts = agent_acts['cell_id'].value_counts().values.astype(float)
        probs  = counts / counts.sum()
        h = entropy(probs, base=2)
        entropies.append(h)

    if not entropies:
        print("  No data.")
        return dict(metric='Cell Entropy', passes=False)

    mean_ent = np.mean(entropies)
    max_ent  = np.log2(non_home['cell_id'].nunique())
    norm_ent = mean_ent / max_ent if max_ent > 0 else 0
    passes   = 0.2 <= norm_ent <= 0.85

    print(f"  Mean entropy (bits)   : {mean_ent:.3f}")
    print(f"  Max possible (bits)   : {max_ent:.3f}")
    print(f"  Normalised entropy    : {norm_ent:.3f}  (target: 0.2-0.85)")
    print(f"  Status                : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Cell Entropy', reference='Grid-level / Song 2010',
        mean_entropy_bits=float(mean_ent), max_entropy_bits=float(max_ent),
        normalised_entropy=float(norm_ent),
        literature_range=(0.2, 0.85), passes=passes, entropies=entropies
    )


# =============================================================================
# METRIC G3 -- Cell Revisit Rate
# =============================================================================

def validate_cell_revisit_rate(acts_grid: pd.DataFrame) -> dict:
    """
    Fraction of non-home cell visits that revisit a previously seen cell,
    analogous to POI-level Metric 3.

    Cell revisit rate is expected to be higher than POI revisit rate
    because cells are coarser -- two visits to different POIs in the same
    500m cell both count as revisits here.  Target: >= 0.70.

    Comparison note: if cell_revisit ~ poi_revisit, agents return to the
    same POI within the same cell (high within-cell fidelity).
    If cell_revisit >> poi_revisit, agents revisit the same neighbourhood
    but switch between POIs inside it.
    """
    print(f"\n{SEP}")
    print("  G3. CELL REVISIT RATE  (grid-level analogue of Pappalardo 2015)")
    print(SEP)

    non_home = acts_grid[
        (acts_grid['activity_type'] != 'HOME') &
        (acts_grid['cell_id'] != 'invalid')
    ].copy()

    revisit_rates = []
    for sim_id, agent_acts in non_home.groupby('sim_id'):
        agent_acts = agent_acts.sort_values('start_time')
        seen, revisits, total = set(), 0, 0
        for cell in agent_acts['cell_id']:
            if cell in seen:
                revisits += 1
            seen.add(cell)
            total += 1
        if total > 0:
            revisit_rates.append(revisits / total)

    if not revisit_rates:
        print("  No data.")
        return dict(metric='Cell Revisit Rate', passes=False)

    mean_revisit = np.mean(revisit_rates)
    passes = mean_revisit >= 0.70

    print(f"  Mean cell revisit rate  : {mean_revisit*100:.1f}%")
    print(f"  Median cell revisit rate: {np.median(revisit_rates)*100:.1f}%")
    print(f"  Target                  : >=70%  (higher than POI-level due to cell aggregation)")
    print(f"  Status                  : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Cell Revisit Rate', reference='Grid-level / Pappalardo 2015',
        mean_revisit_rate=float(mean_revisit),
        median_revisit_rate=float(np.median(revisit_rates)),
        literature_target=0.70, passes=passes, revisit_rates=revisit_rates
    )


# =============================================================================
# METRIC G4 -- Cell Flow Power Law
# =============================================================================

def validate_cell_flow_distribution(flow_df: pd.DataFrame) -> dict:
    """
    Tests whether the distribution of O-D trip volumes between cells follows
    a power law -- the gravity model analogue (Simini et al. 2012).

    In real cities, a small number of cell-pairs dominate total trips
    (commuter corridors, high-density centres).  A power-law slope in
    the range -1.5 to -0.5 is consistent with empirical O-D distributions.

    This metric has no direct POI-level analogue and is genuinely additive --
    it captures movement corridor structure that only emerges at cell resolution.
    """
    print(f"\n{SEP}")
    print("  G4. CELL FLOW POWER LAW  (Simini et al. 2012 -- gravity model analogue)")
    print(SEP)

    trip_counts = flow_df['trips'].sort_values(ascending=False).values.astype(float)
    if len(trip_counts) < 10:
        print("  Insufficient O-D pairs for power-law fit.")
        return dict(metric='Cell Flow Power Law', passes=False)

    ranks = np.arange(1, len(trip_counts) + 1)
    slope, intercept, r, _, _ = stats.linregress(np.log(ranks), np.log(trip_counts))
    passes = -1.5 <= slope <= -0.5

    top10_pairs = max(1, int(len(trip_counts) * 0.1))
    top10_share = trip_counts[:top10_pairs].sum() / trip_counts.sum() * 100

    print(f"  Unique O-D cell pairs  : {len(trip_counts):,}")
    print(f"  Total trips            : {int(trip_counts.sum()):,}")
    print(f"  Top 10% pairs carry    : {top10_share:.1f}% of all trips")
    print(f"  Power-law slope        : {slope:.3f}  (target: -1.5 to -0.5)")
    print(f"  R^2                     : {r**2:.3f}")
    print(f"  Status                 : {'[OK] PASS' if passes else '[FAIL] FAIL'}")

    return dict(
        metric='Cell Flow Power Law', reference='Grid-level / Simini 2012',
        n_od_pairs=int(len(trip_counts)),
        total_trips=int(trip_counts.sum()),
        top10pct_share=float(top10_share),
        slope=float(slope), r_squared=float(r**2),
        literature_slope_range=(-1.5, -0.5),
        passes=passes, trip_counts=trip_counts, ranks=ranks
    )


# =============================================================================
# PLOTS -- standard (metrics 1-8)
# =============================================================================

def plot_all(results: dict, output_dir: str):
    os.makedirs(f'{output_dir}/plots', exist_ok=True)

    # -- 1. Distance distribution ----------------------------------------------
    d = results['distance']['distances']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(d, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(d.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean {d.mean():.2f} km')
    ax1.set_xlabel('Trip Distance (km)'); ax1.set_ylabel('Count')
    ax1.set_title('Trip Distance Distribution'); ax1.legend(); ax1.grid(True, alpha=0.3)

    bins = np.logspace(np.log10(d[d>0].min()), np.log10(d.max()), 40)
    hist, edges = np.histogram(d[d>0], bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    ax2.loglog(centers[mask], hist[mask], 'o', markersize=4, alpha=0.6, label='Data')
    theoretical = centers ** -1.75; theoretical /= theoretical[mask][0] / hist[mask][0]
    ax2.loglog(centers[mask], theoretical[mask], 'g--', linewidth=2, label='Theory alpha=1.75')
    ax2.set_xlabel('Distance (km)'); ax2.set_ylabel('P(d)')
    ax2.set_title('Log-Log Distance (Gonzalez 2008)'); ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/01_distance_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- 2. Radius of gyration -------------------------------------------------
    rogs = results['rog']['rogs']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rogs, bins=30, color='#00A896', edgecolor='black', alpha=0.7)
    ax.axvspan(5, 20, alpha=0.15, color='green', label='Literature range 5-20 km')
    ax.axvline(np.mean(rogs), color='red', linestyle='--', linewidth=2,
               label=f'Mean {np.mean(rogs):.2f} km')
    ax.set_xlabel('Radius of Gyration (km)'); ax.set_ylabel('Agents')
    ax.set_title('Radius of Gyration Distribution (Gonzalez 2008)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/02_radius_of_gyration.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- 3. Revisit rates ------------------------------------------------------
    rr = results['revisit']['revisit_rates']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rr, bins=30, color='#028090', edgecolor='black', alpha=0.7)
    ax.axvline(0.70, color='green', linestyle='--', linewidth=2,
               label='Target 70% (Pappalardo 2015)')
    ax.axvline(np.mean(rr), color='red', linestyle='--', linewidth=2,
               label=f'Mean {np.mean(rr)*100:.1f}%')
    ax.set_xlabel('Revisit Rate'); ax.set_ylabel('Agents')
    ax.set_title('Revisit Rate Distribution (Pappalardo 2015)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/03_revisit_rates.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- 4. Top-N concentration ------------------------------------------------
    df_topn = results['topn']['per_agent']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    categories = ['Top-1', 'Top-3', 'Top-5']
    means = [df_topn['top1'].mean(), df_topn['top3'].mean(), df_topn['top5'].mean()]
    stds  = [df_topn['top1'].std(),  df_topn['top3'].std(),  df_topn['top5'].std()]
    ax1.bar(categories, means, yerr=stds, capsize=6,
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', alpha=0.8)
    ax1.axhspan(60, 80, alpha=0.15, color='green', label='Literature target 60-80%')
    ax1.set_ylabel('% of visits'); ax1.set_title('Top-N POI Concentration (Pappalardo 2015)')
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')

    ax2.hist(df_topn['top3'], bins=30, color='#4CAF50', edgecolor='black', alpha=0.7)
    ax2.axvspan(60, 80, alpha=0.15, color='green', label='Target 60-80%')
    ax2.axvline(df_topn['top3'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean {df_topn['top3'].mean():.1f}%")
    ax2.set_xlabel('Top-3 Concentration (%)'); ax2.set_ylabel('Agents')
    ax2.set_title('Distribution of Top-3 Concentration')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/04_topn_concentration.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- 5. Zipf's law ---------------------------------------------------------
    ranks  = results['zipf']['ranks']
    counts = results['zipf']['counts']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.loglog(ranks, counts, 'o', markersize=3, alpha=0.5, label='Data')
    theoretical = counts[0] * ranks ** -1.0
    ax1.loglog(ranks, theoretical, 'g--', linewidth=2, label='Zipf alpha=1.0')
    ax1.set_xlabel('POI Rank'); ax1.set_ylabel('Visit Count')
    ax1.set_title("Zipf's Law -- POI Popularity (Noulas 2012)")
    ax1.legend(); ax1.grid(True, alpha=0.3, which='both')

    cum = np.cumsum(counts) / np.sum(counts) * 100
    ax2.plot(ranks, cum, linewidth=2, color='#00A896')
    ax2.axhline(80, color='red', linestyle='--', linewidth=1.5, label='80% of visits')
    ax2.set_xlabel('Number of POIs'); ax2.set_ylabel('Cumulative % of visits')
    ax2.set_title('Cumulative Visit Distribution')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/05_zipf_law.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- 6. Temporal patterns --------------------------------------------------
    non_home = results['temporal']['non_home_acts']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, act_type in enumerate(['WORK', 'FOOD', 'ERRAND', 'LEISURE']):
        ax = axes[idx]
        subset = non_home[non_home['activity_type'] == act_type]
        if len(subset) == 0:
            continue
        hours = subset['hour'].values
        ax.hist(hours, bins=24, range=(0, 24),
                color=ACTIVITY_COLORS.get(act_type, 'gray'), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Hour of Day'); ax.set_ylabel('Count')
        ax.set_title(f'{act_type} Start Times'); ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)], rotation=30)
    plt.suptitle('Temporal Activity Patterns (Song 2010)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/06_temporal_patterns.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- 7. Entropy distribution -----------------------------------------------
    ents = results['entropy']['entropies']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ents, bins=30, color='#9467bd', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(ents), color='red', linestyle='--', linewidth=2,
               label=f'Mean {np.mean(ents):.2f} bits')
    ax.set_xlabel('Entropy (bits)'); ax.set_ylabel('Agents')
    ax.set_title('Location Entropy per Agent (Song 2010)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/07_entropy.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -- 8. Inter-visit times --------------------------------------------------
    ivts = results['ivt']['ivts']
    ivts_clipped = ivts[ivts <= np.percentile(ivts, 99)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(ivts_clipped, bins=60, color='coral', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Inter-visit Time (hours)'); ax1.set_ylabel('Count')
    ax1.set_title('Inter-visit Time Distribution (Barabasi 2010)')
    ax1.grid(True, alpha=0.3)

    tail = ivts[ivts >= 1.0]
    bins = np.logspace(0, np.log10(tail.max()), 40)
    hist, edges = np.histogram(tail, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    ax2.loglog(centers[mask], hist[mask], 'o', markersize=4, alpha=0.6)
    ax2.set_xlabel('Inter-visit Time (hours)'); ax2.set_ylabel('P(dt)')
    ax2.set_title('Log-Log Inter-visit Time'); ax2.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/08_inter_visit_times.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n  [OK] Standard plots (1-8) saved to {output_dir}/plots/")


# =============================================================================
# PLOTS -- grid metrics (G1-G4)
# =============================================================================

def plot_grid_metrics(grid_results: dict, output_dir: str) -> None:
    os.makedirs(f'{output_dir}/plots', exist_ok=True)

    # -- G1. Cell Zipf ---------------------------------------------------------
    r = grid_results.get('cell_zipf', {})
    if 'ranks' in r and 'counts' in r:
        ranks, counts = r['ranks'], r['counts']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.loglog(ranks, counts, 'o', markersize=3, alpha=0.5, label='Data')
        theoretical = counts[0] * ranks ** -1.0
        ax1.loglog(ranks, theoretical, 'g--', linewidth=2, label='Zipf alpha=1.0 (POI reference)')
        ax1.set_xlabel('Cell Rank'); ax1.set_ylabel('Visit Count')
        ax1.set_title("G1. Cell Zipf's Law")
        ax1.legend(); ax1.grid(True, alpha=0.3, which='both')

        cum = np.cumsum(counts) / np.sum(counts) * 100
        ax2.plot(ranks, cum, linewidth=2, color='#00A896')
        ax2.axhline(80, color='red', linestyle='--', linewidth=1.5, label='80% of visits')
        ax2.set_xlabel('Number of Cells'); ax2.set_ylabel('Cumulative % of visits')
        ax2.set_title('Cumulative Visit Distribution (cells)')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/G1_cell_zipf.png', dpi=200, bbox_inches='tight')
        plt.close()

    # -- G2. Cell Entropy ------------------------------------------------------
    r = grid_results.get('cell_entropy', {})
    if 'entropies' in r:
        ents = r['entropies']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(ents, bins=30, color='#9467bd', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(ents), color='red', linestyle='--', linewidth=2,
                   label=f'Mean {np.mean(ents):.2f} bits')
        ax.set_xlabel('Cell Entropy (bits)'); ax.set_ylabel('Agents')
        ax.set_title('G2. Cell Location Entropy per Agent')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/G2_cell_entropy.png', dpi=200, bbox_inches='tight')
        plt.close()

    # -- G3. Cell Revisit Rate -------------------------------------------------
    r = grid_results.get('cell_revisit', {})
    if 'revisit_rates' in r:
        rr = r['revisit_rates']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(rr, bins=30, color='#028090', edgecolor='black', alpha=0.7)
        ax.axvline(0.70, color='green', linestyle='--', linewidth=2, label='Target >=70%')
        ax.axvline(np.mean(rr), color='red', linestyle='--', linewidth=2,
                   label=f'Mean {np.mean(rr)*100:.1f}%')
        ax.set_xlabel('Cell Revisit Rate'); ax.set_ylabel('Agents')
        ax.set_title('G3. Cell Revisit Rate Distribution')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/G3_cell_revisit.png', dpi=200, bbox_inches='tight')
        plt.close()

    # -- G4. Cell Flow Power Law -----------------------------------------------
    r = grid_results.get('cell_flow', {})
    if 'trip_counts' in r and 'ranks' in r:
        tc, ranks = r['trip_counts'], r['ranks']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(range(min(50, len(tc))), tc[:50],
                color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3)
        ax1.set_xlabel('O-D Pair Rank (top 50)'); ax1.set_ylabel('Trip Count')
        ax1.set_title('G4. Top 50 Cell O-D Flows')
        ax1.grid(True, alpha=0.3, axis='y')

        mask = tc > 0
        ax2.loglog(ranks[mask], tc[mask], 'o', markersize=3, alpha=0.5, label='Data')
        ref = tc[0] * ranks ** -1.0
        ax2.loglog(ranks, ref, 'g--', linewidth=2, label='Reference alpha=1.0')
        ax2.set_xlabel('O-D Pair Rank'); ax2.set_ylabel('Trip Count')
        ax2.set_title('Log-Log Cell Flow Distribution')
        ax2.legend(); ax2.grid(True, alpha=0.3, which='both')
        plt.suptitle('G4. Cell Flow Power Law  (Simini 2012 / gravity model)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/G4_cell_flow.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  [OK] Grid plots (G1-G4) saved to {output_dir}/plots/")


# =============================================================================
# TEXT REPORT
# =============================================================================

def write_report(results: dict, output_dir: str, args, grid_results: dict = None):
    os.makedirs(output_dir, exist_ok=True)
    lines = []

    lines.append(SEP)
    lines.append("  MOBILITY THEORY VALIDATION REPORT")
    lines.append(SEP)
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Activities: {args.activities}")
    lines.append(f"  Travels   : {args.travels}")
    lines.append("")

    # -- Standard scorecard ----------------------------------------------------
    checks = [
        ('distance', 'Distance Distribution',     'Gonzalez 2008'),
        ('rog',      'Radius of Gyration',         'Gonzalez 2008'),
        ('revisit',  'Revisit Rate',               'Pappalardo 2015'),
        ('topn',     'Top-N Concentration',        'Pappalardo 2015'),
        ('zipf',     "Zipf's Law",                 'Noulas 2012'),
        ('temporal', 'Temporal Patterns',          'Song 2010'),
        ('entropy',  'Activity Entropy',           'Song 2010'),
        ('ivt',      'Inter-visit Times',          'Barabasi 2010'),
    ]

    lines.append(SEP)
    lines.append("  SUMMARY SCORECARD  (POI-level)")
    lines.append(SEP)
    lines.append(f"  {'Metric':<35} {'Reference':<20} {'Result'}")
    lines.append(THIN)

    passed = 0
    for key, name, ref in checks:
        r = results.get(key, {})
        p = r.get('passes', False)
        if key == 'temporal':
            sub = r.get('activity_results', {})
            p = all(v.get('passes', False) for v in sub.values())
        status = '[OK] PASS' if p else '[FAIL] FAIL'
        if p:
            passed += 1
        lines.append(f"  {name:<35} {ref:<20} {status}")

    lines.append(THIN)
    lines.append(f"  Total: {passed}/{len(checks)} checks passed")
    lines.append("")

    # -- Grid scorecard (optional) ---------------------------------------------
    if grid_results:
        grid_checks = [
            ('cell_zipf',    "Cell Zipf's Law",    'Grid / Noulas 2012'),
            ('cell_entropy', 'Cell Entropy',        'Grid / Song 2010'),
            ('cell_revisit', 'Cell Revisit Rate',   'Grid / Pappalardo 2015'),
            ('cell_flow',    'Cell Flow Power Law', 'Grid / Simini 2012'),
        ]

        lines.append(SEP)
        lines.append("  SUMMARY SCORECARD  (grid-level -- 500m cells, additive not replacement)")
        lines.append(SEP)
        lines.append(
            "  Cell-level and POI-level metrics test different aspects of spatial\n"
            "  realism and should be read together.  Cell metrics are expected to\n"
            "  show higher revisit rates and lower entropy than POI metrics due to\n"
            "  spatial aggregation -- that difference is itself informative."
        )
        lines.append("")
        lines.append(f"  {'Metric':<35} {'Reference':<25} {'Result'}")
        lines.append(THIN)

        g_passed = 0
        for key, name, ref in grid_checks:
            r = grid_results.get(key, {})
            p = r.get('passes', False)
            status = '[OK] PASS' if p else '[FAIL] FAIL'
            if p:
                g_passed += 1
            lines.append(f"  {name:<35} {ref:<25} {status}")

        lines.append(THIN)
        lines.append(f"  Total: {g_passed}/{len(grid_checks)} grid checks passed")
        lines.append("")

    # -- Standard detail sections ----------------------------------------------
    _SKIP = {'distances', 'rogs', 'revisit_rates', 'per_agent',
             'ranks', 'counts', 'non_home_acts', 'entropies', 'ivts'}

    for key, name, ref in checks:
        r = results.get(key, {})
        lines.append(SEP)
        lines.append(f"  {name.upper()}  ({ref})")
        lines.append(THIN)
        for k, v in r.items():
            if k in _SKIP:
                continue
            if k == 'activity_results':
                for act, av in v.items():
                    lines.append(
                        f"  {act:<10} peak {av['peak_window'][0]:02d}:00-"
                        f"{av['peak_window'][1]:02d}:00 : "
                        f"{av['pct_in_peak']:.1f}% in window  "
                        f"{'[OK]' if av['passes'] else '[FAIL]'}"
                    )
            else:
                lines.append(f"  {k:<30}: {v}")
        lines.append("")

    # -- Grid detail sections (optional) --------------------------------------
    if grid_results:
        _SKIP_GRID = {'ranks', 'counts', 'entropies', 'revisit_rates', 'trip_counts'}
        lines.append(SEP)
        lines.append("  GRID-LEVEL METRIC DETAILS")
        lines.append(SEP)
        lines.append("")

        for key, name, ref in grid_checks:
            r = grid_results.get(key, {})
            lines.append(SEP)
            lines.append(f"  {name.upper()}  ({ref})")
            lines.append(THIN)
            for k, v in r.items():
                if k in _SKIP_GRID:
                    continue
                lines.append(f"  {k:<35}: {v}")
            lines.append("")

    out_path = f'{output_dir}/validation_report.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [OK] Report saved to {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Mobility Theory Validator')
    parser.add_argument('--activities', required=True,
                        help='Path to activity_log.csv')
    parser.add_argument('--travels',    required=True,
                        help='Path to travel_log.csv')
    parser.add_argument('--output',     default='validation_output',
                        help='Output directory')
    parser.add_argument('--use_grid',   action='store_true',
                        help='Also run cell-level metrics (requires grid_data/ from sim.py --use_grid)')
    parser.add_argument('--grid_dir',   default=None,
                        help='Path to grid_data/ folder (defaults to ../grid_data relative to --activities)')
    args = parser.parse_args()

    print(SEP)
    print("  MOBILITY THEORY VALIDATOR")
    print(SEP)
    print(f"  Activities : {args.activities}")
    print(f"  Travels    : {args.travels}")
    print(f"  Output     : {args.output}")

    if args.use_grid:
        grid_dir = args.grid_dir or os.path.normpath(
            os.path.join(os.path.dirname(args.activities), '..', 'grid_data')
        )
        print(f"  Grid data  : {grid_dir}")

    # -- Standard 8 metrics ----------------------------------------------------
    acts, travs = load_data(args.activities, args.travels)

    results = {
        'distance': validate_distance_distribution(travs),
        'rog':      validate_radius_of_gyration(acts, travs),
        'revisit':  validate_revisit_rate(acts),
        'topn':     validate_topn_concentration(acts),
        'zipf':     validate_zipf_law(acts),
        'temporal': validate_temporal_patterns(acts),
        'entropy':  validate_activity_entropy(acts),
        'ivt':      validate_inter_visit_times(acts),
    }

    # -- Grid metrics G1-G4 (optional) -----------------------------------------
    grid_results = {}
    if args.use_grid:
        print(f"\n{SEP}")
        print("  CELL-LEVEL METRICS  (500m x 500m grid)")
        print(SEP)
        try:
            acts_grid, travs_grid, vc_df, flow_df = load_grid_data(grid_dir)
            grid_results = {
                'cell_zipf':    validate_cell_zipf(vc_df),
                'cell_entropy': validate_cell_entropy(acts_grid),
                'cell_revisit': validate_cell_revisit_rate(acts_grid),
                'cell_flow':    validate_cell_flow_distribution(flow_df),
            }
            plot_grid_metrics(grid_results, args.output)
        except FileNotFoundError as e:
            print(f"\n  [ERROR] {e}")
            print("  Skipping grid metrics.")

    # -- Plots and report ------------------------------------------------------
    plot_all(results, args.output)
    write_report(results, args.output, args, grid_results=grid_results)

    print(f"\n{SEP}")
    print("  DONE")
    print(SEP)


if __name__ == '__main__':
    main()