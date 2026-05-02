"""
==============================================================================
  INDIA VACCINE SUPPLY CHAIN — META-HEURISTIC OPTIMIZER  v4.0
  Three-Tier: Suppliers (S) -> Distribution Centers (D) -> Retailers (R)

  Objectives  (all minimized):
    f1 = Total Cost      (transport + fixed DC + disposal + carbon + penalties)
    f2 = Total Waste     (units lost to decay on all arcs)
    f3 = Total Emissions (kg CO2-eq across all arcs)

  v4.0 UPGRADES over v3.0  (each marked with  ── UPGRADE N ──):
    [1]  Geographical realism   — distance threshold + region-based grouping
    [2]  Minimum DC constraint  — GA/NSGA-II enforce >= MIN_DC_OPEN
    [3]  Distance-based decay   — decay = base + alpha * distance (replaces fixed)
    [4]  Dynamic refrigeration  — cost-benefit decision per arc, not threshold
    [5]  DC utilisation balance — soft penalty for >90% loaded DCs
    [6]  Soft constraint handle — 1e18 hard rejects replaced with graded penalty
    [7]  Transport mode realism — Van / Truck selected by distance; different
                                  cost & emission factors applied
    [8]  Performance            — precomputed sorted supplier & DC lookup arrays
    [9]  Logging                — per-generation: cost, waste, emit, DCs, util%
    [10] Hard DC capacity kept  — 0% overflow still enforced (unchanged)

  Dependencies: numpy, pandas   (NO Gurobi / pymoo needed)
  Usage:        python vaccine_optimizer_v4.py
==============================================================================
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = "."
RANDOM_SEED = 42

# ── Model parameters (WHO/eVIN/IPCC calibrated) ──────────────────────────────
DISPOSAL_C   = 12.0    # ₹ per wasted unit
LAMBDA_C     = 1.50    # ₹ per kg CO2

# ── UPGRADE 3 : Distance-based decay coefficients ────────────────────────────
# decay(d) = BASE_DECAY_NR + ALPHA_DECAY * distance_km   (non-refrigerated)
# decay(d) = BASE_DECAY_R  + ALPHA_DECAY_R * distance_km (refrigerated)
BASE_DECAY_NR  = 0.10   # 10% base loss even for 0-km arc
ALPHA_DECAY    = 0.00015 # +0.015% per km  → 200 km → ~13% extra
BASE_DECAY_R   = 0.01   # 1% base for refrigerated
ALPHA_DECAY_R  = 0.00002 # very small distance effect under cold chain
MAX_DECAY      = 0.45   # cap

# ── UPGRADE 4 : Dynamic refrigeration — cost-benefit break-even ──────────────
# Refrigerate if:  savings_from_less_decay * DISPOSAL_C  >  refrig_premium_cost
# i.e. (delta_NR - delta_R) * demand * DISPOSAL_C  >  r_premium * ship_qty
# This is evaluated per-arc inside the evaluator (no static threshold).

# ── UPGRADE 1 : Geographical distance threshold ──────────────────────────────
MAX_DIST_SD    = 2500.0  # km — supplier→DC routes beyond this get penalty
MAX_DIST_DR    = 350.0   # km — DC→retailer routes beyond this get penalty
DIST_PENALTY   = 0.10    # extra ₹/unit/km beyond threshold (soft penalty)

# ── UPGRADE 2 : Minimum DC constraint ────────────────────────────────────────
MIN_DC_OPEN    = 8       # at least 8 DCs must be active
MIN_DC_PENALTY = 5e6     # ₹ per missing DC below minimum (soft penalty)

# ── UPGRADE 5 : DC utilisation balance ───────────────────────────────────────
UTIL_HIGH_THRESH = 0.90  # >90% utilised → penalty
UTIL_PENALTY_PER_UNIT = 8.0  # ₹ per unit that exceeds 90%

# ── UPGRADE 6 : Soft penalty multipliers (replace 1e15/1e18 hard rejects) ────
UNMET_PENALTY    = 500.0   # ₹ per unmet unit (was 500 → kept, but now graded)
SUP_EXHAUST_PEN  = 300.0   # ₹ per unit unserved due to supplier exhaustion

# ── UPGRADE 7 : Transport mode thresholds ────────────────────────────────────
# Mode selected automatically by distance:
#   Van   : distance <  VAN_MAX_KM
#   Truck : distance >= VAN_MAX_KM
VAN_MAX_KM     = 150.0
# Cost multipliers applied ON TOP of base c_ij / c_jk from arc files
# (these capture mode-specific fuel, toll, driver costs)
VAN_COST_MULT   = 1.20   # vans ~20% more expensive per unit-km (smaller load)
TRUCK_COST_MULT = 1.00   # trucks are the baseline in arc cost files
# Emission multipliers (vans have lower per-km CO2 but smaller payload)
VAN_EMIT_MULT   = 1.35   # vans emit more per unit carried
TRUCK_EMIT_MULT = 1.00

# GA hyper-parameters
GA_POP   = 100
GA_GENS  = 200
GA_CXPB  = 0.80
GA_MUTPB = 0.02
GA_ELITE = 5

# NSGA-II hyper-parameters
NS_POP   = 80
NS_GENS  = 120


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADER
#     UPGRADE 8 : precompute sorted supplier/DC lookup arrays here once.
# ─────────────────────────────────────────────────────────────────────────────
class SupplyChainData:
    """
    Loads CSVs; exposes numpy arrays for fast evaluation.
    UPGRADE 8: precomputes distance-based decay, transport mode masks,
               and sorted index arrays to avoid repeated per-eval sorting.
    """

    def __init__(self, data_dir="."):
        print("=" * 66)
        print("  Loading supply-chain data (v4.0) …")
        t0 = time.time()

        sup    = pd.read_csv(f"data_dir/suppliers.csv")
        dcs    = pd.read_csv(f"data_dir/dcs.csv")
        ret    = pd.read_csv(f"data_dir/retailers.csv")
        arc_sd = pd.read_csv(f"data_dir/arcs_supplier_to_dc.csv")
        arc_dr = pd.read_csv(f"data_dir/arcs_dc_to_retailer.csv")

        self.S  = list(sup["supplier_id"])
        self.D  = list(dcs["dc_id"])
        self.R  = list(ret["retailer_id"])
        self.ns = len(self.S)
        self.nd = len(self.D)
        self.nr = len(self.R)

        s_idx = {s: i for i, s in enumerate(self.S)}
        d_idx = {d: i for i, d in enumerate(self.D)}
        r_idx = {r: i for i, r in enumerate(self.R)}

        self.sup_cap  = sup["monthly_capacity_units"].values.astype(float)
        self.dc_fixed = dcs["fixed_cost_lakhs"].values.astype(float) * 1e5
        self.dc_cap   = dcs["capacity_units"].values.astype(float)
        self.demand   = ret["monthly_demand_units"].values.astype(float)

        # ── UPGRADE 1 : store zone per node for region grouping ───────────────
        self.dc_zone  = dcs["zone"].values          # (nd,) string array
        self.ret_zone = ret["zone"].values           # (nr,) string array
        self.sup_zone = sup["zone"].values           # (ns,)

        ns, nd, nr = self.ns, self.nd, self.nr

        # Base arc arrays (from CSV)
        c_sd_base  = np.zeros((ns, nd)); r_sd  = np.zeros((ns, nd))
        e_sd_base  = np.zeros((ns, nd)); er_sd = np.zeros((ns, nd))
        dist_sd    = np.zeros((ns, nd))

        for _, row in arc_sd.iterrows():
            i = s_idx[row["supplier_id"]]; j = d_idx[row["dc_id"]]
            c_sd_base[i,j]  = row["cost_nonrefrig_per_unit"]
            r_sd[i,j]       = row["refrig_premium_per_unit"]
            e_sd_base[i,j]  = row["emit_nonrefrig_kgco2_per_unit"]
            er_sd[i,j]      = row["emit_refrig_kgco2_per_unit"]
            dist_sd[i,j]    = row["distance_km"]

        c_dr_base  = np.zeros((nd, nr)); r_dr  = np.zeros((nd, nr))
        e_dr_base  = np.zeros((nd, nr)); er_dr = np.zeros((nd, nr))
        dist_dr    = np.zeros((nd, nr))

        for _, row in arc_dr.iterrows():
            j = d_idx[row["dc_id"]]; k = r_idx[row["retailer_id"]]
            c_dr_base[j,k]  = row["cost_nonrefrig_per_unit"]
            r_dr[j,k]       = row["refrig_premium_per_unit"]
            e_dr_base[j,k]  = row["emit_nonrefrig_kgco2_per_unit"]
            er_dr[j,k]      = row["emit_refrig_kgco2_per_unit"]
            dist_dr[j,k]    = row["distance_km"]

        self.dist_sd = dist_sd
        self.dist_dr = dist_dr
        self.r_sd    = r_sd
        self.r_dr    = r_dr

        # ── UPGRADE 3 : compute distance-based decay matrices ─────────────────
        self.d_sd  = np.clip(BASE_DECAY_NR + ALPHA_DECAY   * dist_sd,
                             BASE_DECAY_NR, MAX_DECAY)   # non-refrig S→D
        self.dr_sd = np.clip(BASE_DECAY_R  + ALPHA_DECAY_R * dist_sd,
                             BASE_DECAY_R,  MAX_DECAY)   # refrig     S→D
        self.d_dr  = np.clip(BASE_DECAY_NR + ALPHA_DECAY   * dist_dr,
                             BASE_DECAY_NR, MAX_DECAY)   # non-refrig D→R
        self.dr_dr = np.clip(BASE_DECAY_R  + ALPHA_DECAY_R * dist_dr,
                             BASE_DECAY_R,  MAX_DECAY)   # refrig     D→R

        # ── UPGRADE 7 : transport mode masks & adjusted cost/emit ─────────────
        # S→D mode
        van_sd  = dist_sd < VAN_MAX_KM                   # (ns, nd) bool
        self.c_sd  = c_sd_base * np.where(van_sd, VAN_COST_MULT,  TRUCK_COST_MULT)
        self.e_sd  = e_sd_base * np.where(van_sd, VAN_EMIT_MULT,  TRUCK_EMIT_MULT)
        self.er_sd = er_sd     * np.where(van_sd, VAN_EMIT_MULT,  TRUCK_EMIT_MULT)
        self.mode_sd = np.where(van_sd, "van", "truck")  # for reporting

        # D→R mode
        van_dr  = dist_dr < VAN_MAX_KM                   # (nd, nr) bool
        self.c_dr  = c_dr_base * np.where(van_dr, VAN_COST_MULT,  TRUCK_COST_MULT)
        self.e_dr  = e_dr_base * np.where(van_dr, VAN_EMIT_MULT,  TRUCK_EMIT_MULT)
        self.er_dr = er_dr     * np.where(van_dr, VAN_EMIT_MULT,  TRUCK_EMIT_MULT)

        # ── UPGRADE 1 : distance penalty masks ───────────────────────────────
        # Extra cost per unit per km beyond threshold (soft geographic penalty)
        excess_sd = np.maximum(0, dist_sd - MAX_DIST_SD)
        excess_dr = np.maximum(0, dist_dr - MAX_DIST_DR)
        self.geo_pen_sd = excess_sd * DIST_PENALTY   # (ns, nd) ₹/unit extra
        self.geo_pen_dr = excess_dr * DIST_PENALTY   # (nd, nr) ₹/unit extra

        # ── UPGRADE 8 : precompute sorted indices for O(1) lookup ─────────────
        # For each DC j: suppliers sorted by cost ascending  → shape (nd, ns)
        self.sup_order_for_dc = np.argsort(self.c_sd, axis=0).T  # (nd, ns)

        # For each retailer k: DCs sorted by cost ascending → shape (nr, nd)
        self.dc_order_for_ret = np.argsort(self.c_dr, axis=0).T  # (nr, nd)

        # Retailer sort by demand descending (fixed for all evaluations)
        self.retailer_order = np.argsort(-self.demand)

        print(f"  ✓ Loaded + preprocessed in {time.time()-t0:.1f}s  "
              f"[S={ns}  D={nd}  R={nr}  arcs={ns*nd+nd*nr:,}]")
        print(f"  ✓ Van arcs (D→R <{VAN_MAX_KM}km): "
              f"{int(van_dr.sum()):,}  │  "
              f"Truck arcs: {int((~van_dr).sum()):,}")
        print("=" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EVALUATOR  — shared by GA and NSGA-II
#     All 10 upgrades active here.
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(chromosome, data):
    """
    chromosome : ndarray int (nd,)   1 = DC open, 0 = closed

    Returns dict: total_cost, waste, emissions, breakdown, feasibility.

    Constraints:
      [HARD]  DC capacity       — 0% overflow, spill to next DC  (unchanged)
      [SOFT]  Min DC open       — penalty if < MIN_DC_OPEN       [UPGRADE 2]
      [SOFT]  Distance thresh   — penalty on long arcs            [UPGRADE 1]
      [SOFT]  DC util balance   — penalty on >90% loaded DCs      [UPGRADE 5]
      [SOFT]  Unmet demand      — graded penalty                  [UPGRADE 6]
    """
    D_open = np.where(chromosome == 1)[0]
    nd, nr = data.nd, data.nr

    # ── UPGRADE 2 : minimum DC soft constraint ────────────────────────────────
    n_open = len(D_open)
    c_min_dc = max(0, MIN_DC_OPEN - n_open) * MIN_DC_PENALTY

    if n_open == 0:
        return _infeasible(c_min_dc)

    # ── Step 1 : assign retailers → DCs (HARD capacity, precomputed order) ───
    # UPGRADE 8: use precomputed dc_order_for_ret — no per-eval argsort
    remaining_cap = data.dc_cap.copy()
    assign_r      = np.full(nr, -1, dtype=int)
    use_r_dr      = np.zeros((nd, nr), dtype=bool)

    for k in data.retailer_order:                    # demand-desc order (cached)
        ranked_dcs = data.dc_order_for_ret[k]        # DCs sorted by cost for k

        for j in ranked_dcs:
            if chromosome[j] == 0:                   # DC closed — skip
                continue

            # ── UPGRADE 4 : dynamic refrigeration decision ────────────────────
            # Refrigerate if net savings exceed premium cost
            delta_nr = data.d_dr[j, k]
            delta_r  = data.dr_dr[j, k]
            ship_nr  = data.demand[k] / max(1 - delta_nr, 0.01)
            ship_r   = data.demand[k] / max(1 - delta_r,  0.01)
            waste_savings = (ship_nr * delta_nr - ship_r * delta_r) * DISPOSAL_C
            refrig_cost   = ship_r * data.r_dr[j, k]
            use_r  = waste_savings > refrig_cost     # ← dynamic decision

            dec    = delta_r if use_r else delta_nr
            ship   = data.demand[k] / max(1 - dec, 0.01)

            if remaining_cap[j] >= ship:             # HARD capacity check
                assign_r[k]      = j
                use_r_dr[j, k]   = use_r
                remaining_cap[j] -= ship
                break

    # ── Step 2 : flows D→R ───────────────────────────────────────────────────
    flow_dr = np.zeros((nd, nr))
    unmet   = 0.0

    for k in range(nr):
        j = assign_r[k]
        if j == -1:
            unmet += data.demand[k]
            continue
        dec         = data.dr_dr[j,k] if use_r_dr[j,k] else data.d_dr[j,k]
        flow_dr[j,k] = data.demand[k] / max(1 - dec, 0.01)

    # ── Step 3 : supply DCs from suppliers (precomputed order) ───────────────
    # UPGRADE 8: use precomputed sup_order_for_dc — no per-eval argsort
    dc_need  = flow_dr.sum(axis=1)
    sup_rem  = data.sup_cap.copy()
    flow_sd  = np.zeros((data.ns, nd))
    use_r_sd = np.zeros((data.ns, nd), dtype=bool)
    c_sup_exhaust = 0.0

    for j in D_open[np.argsort(-dc_need[D_open])]:
        need = dc_need[j]
        if need <= 0:
            continue

        filled = False
        for i in data.sup_order_for_dc[j]:          # cheapest supplier first
            if sup_rem[i] < 1.0:
                continue

            # ── UPGRADE 4 : dynamic refrigeration S→D ─────────────────────────
            delta_nr = data.d_sd[i, j]
            delta_r  = data.dr_sd[i, j]
            ship_nr  = need / max(1 - delta_nr, 0.01)
            ship_r   = need / max(1 - delta_r,  0.01)
            waste_sav = (ship_nr * delta_nr - ship_r * delta_r) * DISPOSAL_C
            ref_cost  = ship_r * data.r_sd[i, j]
            use_r     = waste_sav > ref_cost

            dec  = delta_r if use_r else delta_nr
            ship = min(need / max(1 - dec, 0.01), sup_rem[i])

            flow_sd[i, j]  = ship
            use_r_sd[i, j] = use_r
            sup_rem[i]    -= ship
            filled = True
            break

        if not filled:
            # UPGRADE 6: graded soft penalty instead of hard 1e15
            c_sup_exhaust += need * SUP_EXHAUST_PEN
            unmet         += need

    # ── Step 4 : cost components ─────────────────────────────────────────────
    # Base transport
    c_trans_sd = float(np.sum(
        flow_sd * np.where(use_r_sd, data.c_sd + data.r_sd, data.c_sd)))
    c_trans_dr = float(np.sum(
        flow_dr * np.where(use_r_dr, data.c_dr + data.r_dr, data.c_dr)))

    # ── UPGRADE 1 : geographic distance penalty ───────────────────────────────
    c_geo = float(np.sum(flow_sd * data.geo_pen_sd) +
                  np.sum(flow_dr * data.geo_pen_dr))

    c_fixed = float(data.dc_fixed[D_open].sum())

    # Waste
    waste_sd = float(np.sum(
        flow_sd * np.where(use_r_sd, data.dr_sd, data.d_sd)))
    waste_dr = float(np.sum(
        flow_dr * np.where(use_r_dr, data.dr_dr, data.d_dr)))
    waste      = waste_sd + waste_dr
    c_disposal = waste * DISPOSAL_C

    # Emissions
    emissions = float(
        np.sum(flow_sd * np.where(use_r_sd, data.er_sd, data.e_sd)) +
        np.sum(flow_dr * np.where(use_r_dr, data.er_dr, data.e_dr)))
    c_carbon  = emissions * LAMBDA_C

    # ── UPGRADE 5 : DC utilisation balance penalty ────────────────────────────
    # Penalty for each unit exceeding 90% of DC capacity
    c_util_pen = 0.0
    util_pcts  = {}
    for j in D_open:
        used     = data.dc_cap[j] - remaining_cap[j]
        util_pct = used / max(data.dc_cap[j], 1)
        util_pcts[j] = util_pct
        if util_pct > UTIL_HIGH_THRESH:
            overflow_units = (util_pct - UTIL_HIGH_THRESH) * data.dc_cap[j]
            c_util_pen    += overflow_units * UTIL_PENALTY_PER_UNIT

    # Avg utilisation for logging
    avg_util = float(np.mean(list(util_pcts.values()))) if util_pcts else 0.0

    # ── UPGRADE 6 : unmet demand — graded soft penalty ────────────────────────
    c_unmet = unmet * UNMET_PENALTY

    total_cost = (c_trans_sd + c_trans_dr + c_fixed +
                  c_disposal + c_carbon +
                  c_geo + c_util_pen + c_min_dc +
                  c_unmet + c_sup_exhaust)

    return {
        "total_cost":    total_cost,
        "waste":         waste,
        "emissions":     emissions,
        "c_transport":   c_trans_sd + c_trans_dr,
        "c_fixed":       c_fixed,
        "c_disposal":    c_disposal,
        "c_carbon":      c_carbon,
        "c_geo_penalty": c_geo,
        "c_util_penalty":c_util_pen,
        "c_min_dc_pen":  c_min_dc,
        "c_unmet":       c_unmet,
        "unmet_units":   unmet,
        "avg_util_pct":  avg_util,
        "feasible":      (unmet < 1.0 and n_open >= MIN_DC_OPEN),
        "dc_open":       [data.D[j] for j in D_open],
        "n_dc_open":     n_open,
    }


def _infeasible(c_min_dc=MIN_DC_OPEN * MIN_DC_PENALTY):
    """UPGRADE 6: soft infeasible — returns high but finite cost."""
    return {
        "total_cost": 1e10 + c_min_dc, "waste": 1e8, "emissions": 1e8,
        "c_transport": 0, "c_fixed": 0, "c_disposal": 0, "c_carbon": 0,
        "c_geo_penalty": 0, "c_util_penalty": 0,
        "c_min_dc_pen": c_min_dc, "c_unmet": 1e10,
        "unmet_units": 1e8, "avg_util_pct": 0.0,
        "feasible": False, "dc_open": [], "n_dc_open": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SHARED GENETIC OPERATORS
#     UPGRADE 2 : init / mutate / repair enforce >= MIN_DC_OPEN
# ─────────────────────────────────────────────────────────────────────────────
def _init_chromosome(nd, rng):
    """Random chromosome: MIN_DC_OPEN to 30 DCs open."""
    c = np.zeros(nd, dtype=int)
    n = rng.integers(MIN_DC_OPEN, 31)
    c[rng.choice(nd, n, replace=False)] = 1
    return c

def _repair(c, nd, rng):
    """UPGRADE 2: enforce minimum DC count after any operator."""
    deficit = MIN_DC_OPEN - int(c.sum())
    if deficit > 0:
        closed = np.where(c == 0)[0]
        c[rng.choice(closed, min(deficit, len(closed)), replace=False)] = 1
    return c

def _crossover(p1, p2, nd, rng):
    mask = rng.random(nd) < 0.5
    c1   = np.where(mask, p1, p2).copy()
    c2   = np.where(mask, p2, p1).copy()
    return _repair(c1, nd, rng), _repair(c2, nd, rng)

def _mutate(c, nd, rng):
    child = c.copy()
    child[rng.random(nd) < GA_MUTPB] ^= 1
    return _repair(child, nd, rng)

def _tournament(pop, scores, rng, k=3):
    idx  = rng.choice(len(pop), k, replace=False)
    best = idx[np.argmin(scores[idx])]
    return pop[best]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GENETIC ALGORITHM  (single-objective: Total Cost)
#     UPGRADE 9 : per-generation logging of all metrics
# ─────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm:

    def __init__(self, data):
        self.data = data
        self.nd   = data.nd
        self.rng  = np.random.default_rng(RANDOM_SEED)

    def run(self):
        print("\n" + "=" * 66)
        print("  GENETIC ALGORITHM  ▶  Minimize Total Cost  [v4.0]")
        print(f"  Pop={GA_POP}  Gens={GA_GENS}  CX={GA_CXPB}  "
              f"Mut={GA_MUTPB}  Elite={GA_ELITE}")
        print(f"  Min DCs={MIN_DC_OPEN}  DistPen_DR={MAX_DIST_DR}km  "
              f"UtilThresh={UTIL_HIGH_THRESH*100:.0f}%")
        print("  DC Capacity: HARD (0% overflow)  │  "
              "Refrig: DYNAMIC cost-benefit")
        print("=" * 66)
        t0 = time.time()

        pop  = [_init_chromosome(self.nd, self.rng) for _ in range(GA_POP)]
        fits = np.array([evaluate(c, self.data)["total_cost"] for c in pop])

        # UPGRADE 9 : rich log — one dict per generation
        log = []

        for gen in range(GA_GENS):
            elite_idx = np.argsort(fits)[:GA_ELITE]
            new_pop   = [pop[i].copy() for i in elite_idx]

            while len(new_pop) < GA_POP:
                p1 = _tournament(pop, fits, self.rng)
                p2 = _tournament(pop, fits, self.rng)
                if self.rng.random() < GA_CXPB:
                    c1, c2 = _crossover(p1, p2, self.nd, self.rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                new_pop.append(_mutate(c1, self.nd, self.rng))
                if len(new_pop) < GA_POP:
                    new_pop.append(_mutate(c2, self.nd, self.rng))

            pop  = new_pop
            # Evaluate and capture full metrics for best individual
            results  = [evaluate(c, self.data) for c in pop]
            fits     = np.array([r["total_cost"] for r in results])
            best_res = results[int(fits.argmin())]

            # UPGRADE 9 : log every generation
            log.append({
                "generation":  gen + 1,
                "best_cost":   best_res["total_cost"],
                "waste":       best_res["waste"],
                "emissions":   best_res["emissions"],
                "n_dcs":       best_res["n_dc_open"],
                "avg_util_pct":best_res["avg_util_pct"],
                "unmet":       best_res["unmet_units"],
                "c_geo":       best_res["c_geo_penalty"],
                "c_util":      best_res["c_util_penalty"],
                "feasible":    best_res["feasible"],
            })

            if (gen + 1) % 40 == 0 or gen == 0:
                print(f"  Gen {gen+1:>4}/{GA_GENS}  │  "
                      f"Cost ₹{best_res['total_cost']:>14,.0f}  │  "
                      f"Waste {best_res['waste']:>10,.0f}  │  "
                      f"Emit {best_res['emissions']:>9,.0f}  │  "
                      f"DCs {best_res['n_dc_open']:>3}  │  "
                      f"Util {best_res['avg_util_pct']*100:>5.1f}%  │  "
                      f"Unmet {best_res['unmet_units']:>6,.0f}  │  "
                      f"{time.time()-t0:.0f}s", flush=True)

        best_chrom = pop[int(fits.argmin())]
        result     = evaluate(best_chrom, self.data)
        result["log"]      = log
        result["time_sec"] = time.time() - t0
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NSGA-II  (multi-objective: Cost, Waste, Emissions)
#     UPGRADE 9 : per-generation logging
# ─────────────────────────────────────────────────────────────────────────────
class NSGA2:

    def __init__(self, data):
        self.data = data
        self.nd   = data.nd
        self.rng  = np.random.default_rng(RANDOM_SEED + 1)

    def _obj(self, c):
        r = evaluate(c, self.data)
        return np.array([r["total_cost"], r["waste"], r["emissions"]])

    @staticmethod
    def _fast_nds(obj):
        n   = len(obj)
        dom = [[] for _ in range(n)]
        cnt = np.zeros(n, dtype=int)
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                i_d_j = np.all(obj[i]<=obj[j]) and np.any(obj[i]<obj[j])
                j_d_i = np.all(obj[j]<=obj[i]) and np.any(obj[j]<obj[i])
                if i_d_j: dom[i].append(j)
                elif j_d_i: cnt[i] += 1
            if cnt[i] == 0: fronts[0].append(i)
        fi = 0
        while fronts[fi]:
            nxt = []
            for i in fronts[fi]:
                for j in dom[i]:
                    cnt[j] -= 1
                    if cnt[j] == 0: nxt.append(j)
            fi += 1; fronts.append(nxt)
        return [f for f in fronts if f]

    @staticmethod
    def _crowd(obj, front):
        n = len(front)
        if n <= 2: return np.full(n, np.inf)
        dist = np.zeros(n)
        for m in range(obj.shape[1]):
            vals  = np.array([obj[front[i], m] for i in range(n)])
            order = np.argsort(vals)
            dist[order[0]] = dist[order[-1]] = np.inf
            rng = vals[order[-1]] - vals[order[0]]
            if rng == 0: continue
            for i in range(1, n-1):
                dist[order[i]] += (vals[order[i+1]] - vals[order[i-1]]) / rng
        return dist

    def _tournament(self, pop, rank, crowd):
        a, b = self.rng.choice(len(pop), 2, replace=False)
        if rank[a] < rank[b]: return pop[a]
        if rank[b] < rank[a]: return pop[b]
        return pop[a] if crowd[a] >= crowd[b] else pop[b]

    def run(self):
        print("\n" + "=" * 66)
        print("  NSGA-II  ▶  Multi-Objective  (Cost | Waste | Emissions)  [v4.0]")
        print(f"  Pop={NS_POP}  Gens={NS_GENS}  Min DCs={MIN_DC_OPEN}")
        print("  DC Capacity: HARD  │  Refrig: DYNAMIC  │  "
              "Transport: Van/Truck")
        print("=" * 66)
        t0  = time.time()
        pop = [_init_chromosome(self.nd, self.rng) for _ in range(NS_POP)]
        print(f"  Evaluating initial population ({NS_POP}) …", flush=True)
        obj = np.array([self._obj(c) for c in pop])

        # UPGRADE 9 : NSGA-II log
        nsga_log = []

        for gen in range(NS_GENS):
            fronts = self._fast_nds(obj)
            rank   = np.zeros(len(pop), dtype=int)
            crowd  = np.zeros(len(pop))
            for fi, front in enumerate(fronts):
                for idx in front: rank[idx] = fi
                cd = self._crowd(obj, front)
                for li, idx in enumerate(front): crowd[idx] = cd[li]

            offspring = []
            while len(offspring) < NS_POP:
                p1 = self._tournament(pop, rank, crowd)
                p2 = self._tournament(pop, rank, crowd)
                c1, c2 = _crossover(p1, p2, self.nd, self.rng)
                offspring += [_mutate(c1, self.nd, self.rng),
                              _mutate(c2, self.nd, self.rng)]
            offspring = offspring[:NS_POP]
            obj_off   = np.array([self._obj(c) for c in offspring])

            comb_pop = pop + offspring
            comb_obj = np.vstack([obj, obj_off])
            fronts2  = self._fast_nds(comb_obj)

            new_pop, new_obj = [], []
            for front in fronts2:
                if len(new_pop) + len(front) <= NS_POP:
                    for idx in front:
                        new_pop.append(comb_pop[idx])
                        new_obj.append(comb_obj[idx])
                else:
                    needed = NS_POP - len(new_pop)
                    cd     = self._crowd(comb_obj, front)
                    for idx in [front[i] for i in np.argsort(-cd)[:needed]]:
                        new_pop.append(comb_pop[idx])
                        new_obj.append(comb_obj[idx])
                    break

            pop = new_pop
            obj = np.array(new_obj)

            # UPGRADE 9 : log Pareto front stats each generation
            pf_obj = comb_obj[fronts2[0]] if fronts2 else obj[:1]
            nsga_log.append({
                "generation":    gen + 1,
                "pareto_size":   len(fronts2[0]) if fronts2 else 0,
                "best_cost":     float(pf_obj[:, 0].min()),
                "best_waste":    float(pf_obj[:, 1].min()),
                "best_emissions":float(pf_obj[:, 2].min()),
            })

            if (gen + 1) % 30 == 0 or gen == 0:
                print(f"  Gen {gen+1:>4}/{NS_GENS}  │  "
                      f"Pareto {len(fronts2[0]):>3}  │  "
                      f"BestCost ₹{pf_obj[:,0].min():>13,.0f}  │  "
                      f"BestWaste {pf_obj[:,1].min():>11,.0f}  │  "
                      f"BestEmit {pf_obj[:,2].min():>9,.1f}  │  "
                      f"{time.time()-t0:.0f}s", flush=True)

        final_fronts = self._fast_nds(obj)
        pareto_idx   = final_fronts[0] if final_fronts else list(range(len(pop)))
        pareto_sols  = [evaluate(pop[i], self.data) for i in pareto_idx]
        return {
            "pareto_solutions": pareto_sols,
            "pareto_obj":       obj[pareto_idx],
            "nsga_log":         nsga_log,
            "time_sec":         time.time() - t0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RESULT PRINTERS  (updated for new fields)
# ─────────────────────────────────────────────────────────────────────────────
def print_ga(res):
    print("\n" + "=" * 66)
    print("  ✅  GA  —  BEST SOLUTION  [v4.0]")
    print("=" * 66)
    print(f"  Total Cost           ₹{res['total_cost']:>20,.2f}")
    print(f"   ├─ Transport        ₹{res['c_transport']:>20,.2f}")
    print(f"   ├─ Fixed DC         ₹{res['c_fixed']:>20,.2f}")
    print(f"   ├─ Disposal         ₹{res['c_disposal']:>20,.2f}")
    print(f"   ├─ Carbon (λ·CO2)   ₹{res['c_carbon']:>20,.2f}")
    print(f"   ├─ Geo Penalty      ₹{res['c_geo_penalty']:>20,.2f}")  # NEW
    print(f"   ├─ Util Penalty     ₹{res['c_util_penalty']:>20,.2f}") # NEW
    print(f"   ├─ Min-DC Penalty   ₹{res['c_min_dc_pen']:>20,.2f}")   # NEW
    print(f"   └─ Unmet Penalty    ₹{res['c_unmet']:>20,.2f}")
    print(f"  Total Waste           {res['waste']:>20,.2f}  units")
    print(f"  Total Emissions       {res['emissions']:>20,.4f}  kg CO2")
    print(f"  Unmet Demand          {res['unmet_units']:>20,.2f}  units")
    print(f"  DCs Opened            {res['n_dc_open']:>20d}  / 100"
          f"  (min={MIN_DC_OPEN})")
    print(f"  Avg DC Utilisation    {res['avg_util_pct']*100:>19.1f}%")
    print(f"  Feasible              {str(res['feasible']):>20}")
    print(f"  Runtime               {res['time_sec']:>20.1f}  sec")
    print(f"\n  DCs selected: {res['dc_open']}")

def print_nsga(res):
    sols = res["pareto_solutions"]
    obj  = res["pareto_obj"]
    print("\n" + "=" * 66)
    print("  ✅  NSGA-II  —  PARETO FRONT  [v4.0]")
    print("=" * 66)
    print(f"  Pareto solutions : {len(sols)}")
    print(f"  Runtime          : {res['time_sec']:.1f} sec\n")
    print(f"  {'#':>3}  {'Cost (₹)':>15}  {'Waste':>11}  "
          f"{'Emiss(kgCO2)':>13}  {'DCs':>4}  {'Util%':>6}  {'Feasible':>8}")
    print("  " + "─" * 66)
    order = np.argsort(obj[:, 0])
    for rank, i in enumerate(order[:20]):
        s = sols[i]
        print(f"  {rank+1:>3}  {s['total_cost']:>15,.0f}  "
              f"{s['waste']:>11,.0f}  {s['emissions']:>13.1f}  "
              f"{s['n_dc_open']:>4}  {s['avg_util_pct']*100:>5.1f}%"
              f"  {str(s['feasible']):>8}")
    print()
    print("  Best per objective:")
    for label, col in [("Min Cost   ", 0),
                        ("Min Waste  ", 1),
                        ("Min Emiss  ", 2)]:
        i = int(np.argmin(obj[:, col]))
        s = sols[i]
        print(f"  {label} → ₹{s['total_cost']:>13,.0f}  │  "
              f"Waste {s['waste']:>11,.0f}  │  "
              f"Emiss {s['emissions']:>9.1f}  │  "
              f"DCs {s['n_dc_open']:>3}  │  "
              f"Util {s['avg_util_pct']*100:>4.1f}%  │  "
              f"{s['dc_open'][:4]} …")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE OUTPUTS  (UPGRADE 9: save rich logs)
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(ga_res, ns_res):
    # GA rich generation log
    pd.DataFrame(ga_res["log"]).to_csv("ga_generation_log.csv", index=False)

    # GA best DC selection
    pd.DataFrame({
        "dc_id":    ga_res["dc_open"],
        "selected": 1,
    }).to_csv("ga_best_dc_selection.csv", index=False)

    # NSGA-II generation log
    pd.DataFrame(ns_res["nsga_log"]).to_csv("nsga2_generation_log.csv",
                                             index=False)

    # NSGA-II Pareto front (with new fields)
    rows = []
    for i, s in enumerate(ns_res["pareto_solutions"]):
        rows.append({
            "solution_id":   i + 1,
            "total_cost":    round(s["total_cost"], 2),
            "waste":         round(s["waste"], 2),
            "emissions":     round(s["emissions"], 4),
            "n_dcs_open":    s["n_dc_open"],
            "avg_util_pct":  round(s["avg_util_pct"] * 100, 1),
            "feasible":      s["feasible"],
            "unmet_units":   round(s["unmet_units"], 2),
            "c_geo_penalty": round(s["c_geo_penalty"], 2),
            "c_util_penalty":round(s["c_util_penalty"], 2),
            "dcs_open":      "|".join(s["dc_open"]),
        })
    pd.DataFrame(rows).to_csv("nsga2_pareto.csv", index=False)

    print("\n  Output files saved:")
    print("    ✓ ga_generation_log1.csv       ← cost/waste/emit/util per gen")
    print("    ✓ ga_best_dc_selection1.csv")
    print("    ✓ nsga2_generation_log1.csv    ← Pareto stats per gen")
    print("    ✓ nsga2_pareto1.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█" * 66)
    print("   INDIA VACCINE SUPPLY CHAIN — META-HEURISTIC OPTIMIZER  v4.0")
    print("   S(15) → DC(100) → R(500)  │  GA + NSGA-II")
    print("   Upgrades: GeoRealism │ MinDC │ DynDecay │ DynRefrig │")
    print("             UtilBalance │ SoftConstr │ Van/Truck │ Precompute")
    print("█" * 66)

    data = SupplyChainData(DATA_DIR)

    ga_result = GeneticAlgorithm(data).run()
    print_ga(ga_result)

    ns_result = NSGA2(data).run()
    print_nsga(ns_result)

    save_outputs(ga_result, ns_result)
    print("\n  All done! ✅\n")

    
