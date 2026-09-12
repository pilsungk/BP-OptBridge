#!/usr/bin/env python3
# e1_association.py
#
# E1 Adjudication Protocol v1.8 -- full-route association analysis.
#
# Runs only after the estimability precheck has selected a route and the
# protocol has been frozen. Implements Sections 7, 10-14 of the protocol:
#
#   7   within-seed partial Spearman for rho_nu, rho_cos, rho_C
#   10  Fisher-z seed aggregation and seed-level bootstrap CIs
#   11  cell-level material-and-resolved criterion
#   12  cross-cell 9/12 reproducibility (denominator always 12)
#   13  Gate-A verdict, with the lso_pcgrad safeguard of 13.1
#   14  exact log-variance audit of the previous scale variable
#
# Estimability is decided exactly as in the precheck: the same structural
# constants, the same VIF/condition-number thresholds, the same 20/30
# rule. Statistics are computed only where the design is estimable.
#
# Usage:
#   python e1_association.py \
#       --hea results/v3/hea/trajectory_per_seed.csv.gz \
#       --hva results/v3/hva/trajectory_per_seed.csv.gz \
#       --outdir e1_association_out

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROTOCOL_VERSION = "v1.8"
IMPLEMENTATION_REVISION = "1.1"

# ---- frozen constants (Sections 5, 9, 10, 11, 12) -------------------
EPS = 1e-15
EPS_CONST = 1e-12
VIF_MAX = 10.0
KAPPA_MAX = 30.0
MIN_ESTIMABLE_SEEDS = 20
N_SEEDS_EXPECTED = 30
N_CELLS_PER_RULE = 12          # Section 12: denominator is always 12
MIN_REPRODUCIBLE_CELLS = 9     # 9/12
RHO_MATERIAL = 0.20            # Section 11
N_BOOT = 5000
BOOT_SEED = 20260829

RUN_KEYS = ["ansatz", "variant", "n_qubits", "depth", "seed"]
CELL_KEYS = ["ansatz", "variant", "n_qubits", "depth"]

# ---- Section 6 rule classification ---------------------------------
OPTIMIZER_CONTROLLED = ["lso_pcgrad", "lso_grid", "vanilla_ls_nm"]
RULE_DETERMINED = ["pcgrad"]
STRUCTURALLY_CONSTRAINED = ["vanilla", "pcgrad_nm"]

# Section 0.2: pre-classified ineligible for A1/A2 on coverage grounds.
INELIGIBLE_FOR_GATE = ["lso_pcgrad"]

# Section 6.1 declared structural constants.
STRUCTURAL_CONSTANTS = {
    "vanilla": {"nu": 1.0, "cos_theta": 1.0},
    "pcgrad_nm": {"nu": 1.0},
    "vanilla_ls_nm": {"cos_theta": 1.0},
}

# Section 7 primary statistics: name -> (target, controls)
FULL_STATS = {
    "rho_nu": ("nu", ["G", "cos_theta", "C"]),
    "rho_cos": ("cos_theta", ["G", "nu", "C"]),
    "rho_C": ("C", ["G", "nu", "cos_theta"]),
}

# Section 16 pre-registered secondary reference: the cleanest rho_C
# design, available because vanilla fixes both nu and cos(theta).
SECONDARY_STATS = {
    "rho_C_vanilla_ref": ("C", ["G"]),
}

REQUIRED_COLUMNS = [
    "variant", "n_qubits", "depth", "seed", "step", "energy",
    "U_step", "Q_step", "update_norm", "grad_norm_l2", "dir_sum_step",
]


# ====================================================================
# data preparation (identical derivation to the precheck)
# ====================================================================

def load(path, ansatz):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("{0}: missing required columns: {1}".format(
            path, ", ".join(missing)))
    df["ansatz"] = ansatz
    return df.sort_values(RUN_KEYS + ["step"]).reset_index(drop=True)


def derive(df):
    g = df.groupby(RUN_KEYS, sort=False)
    df["dE_act"] = g["energy"].transform(lambda s: s - s.shift(-1))
    df["is_transition"] = df["dE_act"].notna()

    R = df["update_norm"].to_numpy(dtype=float)
    G = df["grad_norm_l2"].to_numpy(dtype=float)
    Q = np.clip(df["Q_step"].to_numpy(dtype=float), 0.0, None)
    U = df["U_step"].to_numpy(dtype=float)
    gTu = df["dir_sum_step"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        sqrtQ = np.sqrt(Q)
        A = np.where(R > EPS, sqrtQ / R, np.nan)
        nu = np.where(G > EPS, R / G, np.nan)
        cos = np.where((G > EPS) & (R > EPS), gTu / (G * R), np.nan)
        C = np.where((np.abs(U) > EPS) & (A > EPS),
                     np.log(np.abs(U) / A), np.nan)

    df["R"] = R
    df["G"] = G
    df["A"] = A
    df["nu"] = nu
    df["cos_theta"] = cos
    df["C"] = C
    df["sqrtQ"] = sqrtQ

    df["valid_base"] = df["is_transition"] & (R > EPS) & (G > EPS)
    df["valid_C"] = df["valid_base"] & np.isfinite(df["C"])
    return df


# ====================================================================
# Section 7 + 9: one seed-level partial Spearman, with estimability
# ====================================================================

def stable_bootstrap_seed(ansatz, variant, n, d, stat):
    """Deterministic per-cell bootstrap seed, stable across processes."""
    key = "{0}|{1}|{2}|{3}|{4}".format(
        ansatz, variant, int(n), int(d), stat)
    digest = hashlib.sha256(key.encode("ascii")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int((BOOT_SEED + offset) % (2 ** 32 - 1))


def design_vif_and_kappa(X):
    """VIF and condition number for a rank-standardized design.

    X contains the target as the first column and the active controls in
    the remaining columns. Structural constants must already be removed.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if p == 0:
        return np.nan, np.nan
    if p == 1:
        return 1.0, 1.0

    Xr = np.column_stack([rankdata(X[:, j]) for j in range(p)])
    sd = Xr.std(axis=0)
    if np.any(sd <= EPS):
        return np.inf, np.inf
    Xs = (Xr - Xr.mean(axis=0)) / sd

    vifs = []
    for j in range(p):
        yj = Xs[:, j]
        Zj = np.column_stack([np.ones(n), np.delete(Xs, j, axis=1)])
        b, *_ = np.linalg.lstsq(Zj, yj, rcond=None)
        resid = yj - Zj @ b
        sst = float(((yj - yj.mean()) ** 2).sum())
        if sst <= EPS:
            return np.inf, np.inf
        r2 = 1.0 - float((resid ** 2).sum()) / sst
        vifs.append(1.0 / max(1.0 - r2, 1e-12))

    sv = np.linalg.svd(Xs, compute_uv=False)
    kappa = float(sv[0] / sv[-1]) if sv[-1] > EPS else np.inf
    return float(max(vifs)), kappa


def prepare_statistic_design(sub, variant, target, controls):
    """Apply the frozen statistic-specific estimability rules.

    The target is never silently dropped. Declared structural controls are
    removed first. Ordinary constant controls and duplicate controls may be
    removed as nuisance columns. An exact target-control rank duplicate makes
    the requested partial statistic unidentified.
    """
    structural = STRUCTURAL_CONSTANTS.get(variant, {})
    notes = []

    if target in structural:
        return None, {"status": "structural_target", "notes": [target]}

    active = []
    for c in controls:
        if c in structural:
            notes.append(c + "(structural_control)")
        else:
            active.append(c)

    need = [target] + active + ["dE_act"]
    s = sub.dropna(subset=need).copy()

    # Algebraic residual df for a partial correlation is n - p - 2.
    # No arbitrary minimum-row threshold is introduced.
    if len(s) - len(active) - 2 <= 0:
        return None, {"status": "insufficient_residual_df",
                      "notes": notes, "n_rows": int(len(s))}

    x = s[target].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or np.std(x) <= EPS:
        return None, {"status": "constant_or_nonfinite_target",
                      "notes": notes, "n_rows": int(len(s))}
    xr = rankdata(x)

    kept = []
    kept_ranks = []
    for c in active:
        v = s[c].to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):
            notes.append(c + "(nonfinite_control)")
            continue
        if np.std(v) <= EPS:
            notes.append(c + "(constant_control)")
            continue
        vr = rankdata(v)

        if abs(np.corrcoef(vr, xr)[0, 1]) > 1.0 - 1e-12:
            return None, {"status": "target_control_duplicate",
                          "notes": notes + [c], "n_rows": int(len(s))}

        duplicate = False
        for prev_name, prev_rank in zip(kept, kept_ranks):
            if abs(np.corrcoef(vr, prev_rank)[0, 1]) > 1.0 - 1e-12:
                notes.append(c + "(" + prev_name + "_duplicate)")
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
            kept_ranks.append(vr)

    if len(s) - len(kept) - 2 <= 0:
        return None, {"status": "insufficient_residual_df",
                      "notes": notes, "n_rows": int(len(s))}

    design = s[[target] + kept].to_numpy(dtype=float)
    max_vif, kappa = design_vif_and_kappa(design)
    if max_vif > VIF_MAX or kappa > KAPPA_MAX:
        return None, {"status": "collinear", "notes": notes,
                      "max_vif": max_vif, "kappa": kappa,
                      "n_rows": int(len(s)), "controls": kept}

    return (s, kept), {"status": "estimable", "notes": notes,
                       "max_vif": max_vif, "kappa": kappa,
                       "n_rows": int(len(s)), "controls": kept}


def partial_spearman(sub, variant, target, controls):
    """Return one seed-level partial Spearman rho or NaN if inestimable."""
    prepared, diag = prepare_statistic_design(sub, variant, target, controls)
    if prepared is None:
        return np.nan, diag

    s, kept = prepared
    y = rankdata(s["dE_act"].to_numpy(dtype=float))
    x = rankdata(s[target].to_numpy(dtype=float))

    if kept:
        Z = np.column_stack([rankdata(s[c].to_numpy(dtype=float))
                             for c in kept])
        M = np.column_stack([np.ones(len(s)), Z])
        bx, *_ = np.linalg.lstsq(M, x, rcond=None)
        by, *_ = np.linalg.lstsq(M, y, rcond=None)
        rx = x - M @ bx
        ry = y - M @ by
    else:
        rx = x - x.mean()
        ry = y - y.mean()

    if np.std(rx) <= EPS or np.std(ry) <= EPS:
        diag = dict(diag)
        diag["status"] = "degenerate_residual"
        return np.nan, diag

    rho = float(np.corrcoef(rx, ry)[0, 1])
    diag = dict(diag)
    diag["status"] = "ok"
    return rho, diag


# ====================================================================
# Section 10: Fisher-z aggregation and seed bootstrap
# ====================================================================

def aggregate_seeds(rhos, boot_seed):
    r = np.asarray([x for x in rhos if np.isfinite(x)], dtype=float)
    if len(r) == 0:
        return {"rho_bar": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_seeds": 0}
    r = np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12)
    z = np.arctanh(r)
    rho_bar = float(np.tanh(z.mean()))

    rng = np.random.RandomState(boot_seed)
    idx = rng.randint(0, len(z), size=(N_BOOT, len(z)))
    reps = np.tanh(z[idx].mean(axis=1))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"rho_bar": rho_bar, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_seeds": int(len(r))}


def material_and_resolved(agg, n_est_seeds):
    """Section 11, with the Section 9 cell-estimability precondition."""
    if n_est_seeds < MIN_ESTIMABLE_SEEDS:
        return False, "cell_underdetermined"
    if not np.isfinite(agg["rho_bar"]):
        return False, "no_estimate"
    if abs(agg["rho_bar"]) < RHO_MATERIAL:
        return False, "below_material_threshold"
    if agg["ci_lo"] <= 0.0 <= agg["ci_hi"]:
        return False, "ci_includes_zero"
    return True, "material_and_resolved"


# ====================================================================
# association pass
# ====================================================================

def run_associations(df, outdir):
    rows = []
    for cell, sub_cell in df.groupby(CELL_KEYS, sort=True):
        ansatz, variant, n, d = cell
        specs = dict(FULL_STATS)
        if variant == "vanilla":
            specs.update(SECONDARY_STATS)

        for stat, (target, controls) in specs.items():
            rhos, statuses, vifs, kappas = [], [], [], []
            for seed, sub in sub_cell.groupby("seed", sort=True):
                s = sub[sub["valid_C"]] if ("C" in [target] + controls) \
                    else sub[sub["valid_base"]]
                rho, diag = partial_spearman(s, variant, target, controls)
                rhos.append(rho)
                statuses.append(diag["status"])
                if np.isfinite(diag.get("max_vif", np.nan)):
                    vifs.append(diag["max_vif"])
                if np.isfinite(diag.get("kappa", np.nan)):
                    kappas.append(diag["kappa"])

            n_est = int(np.sum(np.isfinite(rhos)))
            bseed = stable_bootstrap_seed(ansatz, variant, n, d, stat)
            agg = aggregate_seeds(rhos, bseed)
            ok, why = material_and_resolved(agg, n_est)

            rows.append({
                "ansatz": ansatz, "variant": variant,
                "n_qubits": int(n), "depth": int(d), "statistic": stat,
                "seeds_expected": N_SEEDS_EXPECTED,
                "seeds_estimable": n_est,
                "cell_estimable": bool(n_est >= MIN_ESTIMABLE_SEEDS),
                "rho_bar": agg["rho_bar"],
                "ci_lo": agg["ci_lo"], "ci_hi": agg["ci_hi"],
                "material_and_resolved": ok,
                "reason": why,
                "sign": (int(np.sign(agg["rho_bar"]))
                         if np.isfinite(agg["rho_bar"]) else 0),
                "max_vif_median": (float(np.median(vifs)) if vifs else np.nan),
                "kappa_median": (float(np.median(kappas)) if kappas else np.nan),
                "top_status": pd.Series(statuses).mode().iat[0],
                "bootstrap_seed": bseed,
            })

    out = pd.DataFrame(rows)
    path = os.path.join(outdir, "cell_associations.csv")
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out


# ====================================================================
# Section 12: cross-cell reproducibility
# ====================================================================

def reproducibility(cells, outdir):
    rows = []
    for (ansatz, variant, stat), sub in cells.groupby(
            ["ansatz", "variant", "statistic"], sort=True):
        pos = int(((sub["material_and_resolved"]) & (sub["sign"] > 0)).sum())
        neg = int(((sub["material_and_resolved"]) & (sub["sign"] < 0)).sum())
        dominant = max(pos, neg)
        rows.append({
            "ansatz": ansatz, "variant": variant, "statistic": stat,
            "cells_total": N_CELLS_PER_RULE,
            "cells_observed": int(len(sub)),
            "cells_estimable": int(sub["cell_estimable"].sum()),
            "cells_material_positive": pos,
            "cells_material_negative": neg,
            "cells_same_direction": dominant,
            "direction": ("+" if pos > neg else
                          ("-" if neg > pos else "")),
            # Section 12: denominator is always 12, never the estimable count
            "reproducible": bool(dominant >= MIN_REPRODUCIBLE_CELLS),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(outdir, "reproducibility.csv")
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out


# ====================================================================
# Section 14: exact log-variance audit of the previous scale variable
# ====================================================================

def variance_share_triplet(x, y):
    """Return shares for Var(x+y), including the covariance share."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return None
    total = float(np.var(x + y))
    if total <= EPS:
        return None
    sx = float(np.var(x)) / total
    sy = float(np.var(y)) / total
    scov = 2.0 * float(np.cov(x, y, bias=True)[0, 1]) / total
    return sx, sy, scov, sx + sy + scov


def variance_audit(df, outdir):
    rows = []
    tr = df[df["valid_base"] & (df["A"] > EPS)]
    for cell, sub_cell in tr.groupby(CELL_KEYS, sort=True):
        ansatz, variant, n, d = cell
        sqrtq_parts, r_parts = [], []

        for seed, s in sub_cell.groupby("seed", sort=True):
            R = s["R"].to_numpy(dtype=float)
            A = s["A"].to_numpy(dtype=float)
            G = s["G"].to_numpy(dtype=float)
            nu = s["nu"].to_numpy(dtype=float)

            m1 = ((R > 0.0) & (A > 0.0) &
                  np.isfinite(R) & np.isfinite(A))
            p1 = variance_share_triplet(np.log(R[m1]), np.log(A[m1]))
            if p1 is not None:
                sqrtq_parts.append(p1)

            m2 = ((G > 0.0) & (nu > 0.0) &
                  np.isfinite(G) & np.isfinite(nu))
            p2 = variance_share_triplet(np.log(G[m2]), np.log(nu[m2]))
            if p2 is not None:
                r_parts.append(p2)

        row = {
            "ansatz": ansatz, "variant": variant,
            "n_qubits": int(n), "depth": int(d),
            "n_seeds_sqrtQ_audit": len(sqrtq_parts),
            "n_seeds_R_audit": len(r_parts),
        }

        if sqrtq_parts:
            a = np.asarray(sqrtq_parts, dtype=float)
            row.update({
                "s_R_median": float(np.median(a[:, 0])),
                "s_A_median": float(np.median(a[:, 1])),
                "s_RA_cov_median": float(np.median(a[:, 2])),
                "s_cov_median": float(np.median(a[:, 2])),
                "sum_check_median": float(np.median(a[:, 3])),
            })
        else:
            row.update({
                "s_R_median": np.nan, "s_A_median": np.nan,
                "s_RA_cov_median": np.nan, "s_cov_median": np.nan,
                "sum_check_median": np.nan,
            })

        if r_parts:
            b = np.asarray(r_parts, dtype=float)
            row.update({
                "s_G_within_R_median": float(np.median(b[:, 0])),
                "s_nu_within_R_median": float(np.median(b[:, 1])),
                "s_Gnu_cov_within_R_median": float(np.median(b[:, 2])),
                "sum_check_R_median": float(np.median(b[:, 3])),
            })
        else:
            row.update({
                "s_G_within_R_median": np.nan,
                "s_nu_within_R_median": np.nan,
                "s_Gnu_cov_within_R_median": np.nan,
                "sum_check_R_median": np.nan,
            })
        rows.append(row)

    out = pd.DataFrame(rows)
    path = os.path.join(outdir, "log_variance_audit.csv")
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out


# ====================================================================
# Section 13: Gate-A verdict
# ====================================================================

def gate_a(repro, outdir):
    L = ["E1 GATE-A ADJUDICATION (protocol {0}, implementation {1})".format(
             PROTOCOL_VERSION, IMPLEMENTATION_REVISION),
         "=" * 72, ""]

    eligible = [r for r in OPTIMIZER_CONTROLLED if r not in INELIGIBLE_FOR_GATE]
    L.append("Eligible optimizer-controlled rules: " + ", ".join(eligible))
    L.append("Ineligible on frozen coverage grounds: " +
             ", ".join(INELIGIBLE_FOR_GATE))
    L.append("The eligible pool has exactly {0} rules; A1/A2 therefore "
             "require 2-of-2 unanimity.".format(len(eligible)))
    L.append("")

    def qualifies(stat):
        """Reproducible in both ansatzes for each eligible rule, same sign."""
        detail, ok_rules = [], []
        for rule in eligible:
            per = {}
            for ansatz in ("hea", "hva"):
                sel = repro[(repro["variant"] == rule) &
                            (repro["ansatz"] == ansatz) &
                            (repro["statistic"] == stat)]
                if sel.empty:
                    per[ansatz] = (False, "", 0, 0)
                else:
                    r = sel.iloc[0]
                    per[ansatz] = (bool(r["reproducible"]),
                                   str(r["direction"]),
                                   int(r["cells_same_direction"]),
                                   int(r["cells_estimable"]))
            both = per["hea"][0] and per["hva"][0]
            same = (per["hea"][1] == per["hva"][1] and
                    per["hea"][1] != "")
            detail.append(
                "  {0:14s} hea {1:2d}/{2} {3:1s} est {4:2d}/{2} | "
                "hva {5:2d}/{2} {6:1s} est {7:2d}/{2} -> {8}".format(
                    rule, per["hea"][2], N_CELLS_PER_RULE,
                    per["hea"][1] or ".", per["hea"][3],
                    per["hva"][2], per["hva"][1] or ".", per["hva"][3],
                    "reproducible both" if both and same else "no"))
            if both and same:
                ok_rules.append((rule, per["hea"][1]))
        return ok_rules, detail

    nu_rules, nu_detail = qualifies("rho_nu")
    c_rules, c_detail = qualifies("rho_C")

    L.append("rho_nu across eligible rules:")
    L.extend(nu_detail)
    L.append("")
    L.append("rho_C across eligible rules:")
    L.extend(c_detail)
    L.append("")

    # A2 has no pre-specified positive/negative direction, but every
    # qualifying rule/ansatz result must agree in direction.
    a2 = (len(c_rules) >= len(eligible) and
          len(set(d for _, d in c_rules)) == 1)

    # dE_act = E_t - E_{t+1}; positive means improved realized descent.
    # Therefore A1 requires unanimous positive rho_nu, not merely a
    # common sign. A1 also retains the protocol condition that A2 fails.
    nu_positive_unanimous = (
        len(nu_rules) >= len(eligible) and
        all(d == "+" for _, d in nu_rules))
    a1 = nu_positive_unanimous and not a2

    # A3/A4 may use only eligible optimizer-controlled rules and the two
    # primary Gate-A statistics. Secondary vanilla, rule-determined PCGrad,
    # and ineligible lso_pcgrad results cannot change the Gate-A verdict.
    eligible_primary = repro[
        repro["variant"].isin(eligible) &
        repro["statistic"].isin(["rho_nu", "rho_C"])]
    any_estimable = bool(
        len(eligible_primary) > 0 and
        eligible_primary["cells_estimable"].max() >= MIN_REPRODUCIBLE_CELLS)

    if a1:
        verdict = ("A1 - RELATIVE-STEP-SIZE DOMINANT: positive rho_nu is "
                   "reproducible in both ansatzes for every eligible rule, "
                   "and rho_C does not meet the A2 criterion.")
    elif a2:
        verdict = ("A2 - COMPOSITION INFORMATIVE BEYOND STANDARD GEOMETRY: "
                   "rho_C is reproducible in both ansatzes for every eligible "
                   "rule with a consistent direction.")
    elif not any_estimable:
        verdict = ("A4-U - STATISTICALLY UNDERDETERMINED: no eligible primary "
                   "Gate-A analysis reaches the estimability coverage required "
                   "to adjudicate.")
    elif (any(len(x) > 0 for x in (nu_rules, c_rules)) or
          (eligible_primary["cells_same_direction"] > 0).any()):
        verdict = ("A3 - REGIME DEPENDENT: resolved structure exists within "
                   "eligible primary analyses but does not hold uniformly "
                   "across the eligible rules and ansatzes, so neither A1 nor "
                   "A2 provides a common description.")
    else:
        verdict = ("A4-N - NULL OR INCONCLUSIVE: the eligible primary analysis "
                   "is estimable but neither nu nor C produces reproducible "
                   "material associations beyond the controlled geometry.")

    L.append("VERDICT: " + verdict)
    L.append("")

    if a2 and nu_positive_unanimous:
        L.append("Priority note: rho_nu also satisfies its unanimous positive "
                 "pattern, but A1 is definitionally unavailable because A1 "
                 "requires rho_C not to satisfy A2. The verdict remains A2.")
        L.append("")

    L.append("Section 13.1 safeguard: lso_pcgrad remains excluded from A1/A2 "
             "because its frozen precheck coverage cannot satisfy the 9/12 "
             "criterion in both ansatzes. Its results are descriptive only, "
             "and lso_pcgrad-centered manuscript claims remain unresolved "
             "regardless of this verdict.")
    L.append("")
    L.append("Section 17 scope: a reproducible rho_C is an association, not an "
             "identified mechanism. Mechanism-level claims require the "
             "matched-standard-geometry intervention and directional-curvature "
             "validation reserved for future work.")

    txt = "\n".join(L)
    path = os.path.join(outdir, "gate_a_verdict.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("Saved:", path)
    print()
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hea", required=True)
    ap.add_argument("--hva", required=True)
    ap.add_argument("--outdir", default="e1_association_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = derive(pd.concat([load(args.hea, "hea"), load(args.hva, "hva")],
                          ignore_index=True))
    print("records: {0} rows, {1} runs".format(
        len(df), df.groupby(RUN_KEYS).ngroups))
    print("protocol: {0}; implementation revision: {1}".format(
        PROTOCOL_VERSION, IMPLEMENTATION_REVISION))
    print()

    cells = run_associations(df, args.outdir)
    repro = reproducibility(cells, args.outdir)
    variance_audit(df, args.outdir)
    print()
    gate_a(repro, args.outdir)


if __name__ == "__main__":
    main()
