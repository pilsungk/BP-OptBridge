#!/usr/bin/env python3
# e1_precheck.py
#
# E1 Adjudication Protocol v1.5 -- Section 9 / 9.1 estimability precheck.
#
# This script decides the analysis ROUTE (E1-lite versus full E1). It
# deliberately does NOT compute any association between the predictors
# and realized descent: seeing rho before the route is fixed would
# defeat the freeze rule of Section 18.
#
# Input : the canonical V3 trajectory records, one per ansatz.
# Output: identity/validity audits, statistic-specific estimability
#         tables for both candidate routes, and the route comparison.
#
# Usage:
#   python e1_precheck.py \
#       --hea results/v3/hea/trajectory_per_seed.csv.gz \
#       --hva results/v3/hva/trajectory_per_seed.csv.gz \
#       --outdir e1_precheck_out

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROTOCOL_VERSION = "v1.6"

# Section 9 thresholds, pre-specified.
VIF_MAX = 10.0
KAPPA_MAX = 30.0
MIN_ESTIMABLE_SEEDS = 20
N_SEEDS_EXPECTED = 30

# Section 5 operational numerical-zero threshold.
EPS = 1e-15

# Section 6.1 structural-constant audit tolerance.
EPS_CONST = 1e-12

RUN_KEYS = ["ansatz", "variant", "n_qubits", "depth", "seed"]
CELL_KEYS = ["ansatz", "variant", "n_qubits", "depth"]

# Section 6 classification by relative step-size status.
STRUCTURALLY_CONSTRAINED = ["vanilla", "pcgrad_nm"]  # nu == 1
OPTIMIZER_CONTROLLED = ["lso_pcgrad", "lso_grid", "vanilla_ls_nm"]
RULE_DETERMINED = ["pcgrad"]

# Section 6.1 declared structural constants.
# These are removed by rule definition, not inferred from numerical SD.
STRUCTURAL_CONSTANTS = {
    "vanilla": {"nu": 1.0, "cos_theta": 1.0},
    "pcgrad_nm": {"nu": 1.0},
    "vanilla_ls_nm": {"cos_theta": 1.0},
}

# Statistic-specific estimability specifications.
# Each entry is: statistic_name: (target, controls)
LITE_STATS = {
    "rho_A_lite": ("A", ["G", "nu"]),
    "rho_nu_lite": ("nu", ["G", "A"]),
}
FULL_STATS = {
    "rho_nu_full": ("nu", ["G", "cos_theta", "C"]),
    "rho_cos_full": ("cos_theta", ["G", "nu", "C"]),
    "rho_C_full": ("C", ["G", "nu", "cos_theta"]),
}

REQUIRED_COLUMNS = [
    "variant",
    "n_qubits",
    "depth",
    "seed",
    "step",
    "energy",
    "U_step",
    "Q_step",
    "update_norm",
    "grad_norm_l2",
    "dir_sum_step",
]


def load(path, ansatz):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "{0}: missing required columns: {1}".format(
                path, ", ".join(missing)
            )
        )
    df["ansatz"] = ansatz
    return df.sort_values(RUN_KEYS + ["step"]).reset_index(drop=True)


def derive(df):
    """Derive Section 4 quantities and Section 5 validity flags.

    cos(theta) is computed independently from recorded g^T u
    (dir_sum_step), so UA = G cos(theta) remains a genuine identity audit.
    """
    g = df.groupby(RUN_KEYS, sort=False)

    # Realized decrease: E_t - E_{t+1}; final state has no transition.
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
        cos = np.where(
            (G > EPS) & (R > EPS), gTu / (G * R), np.nan
        )
        cos_ua = np.where(G > EPS, (U * A) / G, np.nan)
        C = np.where(
            (np.abs(U) > EPS) & (A > EPS),
            np.log(np.abs(U) / A),
            np.nan,
        )

    df["R"] = R
    df["G"] = G
    df["A"] = A
    df["nu"] = nu
    df["cos_theta"] = cos
    df["cos_theta_from_UA"] = cos_ua
    df["C"] = C

    df["valid_base"] = (
        df["is_transition"] & (R > EPS) & (G > EPS)
    )
    df["valid_C"] = df["valid_base"] & np.isfinite(df["C"])
    return df


def identity_audit(df, outdir):
    """Independent consistency checks on the derived coordinates."""
    tr = df[df["valid_base"]]
    if tr.empty:
        raise ValueError("No valid transitions are available for identity audit.")

    lines = [
        "DERIVED-QUANTITY IDENTITY AUDIT",
        "=" * 60,
        "",
        "protocol: {0}".format(PROTOCOL_VERSION),
        "transitions audited: {0}".format(len(tr)),
        "",
    ]

    ua_diff = np.abs(
        tr["cos_theta"].to_numpy(dtype=float)
        - tr["cos_theta_from_UA"].to_numpy(dtype=float)
    )
    lines.append(
        "UA = G cos(theta)   max |cos - UA/G|          = {0:.3e}".format(
            float(np.nanmax(ua_diff))
        )
    )

    if "norm_ratio" in tr.columns:
        nr_diff = np.abs(
            tr["nu"].to_numpy(dtype=float)
            - tr["norm_ratio"].to_numpy(dtype=float)
        )
        lines.append(
            "nu vs norm_ratio    max |nu - norm_ratio|     = {0:.3e}".format(
                float(np.nanmax(nr_diff))
            )
        )
    else:
        lines.append("nu vs norm_ratio    recorded column absent")

    sq = np.sqrt(
        np.clip(tr["Q_step"].to_numpy(dtype=float), 0.0, None)
    )
    ra_diff = np.abs(
        sq
        - tr["R"].to_numpy(dtype=float)
        * tr["A"].to_numpy(dtype=float)
    )
    lines.append(
        "sqrt(Q) = R A       max |sqrt(Q) - R A|       = {0:.3e}".format(
            float(np.nanmax(ra_diff))
        )
    )

    cos_vals = tr["cos_theta"].to_numpy(dtype=float)
    n_cos_bad = int((np.abs(cos_vals) > 1.0 + 1e-9).sum())
    lines.append(
        "|cos(theta)| <= 1   violations                 = {0} of {1}".format(
            n_cos_bad, len(tr)
        )
    )
    lines.append("")
    lines.append(
        "Numerical zero threshold for validity checks: EPS = {0:g}".format(
            EPS
        )
    )

    txt = "\n".join(lines)
    path = os.path.join(outdir, "identity_audit.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("Saved:", path)
    print(txt)
    print()



def structural_constant_audit(df, outdir):
    """Audit declared structural constants without using them for discovery."""
    rows = []

    for variant, constants in STRUCTURAL_CONSTANTS.items():
        sub = df[
            (df["variant"] == variant) & df["valid_base"]
        ]

        for coord, expected in constants.items():
            vals = sub[coord].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                max_rel = np.nan
                n_bad = 0
                passed = False
            else:
                denom = max(abs(expected), 1.0)
                rel = np.abs(vals - expected) / denom
                max_rel = float(np.max(rel))
                n_bad = int(np.sum(rel > EPS_CONST))
                passed = bool(n_bad == 0)

            rows.append({
                "variant": variant,
                "coordinate": coord,
                "expected_value": expected,
                "n_values": int(len(vals)),
                "max_relative_deviation": max_rel,
                "n_violations": n_bad,
                "tolerance": EPS_CONST,
                "audit_pass": passed,
            })

    out = pd.DataFrame(rows)
    path = os.path.join(outdir, "structural_constant_audit.csv")
    out.to_csv(path, index=False)
    print("Saved:", path)

    print("STRUCTURAL-CONSTANT AUDIT")
    print("=" * 60)
    for _, r in out.iterrows():
        print(
            "{0:14s} {1:10s} expected={2:.1f} "
            "max_rel={3:.3e} violations={4}/{5} pass={6}".format(
                str(r["variant"]),
                str(r["coordinate"]),
                float(r["expected_value"]),
                float(r["max_relative_deviation"])
                if np.isfinite(r["max_relative_deviation"])
                else float("nan"),
                int(r["n_violations"]),
                int(r["n_values"]),
                bool(r["audit_pass"]),
            )
        )
    print(
        "Structural constants are removed by rule definition regardless "
        "of floating-point residual variation."
    )
    print()

    return out

def vif_and_kappa(X):
    """Return max VIF and condition number after rank standardization."""
    n, p = X.shape
    if p == 0:
        return np.nan, np.nan

    Xr = np.column_stack([rankdata(X[:, j]) for j in range(p)])
    sd = Xr.std(axis=0)
    if np.any(sd <= EPS):
        return np.inf, np.inf
    Xs = (Xr - Xr.mean(axis=0)) / sd

    if p == 1:
        return 1.0, 1.0

    vifs = []
    for j in range(p):
        y = Xs[:, j]
        Z = np.column_stack([np.ones(n), np.delete(Xs, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot <= EPS:
            return np.inf, np.inf
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
        vifs.append(1.0 / max(1.0 - r2, 1e-12))

    sv = np.linalg.svd(Xs, compute_uv=False)
    kappa = float(sv[0] / sv[-1]) if sv[-1] > EPS else np.inf
    return float(max(vifs)), kappa


def ranked_duplicate(x, y):
    """True for exact monotone duplicate/anti-duplicate rank coordinates."""
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    if rx.std() <= EPS or ry.std() <= EPS:
        return False
    corr = float(np.corrcoef(rx, ry)[0, 1])
    return abs(corr) > 1.0 - 1e-12


def assess_statistic(sub, variant, target, controls):
    """Assess one seed-level partial-association design.

    Declared structural constants are handled before numerical diagnostics.
    A structural target is unidentifiable by definition. Structural controls
    are removed by rule definition. Ordinary constant/duplicate checks are
    then applied only to the remaining coordinates.
    """
    structural = STRUCTURAL_CONSTANTS.get(variant, {})

    if target in structural:
        return {
            "identifiable": False,
            "estimable": False,
            "vif": np.nan,
            "kappa": np.nan,
            "n_rows": 0,
            "notes": [target + "(structural_target)"],
        }

    active_controls = []
    structural_notes = []
    for c in controls:
        if c in structural:
            structural_notes.append(c + "(structural_control)")
        else:
            active_controls.append(c)

    required = [target] + list(active_controls)
    need_C = "C" in required

    s = sub[sub["valid_C"]] if need_C else sub[sub["valid_base"]]
    if s.empty:
        return {
            "identifiable": False,
            "estimable": False,
            "vif": np.nan,
            "kappa": np.nan,
            "n_rows": 0,
            "notes": structural_notes + ["no_valid_rows"],
        }

    finite = np.ones(len(s), dtype=bool)
    for c in required:
        finite &= np.isfinite(s[c].to_numpy(dtype=float))
    s = s.loc[finite, required]
    n_rows = len(s)

    if n_rows == 0:
        return {
            "identifiable": False,
            "estimable": False,
            "vif": np.nan,
            "kappa": np.nan,
            "n_rows": 0,
            "notes": structural_notes + ["no_finite_rows"],
        }

    notes = list(structural_notes)
    target_vals = s[target].to_numpy(dtype=float)

    # This is now only an ordinary numerical degeneracy check.
    # It is not used to discover protocol-declared structural constants.
    if target_vals.std() <= EPS:
        return {
            "identifiable": False,
            "estimable": False,
            "vif": np.nan,
            "kappa": np.nan,
            "n_rows": n_rows,
            "notes": notes + [target + "(constant_target)"],
        }

    kept_controls = []
    for c in active_controls:
        vals = s[c].to_numpy(dtype=float)

        if vals.std() <= EPS:
            notes.append(c + "(constant_control)")
            continue

        if ranked_duplicate(target_vals, vals):
            return {
                "identifiable": False,
                "estimable": False,
                "vif": np.inf,
                "kappa": np.inf,
                "n_rows": n_rows,
                "notes": notes
                + [target + "==" + c + "(target_control_duplicate)"],
            }

        duplicate_control = None
        for c2 in kept_controls:
            if ranked_duplicate(
                vals, s[c2].to_numpy(dtype=float)
            ):
                duplicate_control = c2
                break

        if duplicate_control is not None:
            notes.append(
                c + "==" + duplicate_control + "(control_duplicate)"
            )
            continue

        kept_controls.append(c)

    use = [target] + kept_controls

    # Algebraic feasibility only; no arbitrary minimum-transition cutoff.
    if n_rows <= len(use) + 1:
        notes.append("insufficient_residual_df")
        return {
            "identifiable": True,
            "estimable": False,
            "vif": np.nan,
            "kappa": np.nan,
            "n_rows": n_rows,
            "notes": notes,
        }

    v, k = vif_and_kappa(s[use].to_numpy(dtype=float))
    estimable = bool(v <= VIF_MAX and k <= KAPPA_MAX)

    return {
        "identifiable": True,
        "estimable": estimable,
        "vif": v,
        "kappa": k,
        "n_rows": n_rows,
        "notes": notes,
    }


def route_required_stats(basis_name, variant):
    """Statistics that must be estimable for route-level adjudication."""
    if basis_name == "lite":
        if variant in OPTIMIZER_CONTROLLED or variant in RULE_DETERMINED:
            return ["rho_A_lite", "rho_nu_lite"]
        return ["rho_A_lite"]

    if basis_name == "full":
        # Gate A is driven by rho_nu and rho_C. rho_cos is a reference
        # statistic and is structurally unavailable for vanilla_ls_nm.
        if variant in OPTIMIZER_CONTROLLED or variant in RULE_DETERMINED:
            return ["rho_nu_full", "rho_C_full"]
        if variant == "pcgrad_nm":
            return ["rho_cos_full", "rho_C_full"]
        if variant == "vanilla":
            return ["rho_C_full"]
        return ["rho_nu_full", "rho_C_full"]

    raise ValueError("Unknown basis: {0}".format(basis_name))


def precheck(df, basis_name, stat_specs, outdir):
    """Compute statistic-specific and route-level estimability."""
    rows = []

    for cell, sub_cell in df.groupby(CELL_KEYS, sort=True):
        ansatz, variant, n, d = cell
        seeds_present = int(sub_cell["seed"].nunique())

        stat_seed_results = dict((name, []) for name in stat_specs)
        note_counts = {}

        for seed, sub_seed in sub_cell.groupby("seed", sort=True):
            for stat_name, spec in stat_specs.items():
                target, controls = spec
                result = assess_statistic(sub_seed, variant, target, controls)
                stat_seed_results[stat_name].append((seed, result))
                for note in result["notes"]:
                    key = stat_name + ":" + note
                    note_counts[key] = note_counts.get(key, 0) + 1

        row = {
            "basis": basis_name,
            "ansatz": ansatz,
            "variant": variant,
            "n_qubits": int(n),
            "depth": int(d),
            "seeds_expected": N_SEEDS_EXPECTED,
            "seeds_present": seeds_present,
        }

        per_stat_estimable_seed_sets = {}
        for stat_name, seed_results in stat_seed_results.items():
            identifiable = [
                r for _, r in seed_results if r["identifiable"]
            ]
            estimable = [
                r for _, r in seed_results if r["estimable"]
            ]
            estimable_seed_set = set(
                seed for seed, r in seed_results if r["estimable"]
            )
            per_stat_estimable_seed_sets[stat_name] = estimable_seed_set

            vifs = [
                r["vif"] for r in identifiable
                if np.isfinite(r["vif"])
            ]
            kappas = [
                r["kappa"] for r in identifiable
                if np.isfinite(r["kappa"])
            ]

            row[stat_name + "_seeds_identifiable"] = len(identifiable)
            row[stat_name + "_seeds_estimable"] = len(estimable)
            row[stat_name + "_fraction_expected"] = (
                len(estimable) / N_SEEDS_EXPECTED
            )
            row[stat_name + "_cell_estimable"] = bool(
                len(estimable) >= MIN_ESTIMABLE_SEEDS
            )
            row[stat_name + "_vif_median"] = (
                float(np.median(vifs)) if vifs else np.nan
            )
            row[stat_name + "_vif_p90"] = (
                float(np.percentile(vifs, 90)) if vifs else np.nan
            )
            row[stat_name + "_kappa_median"] = (
                float(np.median(kappas)) if kappas else np.nan
            )

        # Route-level estimability requires the same seed to support every
        # statistic required for that route in this rule.
        required_stats = route_required_stats(basis_name, variant)
        missing_specs = [s for s in required_stats if s not in stat_specs]
        if missing_specs:
            raise RuntimeError(
                "Missing statistic specs for route: {0}".format(
                    ", ".join(missing_specs)
                )
            )

        if required_stats:
            route_seed_set = set.intersection(
                *(per_stat_estimable_seed_sets[s] for s in required_stats)
            )
        else:
            route_seed_set = set()

        n_route_est = len(route_seed_set)
        row["route_required_stats"] = ";".join(required_stats)
        row["seeds_route_estimable"] = n_route_est
        row["estimable_fraction_expected"] = (
            n_route_est / N_SEEDS_EXPECTED
        )
        row["estimable_fraction_present"] = (
            n_route_est / seeds_present if seeds_present else np.nan
        )
        row["cell_estimable"] = bool(
            n_route_est >= MIN_ESTIMABLE_SEEDS
        )
        row["structural_notes"] = "; ".join(
            "{0} x{1}".format(k, v)
            for k, v in sorted(note_counts.items())
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    path = os.path.join(
        outdir, "estimability_{0}.csv".format(basis_name)
    )
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out


def validity_report(df, outdir):
    rows = []
    for cell, sub in df.groupby(CELL_KEYS, sort=True):
        tr = sub[sub["is_transition"]]
        n_tr = len(tr)
        if n_tr == 0:
            continue

        ansatz, variant, n, d = cell
        valid = tr[tr["valid_base"]]

        rows.append({
            "ansatz": ansatz,
            "variant": variant,
            "n_qubits": int(n),
            "depth": int(d),
            "n_transitions": n_tr,
            "n_valid_base": int(len(valid)),
            "f_zero_update": float((tr["R"] <= EPS).mean()),
            "f_zero_grad": float((tr["G"] <= EPS).mean()),
            "f_undefined_C_given_valid": (
                float((~valid["valid_C"]).mean())
                if len(valid) else np.nan
            ),
            "f_invalid_C_total": float((~tr["valid_C"]).mean()),
            "f_worse_given_valid": (
                float((valid["dE_act"] < 0).mean())
                if len(valid) else np.nan
            ),
            "f_worse_total": float((tr["dE_act"] < 0).mean()),
        })

    out = pd.DataFrame(rows)
    path = os.path.join(outdir, "validity_report.csv")
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out


def route_summary(lite, full, outdir):
    """Compare lite/full estimability without reading any associations."""
    key = ["ansatz", "variant", "n_qubits", "depth"]
    keep = key + [
        "cell_estimable",
        "estimable_fraction_expected",
        "seeds_route_estimable",
        "seeds_present",
        "route_required_stats",
    ]

    m = lite[keep].merge(
        full[keep],
        on=key,
        suffixes=("_lite", "_full"),
    )
    path = os.path.join(outdir, "route_comparison.csv")
    m.to_csv(path, index=False)
    print("Saved:", path)

    lines = [
        "E1 ROUTE SELECTION (protocol {0}, Section 9.1)".format(
            PROTOCOL_VERSION
        ),
        "=" * 60,
        "",
    ]

    for group, label in (
        (OPTIMIZER_CONTROLLED, "optimizer-controlled"),
        (RULE_DETERMINED, "rule-determined"),
        (STRUCTURALLY_CONSTRAINED, "structurally constrained"),
    ):
        lines.append("[{0}]".format(label))
        for rule in group:
            s = m[m["variant"] == rule]
            if s.empty:
                continue
            for ansatz in sorted(s["ansatz"].unique()):
                t = s[s["ansatz"] == ansatz]
                lines.append(
                    "  {0:14s} {1}: lite {2}/{3} cells estimable, "
                    "full {4}/{3} cells estimable".format(
                        rule,
                        ansatz,
                        int(t["cell_estimable_lite"].sum()),
                        len(t),
                        int(t["cell_estimable_full"].sum()),
                    )
                )
        lines.append("")

    oc = m[m["variant"].isin(OPTIMIZER_CONTROLLED)].copy()
    lite_mask = oc["cell_estimable_lite"].astype(bool).to_numpy()
    full_mask = oc["cell_estimable_full"].astype(bool).to_numpy()

    both = int(np.sum(lite_mask & full_mask))
    lite_only = int(np.sum(lite_mask & ~full_mask))
    full_only = int(np.sum(~lite_mask & full_mask))
    neither = int(np.sum(~lite_mask & ~full_mask))
    total = len(oc)

    lines.append(
        "Optimizer-controlled cells: both {0}, lite-only {1}, "
        "full-only {2}, neither {3}, total {4}".format(
            both, lite_only, full_only, neither, total
        )
    )

    cen = m[m["variant"] == "lso_pcgrad"]
    cen_lite = int(cen["cell_estimable_lite"].sum())
    cen_full = int(cen["cell_estimable_full"].sum())
    lines.append(
        "Central rule lso_pcgrad: lite {0}/{1}, full {2}/{1}".format(
            cen_lite, len(cen), cen_full
        )
    )
    lines.append("")

    # Protocol v1.6 Section 9.1 mixed-route rule.
    if total == 0:
        lines.append(
            "-> No optimizer-controlled cells were found. Route cannot "
            "be selected."
        )
    elif both == 0 and lite_only == 0 and full_only == 0:
        lines.append(
            "-> BOTH bases are underdetermined for all optimizer-controlled "
            "cells. This is an estimability failure, not scientific evidence."
        )
    elif full_only > lite_only:
        lines.append(
            "-> FULL E1 SELECTED. In optimizer-controlled cells, full-only "
            "exceeds lite-only ({0} > {1}). This is the frozen v1.6 mixed-"
            "route rule.".format(full_only, lite_only)
        )
    elif lite_only > full_only:
        lines.append(
            "-> E1-LITE FIRST SELECTED. In optimizer-controlled cells, "
            "lite-only exceeds full-only ({0} > {1}). Any positive lite "
            "result still requires full-E1 escalation where estimable.".format(
                lite_only, full_only
            )
        )
    elif full_only == 0 and lite_only == 0 and both > 0:
        lines.append(
            "-> Lite and full routes have identical cellwise estimability. "
            "E1-lite may be used as the first screen, subject to Section 19.2."
        )
    else:
        lines.append(
            "-> ROUTE TIE. full-only equals lite-only ({0} = {1}). "
            "Protocol v1.6 requires a new amendment before any association "
            "statistic is computed.".format(full_only, lite_only)
        )

    if len(cen) and (
        cen_lite < len(cen) or cen_full < len(cen)
    ):
        lines.append("")
        lines.append(
            "-> Section 13.1 caution: lso_pcgrad is not estimable in every "
            "cell. A Gate-A verdict reached through lso_grid and "
            "vanilla_ls_nm leaves lso_pcgrad-centered claims unresolved in "
            "the affected cells."
        )

    lines.append("")
    lines.append(
        "This file reports estimability only. No association statistic "
        "has been computed."
    )

    txt = "\n".join(lines)
    path = os.path.join(outdir, "route_decision.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("Saved:", path)
    print()
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hea", required=True)
    ap.add_argument("--hva", required=True)
    ap.add_argument("--outdir", default="e1_precheck_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.concat(
        [
            load(args.hea, "hea"),
            load(args.hva, "hva"),
        ],
        ignore_index=True,
    )
    df = derive(df)

    print(
        "records: {0} rows, {1} runs, variants {2}".format(
            len(df),
            df.groupby(RUN_KEYS).ngroups,
            sorted(df["variant"].unique()),
        )
    )
    print()

    identity_audit(df, args.outdir)
    structural_constant_audit(df, args.outdir)
    validity_report(df, args.outdir)

    lite = precheck(
        df,
        "lite",
        LITE_STATS,
        args.outdir,
    )
    full = precheck(
        df,
        "full",
        FULL_STATS,
        args.outdir,
    )

    print()
    route_summary(lite, full, args.outdir)


if __name__ == "__main__":
    main()
