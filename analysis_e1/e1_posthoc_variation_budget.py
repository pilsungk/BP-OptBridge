#!/usr/bin/env python3
# e1_posthoc_variation_budget.py
#
# POST HOC ANALYSIS. This script was written after the frozen E1
# adjudication had been executed and its verdict inspected. It is not
# part of the prespecified protocol, it does not enter the Gate-A
# decision, and it cannot change the A3 verdict recorded in
# gate_a_verdict.txt.
#
# Question. The E1 association analysis found the incremental
# association of the optimizer-relative displacement, rho_nu, to be
# positive wherever material but not reproducible across the eligible
# rules and both ansatzes. This script asks a descriptive follow-up
# question: does the cell-to-cell variation in rho_nu track how much of
# the step-norm variation is carried by nu in that cell?
#
# Inputs are the canonical E1 outputs, which are read but never
# modified:
#   cell_associations.csv   (rho_nu per cell, with the estimability flag)
#   log_variance_audit.csv  (variance shares per cell)
#
# Estimability rule. cell_associations.csv reports a raw association
# estimate for every cell, including cells that failed the estimability
# criterion of the protocol. Those cells are excluded here, exactly as
# they are excluded from the adjudication. Mixing them back in would
# make the estimability rule meaningless.
#
# Usage:
#   python e1_posthoc_variation_budget.py \
#       --assoc  e1_association_out_v1_1/cell_associations.csv \
#       --audit  e1_association_out_v1_1/log_variance_audit.csv \
#       --outdir e1_posthoc_out

import argparse
import os

import numpy as np
import pandas as pd

ANALYSIS_KIND = "post hoc"
PROTOCOL_REFERENCE = "v1.8 (frozen; this analysis is outside it)"

# Section 0.2 of the protocol: lso_pcgrad is ineligible on coverage
# grounds, so the eligible optimizer-controlled pool is these two rules.
ELIGIBLE_RULES = ["lso_grid", "vanilla_ls_nm"]
KEY = ["ansatz", "variant", "n_qubits", "depth"]


def load(assoc_path, audit_path):
    a = pd.read_csv(assoc_path)
    v = pd.read_csv(audit_path)

    need_a = KEY + ["statistic", "cell_estimable", "rho_bar"]
    need_v = KEY + ["s_nu_within_R_median", "s_G_within_R_median"]
    for df, need, path in ((a, need_a, assoc_path), (v, need_v, audit_path)):
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError("{0}: missing columns {1}".format(
                path, ", ".join(missing)))

    a = a[(a["statistic"] == "rho_nu") & (a["variant"].isin(ELIGIBLE_RULES))]
    m = a.merge(v, on=KEY, how="inner", validate="one_to_one")

    # Denominator-adjusted ratio of the two reported median shares. At
    # the level of a single seed this equals Var(log nu) / Var(log G),
    # but the reported shares are per-cell medians over seeds, so the
    # ratio of medians is not itself a variance ratio. It is used only
    # as a check that the pattern does not depend on carrying the shared
    # denominator Var(log R).
    with np.errstate(divide="ignore", invalid="ignore"):
        m["nu_over_G"] = np.where(
            m["s_G_within_R_median"].abs() > 0,
            m["s_nu_within_R_median"] / m["s_G_within_R_median"],
            np.nan)
    return m


def pearson(x, y):
    """Pearson correlation that refuses to drop values silently.

    A quiet NaN drop would let the script report one cell count while
    correlating a different number of cells, which is exactly the kind
    of mismatch this diagnostic exists to avoid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("non-finite values in the correlation inputs")
    if len(x) < 3:
        raise ValueError("fewer than three cells in the correlation inputs")
    return float(np.corrcoef(x, y)[0, 1])


# Expected composition of the analysis set for the canonical E1 outputs.
# These are provenance guards, not analysis parameters.
EXPECTED_MERGED = 48
EXPECTED_ESTIMABLE = 46
EXPECTED_EXCLUDED = [("hea", "lso_grid", 10, 8),
                     ("hva", "vanilla_ls_nm", 4, 8)]

FINITE_COLUMNS = ["rho_bar", "s_nu_within_R_median",
                  "s_G_within_R_median", "nu_over_G"]


def check_inputs(merged, kept, dropped, strict=True):
    """Verify that the inputs are the canonical E1 outputs.

    A silently different CSV would produce a plausible but wrong number,
    so the expected cell composition is checked explicitly.
    """
    problems = []
    if len(merged) != EXPECTED_MERGED:
        problems.append("merged cells {0}, expected {1}".format(
            len(merged), EXPECTED_MERGED))
    if len(kept) != EXPECTED_ESTIMABLE:
        problems.append("estimable cells {0}, expected {1}".format(
            len(kept), EXPECTED_ESTIMABLE))

    got = sorted((r["ansatz"], r["variant"], int(r["n_qubits"]),
                  int(r["depth"])) for _, r in dropped.iterrows())
    if got != sorted(EXPECTED_EXCLUDED):
        problems.append("excluded cells {0}, expected {1}".format(
            got, sorted(EXPECTED_EXCLUDED)))

    for c in FINITE_COLUMNS:
        bad = int((~np.isfinite(kept[c].to_numpy(dtype=float))).sum())
        if bad:
            problems.append("{0} has {1} non-finite values among the "
                            "estimable cells".format(c, bad))

    if problems:
        msg = "input check failed: " + "; ".join(problems)
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assoc", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--outdir", default="e1_posthoc_out")
    ap.add_argument("--no-guard", action="store_true",
                    help="downgrade the input composition check to a warning")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    m = load(args.assoc, args.audit)
    kept = m[m["cell_estimable"]].copy()
    dropped = m[~m["cell_estimable"]].copy()

    check_inputs(m, kept, dropped, strict=not args.no_guard)

    kept_out = kept[KEY + ["rho_bar", "s_nu_within_R_median",
                           "s_G_within_R_median", "nu_over_G"]]
    path = os.path.join(args.outdir, "posthoc_cells_used.csv")
    kept_out.to_csv(path, index=False)
    print("Saved:", path)

    rows = []
    for label, sub in ([("pooled", kept)]
                       + [(a, g) for a, g in kept.groupby("ansatz")]):
        rows.append({
            "scope": label,
            "n_cells": int(len(sub)),
            "corr_rho_nu_vs_s_nu": pearson(sub["rho_bar"],
                                           sub["s_nu_within_R_median"]),
            "corr_rho_nu_vs_nu_over_G": pearson(sub["rho_bar"],
                                                sub["nu_over_G"]),
        })
    summary = pd.DataFrame(rows)
    path = os.path.join(args.outdir, "posthoc_variation_budget.csv")
    summary.to_csv(path, index=False)
    print("Saved:", path)

    L = ["POST HOC VARIATION-BUDGET DIAGNOSTIC", "=" * 64, ""]
    L.append("analysis kind      : {0}".format(ANALYSIS_KIND))
    L.append("protocol reference : {0}".format(PROTOCOL_REFERENCE))
    L.append("")
    L.append("This diagnostic was computed after the frozen E1 verdict was")
    L.append("inspected. It is descriptive, it does not re-adjudicate Gate A,")
    L.append("and it does not license a causal reading.")
    L.append("")
    L.append("Eligible rules     : {0}".format(", ".join(ELIGIBLE_RULES)))
    L.append("Cells merged       : {0}".format(len(m)))
    L.append("Cells used         : {0} (cell_estimable = True)".format(len(kept)))
    L.append("Cells excluded     : {0}".format(len(dropped)))
    for _, r in dropped.iterrows():
        L.append("    excluded: {0} {1} n={2} d={3} (raw rho_bar {4:+.4f} "
                 "not used)".format(r["ansatz"], r["variant"],
                                    int(r["n_qubits"]), int(r["depth"]),
                                    float(r["rho_bar"])))
    L.append("")
    L.append("Correlation of the cell-level rho_nu with the share of")
    L.append("step-norm log-variance carried by nu:")
    L.append("")
    L.append("  {0:8s} {1:>7s} {2:>16s} {3:>20s}".format(
        "scope", "cells", "vs s_nu", "vs s_nu / s_G"))
    for _, r in summary.iterrows():
        L.append("  {0:8s} {1:7d} {2:+16.4f} {3:+20.4f}".format(
            r["scope"], int(r["n_cells"]),
            r["corr_rho_nu_vs_s_nu"], r["corr_rho_nu_vs_nu_over_G"]))
    L.append("")
    L.append("The second column uses s_nu, whose denominator is the")
    L.append("step-norm variance Var(log R). The third divides the two")
    L.append("reported median shares, which removes that shared")
    L.append("denominator without itself being a variance ratio. Close")
    L.append("agreement between the columns indicates that the pattern")
    L.append("does not depend on carrying Var(log R) in the denominator.")
    L.append("")
    L.append("Interpretation is limited to the following statement. The")
    L.append("cell-level association rho_nu tended to be larger in cells")
    L.append("where the optimizer-relative displacement accounted for a")
    L.append("larger share of the step-norm log-variance. This is a")
    L.append("correlation between two cell-level summaries. No claim is")
    L.append("made about why the variation budget differs across cells,")
    L.append("and no mechanism is identified.")

    txt = "\n".join(L)
    path = os.path.join(args.outdir, "posthoc_variation_budget.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("Saved:", path)
    print()
    print(txt)


if __name__ == "__main__":
    main()
