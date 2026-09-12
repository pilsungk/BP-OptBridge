#!/usr/bin/env python3
# e1_posthoc_higher_order.py
#
# POST HOC SENSITIVITY ANALYSIS. Implements the frozen protocol
# "Post-hoc higher-order analysis of the residual composition
# coordinate", v1.2.
#
# This is not a re-adjudication. The confirmatory decision rules of the
# E1 protocol are deliberately not applied here: no |rho| >= 0.20
# threshold, no nine-of-twelve reproducibility rule, and no verdict
# categories. Results are described, not graded.
#
# Question. E1 asked whether C retains an incremental association with
# the realized decrease after standard first-order geometry is
# controlled. The realized decrease already contains finite-step and
# higher-order effects, but they sit underneath a dominant first-order
# component. This script subtracts the first-order prediction explicitly
# and asks whether C relates to what remains.
#
# Outcomes (protocol Section 2):
#     r_t     = dE_act - dE_pred            (first-order residual)
#     kappa_t = -2 r_t / (eta^2 R_t^2)      (curvature proxy, NOT a
#                                            measured Hessian quantity)
#
# Estimator. The seed-level partial Spearman, the Fisher-z aggregation,
# the seed bootstrap, and the estimability screen are taken from the
# frozen E1 script so that the procedure is identical. The one change is
# that the outcome column is a parameter rather than being fixed to
# dE_act, and a startup self-test verifies that the parameterized
# version reproduces the frozen one when the outcome is dE_act.
#
# Usage:
#   python e1_posthoc_higher_order.py \
#       --hea results/v3/hea/trajectory_per_seed.csv.gz \
#       --hva results/v3/hva/trajectory_per_seed.csv.gz \
#       --lr 0.05 \
#       --outdir posthoc_higher_order_out
#
# Reduced implementation-test data may use --allow-incomplete. The
# canonical analysis must not use that flag.

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROTOCOL = "post-hoc higher-order C analysis, v1.2 (frozen 2026-09-03)"
ANALYSIS_KIND = "post hoc sensitivity"

try:
    import e1_association_v1_1 as e1
except ImportError:
    sys.exit("e1_association_v1_1.py must be importable from this directory. "
             "The frozen estimator is reused from it rather than reimplemented.")

# ---- constants inherited unchanged from the frozen E1 script --------
EPS = e1.EPS
VIF_MAX = e1.VIF_MAX
KAPPA_MAX = e1.KAPPA_MAX
MIN_ESTIMABLE_SEEDS = e1.MIN_ESTIMABLE_SEEDS
N_SEEDS_EXPECTED = e1.N_SEEDS_EXPECTED
RUN_KEYS = e1.RUN_KEYS
CELL_KEYS = e1.CELL_KEYS
STRUCTURAL_CONSTANTS = e1.STRUCTURAL_CONSTANTS

# ---- protocol constants --------------------------------------------
# Section 5.2: kappa is a ratio with R^2 in the denominator, so
# transitions whose update norm has effectively collapsed are removed.
# The threshold is relative to the seed's own median over E1-valid
# transitions. It is a numerical safeguard, not a tuned parameter.
KAPPA_R_FLOOR_FRACTION = 1e-3

# Section 3: three rules, with their standing.
RULE_ROLE = {
    "vanilla": "primary",
    "vanilla_ls_nm": "near-primary",
    "lso_grid": "secondary",
}

# Section 4: the target is always C. Conditioning depends on the rule.
# For vanilla_ls_nm the two entries answer different questions and are
# both reported; the second is a check for selection-induced
# overcontrol, not a cleaner variant of the first.
C_SPECS = {
    "vanilla": [("C_given_G", ["G"], "main")],
    "vanilla_ls_nm": [("C_given_G_nu", ["G", "nu"], "main"),
                      ("C_given_G", ["G"], "overcontrol check")],
    "lso_grid": [("C_given_G_nu_cos", ["G", "nu", "cos_theta"], "main")],
}

# Section 4: nu may appear only as a conditioning variable.
# No association in this package uses nu as the target.

OUTCOMES = ["r", "kappa"]


# ====================================================================
# derivation
# ====================================================================

def load_and_derive(path, ansatz, lr):
    """E1 derivation, plus the two higher-order outcomes.

    The supplied learning rate is checked against the stored first-order
    prediction before it is used in the kappa denominator.
    """
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("--lr must be a finite positive number")

    df = e1.load(path, ansatz)
    df = e1.derive(df)

    if "pred_decrease_lin" not in df.columns:
        raise ValueError(path + ": pred_decrease_lin column is missing")

    # Fail fast if the CLI learning rate is inconsistent with the stored
    # first-order prediction. The paper convention is
    # dE_pred = eta * g^T u = eta * dir_sum_step.
    m = (df["is_transition"]
         & df["pred_decrease_lin"].notna()
         & df["dir_sum_step"].notna())
    if not bool(m.any()):
        raise ValueError(path + ": no finite transitions for learning-rate audit")
    observed = df.loc[m, "pred_decrease_lin"].to_numpy(dtype=float)
    expected = lr * df.loc[m, "dir_sum_step"].to_numpy(dtype=float)
    if not np.allclose(observed, expected, rtol=1e-10, atol=1e-12):
        abs_diff = np.abs(observed - expected)
        scale = np.maximum(np.abs(expected), 1e-15)
        rel_diff = abs_diff / scale
        raise ValueError(
            path + ": --lr is inconsistent with pred_decrease_lin; "
            + "max_abs_diff={0:.3e}, max_rel_diff={1:.3e}".format(
                float(abs_diff.max()), float(rel_diff.max())))
    print("  learning-rate audit {0}: pass, max |stored-eta*gTu|={1:.3e}".format(
        ansatz, float(np.max(np.abs(observed - expected)))))

    # r_t on the paper's sign convention: both dE_act and
    # pred_decrease_lin are positive when energy improves.
    df["r"] = df["dE_act"] - df["pred_decrease_lin"]

    # kappa_t = -2 r / (eta^2 R^2). From
    #   dE_act = eta g^T u - (eta^2 / 2) u^T H u,
    #   dE_pred = eta g^T u,
    # so r = -(eta^2 / 2) u^T H u and
    # u_hat^T H u_hat = -2 r / (eta^2 R^2).
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (lr ** 2) * (df["R"].to_numpy(dtype=float) ** 2)
        df["kappa"] = np.where(
            denom > 0,
            -2.0 * df["r"].to_numpy(dtype=float) / denom,
            np.nan)

    # Normalized step index, used as the time control of Section 6.
    g = df.groupby(RUN_KEYS, sort=False)
    df["tstep"] = g["step"].transform(
        lambda s: s / max(float(s.max()), 1.0))

    # Section 5.2 exclusion, applied to the kappa analysis only. The
    # seed-relative median is computed over E1-valid C transitions.
    valid = df["valid_C"]
    med = (df[valid].groupby(RUN_KEYS)["R"].median()
           .rename("R_seed_median").reset_index())
    df = df.merge(med, on=RUN_KEYS, how="left")
    df["kappa_usable"] = (
        valid
        & df["R_seed_median"].notna()
        & (df["R"] >= KAPPA_R_FLOOR_FRACTION * df["R_seed_median"]))
    return df


# ====================================================================
# estimator, parameterized by outcome
# ====================================================================

def partial_spearman(sub, variant, target, controls, outcome):
    """Structurally identical to the frozen E1 estimator, with the
    outcome column supplied rather than fixed to dE_act.

    The equivalence is checked at startup by selftest_estimator().
    """
    structural = STRUCTURAL_CONSTANTS.get(variant, {})
    notes = []

    if target in structural:
        return np.nan, {"status": "structural_target"}

    active = []
    for c in controls:
        if c in structural:
            notes.append(c + "(structural_control)")
        else:
            active.append(c)

    need = [target] + active + [outcome]
    s = sub.dropna(subset=need)
    if len(s) - len(active) - 2 <= 0:
        return np.nan, {"status": "insufficient_rows"}

    y = rankdata(s[outcome].to_numpy(dtype=float))
    x = s[target].to_numpy(dtype=float)
    if np.std(x) <= EPS:
        return np.nan, {"status": "constant_target"}
    xr = rankdata(x)

    Z, kept = [], []
    for c in active:
        v = s[c].to_numpy(dtype=float)
        if np.std(v) <= EPS:
            notes.append(c + "(constant_control)")
            continue
        vr = rankdata(v)
        if abs(np.corrcoef(vr, xr)[0, 1]) > 1.0 - 1e-12:
            return np.nan, {"status": "target_control_duplicate"}
        dup = False
        for w in Z:
            if abs(np.corrcoef(vr, w)[0, 1]) > 1.0 - 1e-12:
                dup = True
                break
        if not dup:
            Z.append(vr)
            kept.append(c)

    if Z:
        D = np.column_stack([(z - z.mean()) / z.std() for z in Z])
        # the design for the collinearity screen includes the target,
        # exactly as in the frozen precheck and association scripts
        tgt = (xr - xr.mean()) / xr.std()
        full = np.column_stack([tgt, D])
        vifs = []
        for j in range(full.shape[1]):
            yj = full[:, j]
            Zj = np.column_stack([np.ones(len(full)),
                                  np.delete(full, j, axis=1)])
            b, *_ = np.linalg.lstsq(Zj, yj, rcond=None)
            resid = yj - Zj @ b
            sst = float(((yj - yj.mean()) ** 2).sum())
            r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > EPS else 1.0
            vifs.append(1.0 / max(1.0 - r2, 1e-12))
        sv = np.linalg.svd(full, compute_uv=False)
        kappa_cond = float(sv[0] / sv[-1]) if sv[-1] > EPS else np.inf
        if max(vifs) > VIF_MAX or kappa_cond > KAPPA_MAX:
            return np.nan, {"status": "collinear",
                            "max_vif": float(max(vifs)),
                            "kappa": kappa_cond}
        M = np.column_stack([np.ones(len(s))] + Z)
        bx, *_ = np.linalg.lstsq(M, xr, rcond=None)
        by, *_ = np.linalg.lstsq(M, y, rcond=None)
        rx, ry = xr - M @ bx, y - M @ by
        max_vif, kap = float(max(vifs)), kappa_cond
    else:
        rx, ry = xr - xr.mean(), y - y.mean()
        max_vif, kap = np.nan, np.nan

    if np.std(rx) <= EPS or np.std(ry) <= EPS:
        return np.nan, {"status": "degenerate_residual"}

    return float(np.corrcoef(rx, ry)[0, 1]), {
        "status": "ok", "max_vif": max_vif, "kappa": kap,
        "n_rows": len(s), "controls": kept, "notes": notes}


def selftest_estimator(df):
    """Verify that the parameterized estimator reproduces the frozen one
    when the outcome is dE_act. A silent divergence here would make the
    whole package incomparable with E1."""
    checked = 0
    for (ansatz, variant, n, d), cell in df.groupby(CELL_KEYS, sort=True):
        if variant != "lso_grid":
            continue
        for seed, sub in cell.groupby("seed", sort=True):
            s = sub[sub["valid_C"]]
            a, _ = e1.partial_spearman(s, variant, "C",
                                       ["G", "nu", "cos_theta"])
            b, _ = partial_spearman(s, variant, "C",
                                    ["G", "nu", "cos_theta"], "dE_act")
            if np.isfinite(a) != np.isfinite(b):
                raise AssertionError("estimator selftest: definedness differs")
            if np.isfinite(a) and abs(a - b) > 1e-12:
                raise AssertionError(
                    "estimator selftest: {0} vs {1}".format(a, b))
            checked += 1
            if checked >= 40:
                print("  estimator selftest: {0} seed-level comparisons, "
                      "no divergence".format(checked))
                return
    print("  estimator selftest: {0} comparisons, no divergence".format(checked))


# ====================================================================
# analysis pass
# ====================================================================

def e1_bootstrap_seed(ansatz, variant, n, d, outcome, spec, time_ctrl):
    """Use the frozen E1 deterministic bootstrap-seed generator itself."""
    stat = "posthoc_{0}_{1}_{2}".format(
        outcome, spec, "time" if time_ctrl else "notime")
    return e1.stable_bootstrap_seed(ansatz, variant, n, d, stat)


def audit_canonical_coverage(df, allow_incomplete=False):
    """Fail fast if canonical inputs do not match the frozen analysis grid.

    The canonical run requires 12 (n, d) cells per analyzed rule and
    ansatz, with 30 seeds in every cell. The override exists only for
    synthetic or reduced-data implementation tests.
    """
    expected_cells = int(getattr(e1, "N_CELLS_PER_RULE", 12))
    issues = []
    lines = []
    for ansatz in ("hea", "hva"):
        for variant in RULE_ROLE:
            s = df[(df["ansatz"] == ansatz) & (df["variant"] == variant)]
            cells = s[["n_qubits", "depth"]].drop_duplicates()
            n_cells = int(len(cells))
            seed_counts = s.groupby(["n_qubits", "depth"])["seed"].nunique()
            bad = seed_counts[seed_counts != N_SEEDS_EXPECTED]
            lines.append("  {0:3s} {1:14s}: cells {2}/{3}, bad-seed-cells {4}".format(
                ansatz, variant, n_cells, expected_cells, int(len(bad))))
            if n_cells != expected_cells:
                issues.append("{0}/{1}: {2} cells, expected {3}".format(
                    ansatz, variant, n_cells, expected_cells))
            for (n, d), count in bad.items():
                issues.append("{0}/{1}/n={2}/d={3}: {4} seeds, expected {5}".format(
                    ansatz, variant, int(n), int(d), int(count),
                    N_SEEDS_EXPECTED))

    print("canonical coverage audit:")
    for line in lines:
        print(line)
    if issues:
        if allow_incomplete:
            print("  WARNING: incomplete coverage accepted only because "
                  "--allow-incomplete was supplied")
        else:
            raise ValueError(
                "canonical coverage audit failed:\n  " + "\n  ".join(issues))
    else:
        print("  coverage audit: pass")


def run(df, outdir):
    rows = []
    for (ansatz, variant, n, d), cell in df.groupby(CELL_KEYS, sort=True):
        if variant not in RULE_ROLE:
            continue
        for outcome in OUTCOMES:
            mask_col = "kappa_usable" if outcome == "kappa" else "valid_C"
            specs = C_SPECS[variant]
            for time_ctrl in (False, True):
                for name, controls, kind in specs:
                    target = "C"
                    ctrl = list(controls) + (["tstep"] if time_ctrl else [])
                    rhos, statuses, excl = [], [], []
                    for seed, sub in cell.groupby("seed", sort=True):
                        base = sub[sub["valid_C"]]
                        used = sub[sub[mask_col]]
                        if len(base):
                            excl.append(1.0 - len(used) / len(base))
                        rho, diag = partial_spearman(used, variant, target,
                                                     ctrl, outcome)
                        rhos.append(rho)
                        statuses.append(diag["status"])
                    n_est = int(np.sum(np.isfinite(rhos)))
                    bseed = e1_bootstrap_seed(
                        ansatz, variant, n, d, outcome, name, time_ctrl)
                    agg = e1.aggregate_seeds(rhos, bseed)
                    rows.append({
                        "ansatz": ansatz, "variant": variant,
                        "role": RULE_ROLE[variant],
                        "n_qubits": int(n), "depth": int(d),
                        "outcome": outcome, "target": target,
                        "spec": name, "spec_kind": kind,
                        "time_controlled": time_ctrl,
                        "seeds_expected": N_SEEDS_EXPECTED,
                        "seeds_estimable": n_est,
                        "cell_estimable": bool(n_est >= MIN_ESTIMABLE_SEEDS),
                        "rho_bar": agg["rho_bar"],
                        "ci_lo": agg["ci_lo"], "ci_hi": agg["ci_hi"],
                        "bootstrap_seed": int(bseed),
                        "excluded_fraction": (float(np.mean(excl))
                                              if excl else np.nan),
                        "top_status": pd.Series(statuses).mode().iat[0],
                    })
    out = pd.DataFrame(rows)
    path = os.path.join(outdir, "posthoc_higher_order_cells.csv")
    out.to_csv(path, index=False)
    print("Saved:", path)
    return out


def describe(cells, outdir):
    """Section 8: describe, do not adjudicate. No thresholds, no verdict
    categories. Sign, magnitude, heterogeneity, and intervals only."""
    L = ["POST-HOC HIGHER-ORDER ANALYSIS OF C", "=" * 68, ""]
    L.append("analysis kind : {0}".format(ANALYSIS_KIND))
    L.append("protocol      : {0}".format(PROTOCOL))
    L.append("")
    L.append("This analysis does not re-adjudicate the E1 verdict and defines")
    L.append("no verdict categories of its own. The tables below report the")
    L.append("cell-level estimates, their spread, and their intervals. The")
    L.append("time-controlled rows are primary for interpretation.")
    L.append("")

    for outcome in OUTCOMES:
        label = ("first-order residual r" if outcome == "r"
                 else "curvature proxy kappa")
        L.append("-" * 68)
        L.append("OUTCOME: {0}".format(label))
        L.append("-" * 68)
        for tc in (True, False):
            L.append("")
            L.append("  time control: {0}{1}".format(
                "YES" if tc else "no",
                "   [primary for interpretation]" if tc else
                "    [continuity with E1]"))
            sub = cells[(cells.outcome == outcome)
                        & (cells.time_controlled == tc)]
            L.append("")
            L.append("  {0:14s} {1:5s} {2:20s} {3:>5s} {4:>8s} {5:>8s} "
                     "{6:>8s}".format(
                         "rule", "ansat", "spec", "est", "median",
                         "min", "max"))
            for (v, a, sp), g in sub.groupby(["variant", "ansatz", "spec"],
                                             sort=True):
                e = g[g.cell_estimable]
                if len(e) == 0:
                    L.append("  {0:14s} {1:5s} {2:20s} {3:>5s}  "
                             "not estimable in any cell".format(
                                 v, a, sp, "0/12"))
                    continue
                L.append("  {0:14s} {1:5s} {2:20s} {3:>5s} {4:>+8.3f} "
                         "{5:>+8.3f} {6:>+8.3f}".format(
                             v, a, sp, "{0}/12".format(int(len(e))),
                             float(e.rho_bar.median()),
                             float(e.rho_bar.min()),
                             float(e.rho_bar.max())))
        L.append("")

    L.append("-" * 68)
    L.append("EXCLUDED FRACTION under protocol Section 5.2 (kappa only)")
    L.append("-" * 68)
    k = cells[cells.outcome == "kappa"]
    for (v, a), g in k.groupby(["variant", "ansatz"], sort=True):
        L.append("  {0:14s} {1:5s} mean {2:.4f}  max {3:.4f}".format(
            v, a, float(g.excluded_fraction.mean()),
            float(g.excluded_fraction.max())))

    L.append("")
    L.append("-" * 68)
    L.append("STOP RULE (protocol Section 8)")
    L.append("-" * 68)
    L.append("This is the last analysis run for the manuscript, whatever it")
    L.append("shows. The E1 adjudication is not reopened. No further")
    L.append("analysis, experiment, or regeneration follows. Any recurrent")
    L.append("pattern is a post-hoc observation and a direction for future")
    L.append("work, never a mechanism.")

    txt = "\n".join(L)
    path = os.path.join(outdir, "posthoc_higher_order_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("Saved:", path)
    print()
    print(txt)


def write_checksums(items, outdir):
    """Write the provenance hashes required by protocol Section 10."""
    lines = []
    for label, p in items:
        p = os.path.abspath(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append("{0}  {1}  {2}".format(h.hexdigest(), label, p))
    path = os.path.join(outdir, "input_checksums.sha256")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved:", path)
    for line in lines:
        print("   ", line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hea", required=True)
    ap.add_argument("--hva", required=True)
    ap.add_argument("--lr", type=float, required=True,
                    help="learning rate used in the runs (config.json: lr)")
    ap.add_argument("--outdir", default="posthoc_higher_order_out")
    ap.add_argument(
        "--allow-incomplete", action="store_true",
        help="allow reduced test data; never use for the canonical analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("protocol:", PROTOCOL)
    print("learning rate:", args.lr)
    print()
    write_checksums([
        ("analysis_script", __file__),
        ("frozen_e1_estimator", e1.__file__),
        ("hea_trajectory", args.hea),
        ("hva_trajectory", args.hva),
    ], args.outdir)
    print()

    df = pd.concat([load_and_derive(args.hea, "hea", args.lr),
                    load_and_derive(args.hva, "hva", args.lr)],
                   ignore_index=True)
    print("records: {0} rows, {1} runs".format(
        len(df), df.groupby(RUN_KEYS).ngroups))
    audit_canonical_coverage(df, allow_incomplete=args.allow_incomplete)
    selftest_estimator(df)
    print()

    cells = run(df, args.outdir)
    print()
    describe(cells, args.outdir)


if __name__ == "__main__":
    main()
