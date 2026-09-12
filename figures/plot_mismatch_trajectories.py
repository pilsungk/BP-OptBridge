#!/usr/bin/env python3
# plot_mismatch_trajectories.py
#
# Figure for Section 5 (diagnostic-optimization mismatch):
# 2x2 panel of seed-averaged optimization trajectories.
#   Rows: ansatz families, marked once per row by rotated margin labels
#   Cols: (a,c) energy vs step, (b,d) parameter-averaged B_eff vs step
# Curves: the three primary update rules (vanilla, pcgrad, lso_pcgrad).
#
# Usable in two ways:
#   1) As a module inside a Jupyter notebook:
#        from plot_mismatch_trajectories import make_mismatch_figure
#        fig = make_mismatch_figure(
#            hea_csv="../results/v3/hea/trajectory_per_seed.csv.gz",
#            hva_csv="../results/v3/hva/trajectory_per_seed.csv.gz",
#            n_qubits=10, depth=8,
#            out="figures/fig_mismatch_n10_d8.pdf")
#      The figure object is returned and renders inline; audit failures
#      raise AuditError instead of terminating the kernel.
#   2) As a command-line script:
#        python plot_mismatch_trajectories.py \
#            --hea_csv ... --hva_csv ... --n_qubits 10 --depth 8 \
#            --out figures/fig_mismatch_n10_d8.pdf
#
# The code only reads existing trajectory records from the v3 suite;
# it runs no simulation and modifies no data.
#
# Data audits (all failures raise AuditError, none silently repaired):
#   A1. (variant, seed, step) rows must be unique.
#   A2. All variants must share one step grid and one seed set, with
#       every seed present at every step.
#   A3. At the initial step, all variants of the same seed must record
#       identical energy and B_eff up to step0_tol, since all rules
#       start from identical parameters.
#
# Cross-checking: the printed final-step means are per-condition values
# for the selected (n, d). Compare them against the corresponding rows
# of the v3 final_compare.csv, NOT against the condition-balanced
# aggregates of Table 3 in the manuscript.

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRIMARY_VARIANTS = ["vanilla", "pcgrad", "lso_pcgrad"]

VARIANT_LABELS = {
    "vanilla": "Vanilla",
    "pcgrad": "PCGrad",
    "lso_pcgrad": "LSO-PCGrad",
}

# Colorblind-safe palette (Okabe-Ito)
VARIANT_COLORS = {
    "vanilla": "#444444",
    "pcgrad": "#D55E00",
    "lso_pcgrad": "#0072B2",
}

BEFF_CANDIDATES = ["B_eff_mean", "B_eff", "beff_mean", "B_eff_param_mean"]

BASE_REQUIRED_COLS = ["n_qubits", "depth", "variant", "seed", "step", "energy"]


class AuditError(RuntimeError):
    # Raised when an input-data audit fails; safe inside notebooks
    pass


def detect_beff_col(df, requested, path):
    if requested:
        if requested in df.columns:
            return requested
        raise AuditError(
            "{0}: requested B_eff column '{1}' not found. "
            "Available columns: {2}".format(
                path, requested, ", ".join(df.columns)))
    for cand in BEFF_CANDIDATES:
        if cand in df.columns:
            return cand
    raise AuditError(
        "{0}: no B_eff column found among candidates {1}. "
        "Available columns: {2}".format(
            path, BEFF_CANDIDATES, ", ".join(df.columns)))


def audit_uniqueness(df, path):
    # A1: (variant, seed, step) must identify exactly one row
    dup_mask = df.duplicated(subset=["variant", "seed", "step"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        example = df.loc[dup_mask, ["variant", "seed", "step"]].head(5)
        raise AuditError(
            "{0}: {1} duplicated (variant, seed, step) rows found. "
            "First cases:\n{2}\nFix the upstream data before "
            "plotting.".format(path, n_dup, example.to_string(index=False)))
    print("[AUDIT] A1 uniqueness: pass ({0} rows)".format(len(df)))


def audit_step_coverage(df, path, variants):
    # A2: one common step grid and one common seed set across variants,
    # with every seed present at every step
    step_sets = {}
    seed_sets = {}
    for v in variants:
        sub = df[df["variant"] == v]
        if sub.empty:
            raise AuditError(
                "{0}: variant '{1}' has no rows for the selected "
                "condition.".format(path, v))
        step_sets[v] = set(int(s) for s in sub["step"].unique())
        seed_sets[v] = set(int(s) for s in sub["seed"].unique())

    ref_v = variants[0]
    for v in variants[1:]:
        if step_sets[v] != step_sets[ref_v]:
            raise AuditError(
                "{0}: step grid of '{1}' differs from '{2}' "
                "(symmetric difference size {3}).".format(
                    path, v, ref_v,
                    len(step_sets[v] ^ step_sets[ref_v])))
        if seed_sets[v] != seed_sets[ref_v]:
            raise AuditError(
                "{0}: seed set of '{1}' differs from '{2}'.".format(
                    path, v, ref_v))

    n_seeds = len(seed_sets[ref_v])
    for v in variants:
        counts = df[df["variant"] == v].groupby("step")["seed"].nunique()
        bad = counts[counts != n_seeds]
        if len(bad) > 0:
            raise AuditError(
                "{0}: variant '{1}' has steps with incomplete seed "
                "coverage (expected {2} seeds). First cases:\n{3}".format(
                    path, v, n_seeds, bad.head(5).to_string()))

    steps = sorted(step_sets[ref_v])
    print("[AUDIT] A2 coverage: pass ({0} seeds x steps {1}-{2}, "
          "all variants aligned)".format(n_seeds, steps[0], steps[-1]))
    if n_seeds != 30:
        print("[WARN] expected 30 seeds, found {0}".format(n_seeds))
    return steps[0]


def audit_step0_identity(df, beff_col, path, variants, step0, tol):
    # A3: identical initial parameters imply identical recorded energy
    # and B_eff at the initial step for every seed, across variants
    sub = df[df["step"] == step0]
    max_diffs = {}
    for col in ["energy", beff_col]:
        pivot = sub.pivot_table(index="seed", columns="variant",
                                values=col, aggfunc="first")
        pivot = pivot[variants]
        spread = (pivot.max(axis=1) - pivot.min(axis=1)).abs()
        max_diffs[col] = float(spread.max())
    msg = ", ".join("max |d {0}| = {1:.3e}".format(k, v)
                    for k, v in max_diffs.items())
    if any(v > tol for v in max_diffs.values()):
        raise AuditError(
            "{0}: step-{1} identity audit failed ({2}, tol {3:.1e}). "
            "Rules do not share initial states; inspect the suite "
            "before plotting.".format(path, step0, msg, tol))
    print("[AUDIT] A3 step-{0} identity: pass ({1}, tol {2:.1e})".format(
        step0, msg, tol))


def load_family(path, family, n, d, variants, beff_col_arg, step0_tol):
    if not os.path.isfile(path):
        raise AuditError("file not found: {0}".format(path))
    df = pd.read_csv(path)

    missing = [c for c in BASE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise AuditError(
            "{0}: missing required columns: {1}. Available: {2}".format(
                path, ", ".join(missing), ", ".join(df.columns)))

    beff_col = detect_beff_col(df, beff_col_arg, path)

    # Filter by ansatz column if present; otherwise assume family-specific file
    if "ansatz" in df.columns:
        before = len(df)
        df = df[df["ansatz"].astype(str).str.lower() == family]
        print("[INFO] {0}: filtered by ansatz == '{1}' "
              "({2} -> {3} rows)".format(path, family, before, len(df)))

    df = df[(df["n_qubits"] == n) & (df["depth"] == d)]
    df = df[df["variant"].isin(variants)]

    if df.empty:
        raise AuditError(
            "{0}: no rows for n={1}, d={2}, variants={3}".format(
                path, n, d, variants))

    print("[INFO] source: {0} (family={1}, n={2}, d={3}, "
          "B_eff column='{4}')".format(path, family, n, d, beff_col))

    audit_uniqueness(df, path)
    step0 = audit_step_coverage(df, path, variants)
    audit_step0_identity(df, beff_col, path, variants, step0, step0_tol)

    # Per-condition provenance report; compare against the (n, d) rows
    # of the v3 final_compare.csv, not the Table 3 aggregates
    final_step = int(df["step"].max())
    for v in variants:
        final = df[(df["variant"] == v) & (df["step"] == final_step)]
        print("[INFO]   variant '{0}': final-step {1} E mean={2:.6f}, "
              "{3} mean={4:.6f}".format(
                  v, final_step, final["energy"].mean(),
                  beff_col, final[beff_col].mean()))
    print("[INFO]   cross-check the values above against the "
          "(n={0}, d={1}) rows of final_compare.csv "
          "(per-condition), not Table 3 (condition-balanced).".format(n, d))

    return df, beff_col


def agg_curves(df, value_col, band):
    # Per-step center line and band edges across seeds
    g = df.groupby("step")[value_col]
    steps = np.array(sorted(df["step"].unique()))
    if band == "iqr":
        center = g.median()
        lo = g.quantile(0.25)
        hi = g.quantile(0.75)
    else:
        center = g.mean()
        sd = g.std(ddof=1)
        lo = center - sd
        hi = center + sd
    center = center.reindex(steps).to_numpy()
    lo = lo.reindex(steps).to_numpy()
    hi = hi.reindex(steps).to_numpy()
    return steps, center, lo, hi


def plot_panel(ax, df, value_col, variants, band):
    for v in variants:
        sub = df[df["variant"] == v]
        if sub.empty:
            continue
        steps, center, lo, hi = agg_curves(sub, value_col, band)
        color = VARIANT_COLORS.get(v, None)
        label = VARIANT_LABELS.get(v, v)
        ax.plot(steps, center, color=color, linewidth=1.6, label=label)
        if band != "none":
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15,
                            linewidth=0)
    ax.grid(True, alpha=0.3)


def make_mismatch_figure(hea_csv, hva_csv, n_qubits=10, depth=8,
                         variants=None, beff_col="", band="sd",
                         step0_tol=1e-9, out=None):
    # Build the 2x2 mismatch figure and return the matplotlib figure.
    # Raises AuditError if any input-data audit fails.
    if variants is None:
        variants = list(PRIMARY_VARIANTS)
    if band not in ("sd", "iqr", "none"):
        raise ValueError("band must be one of 'sd', 'iqr', 'none'")

    hea_df, hea_beff = load_family(hea_csv, "hea", n_qubits, depth,
                                   variants, beff_col, step0_tol)
    hva_df, hva_beff = load_family(hva_csv, "hva", n_qubits, depth,
                                   variants, beff_col, step0_tol)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharex=True)

    panel_specs = [
        (axes[0, 0], hea_df, "energy", "Energy"),
        (axes[0, 1], hea_df, hea_beff,
         r"$B_{\mathrm{eff}}$ (parameter average)"),
        (axes[1, 0], hva_df, "energy", "Energy"),
        (axes[1, 1], hva_df, hva_beff,
         r"$B_{\mathrm{eff}}$ (parameter average)"),
    ]

    for ax, df, col, ylabel in panel_specs:
        plot_panel(ax, df, col, variants, band)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10)

    axes[1, 0].set_xlabel("Optimization step", fontsize=12)
    axes[1, 1].set_xlabel("Optimization step", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))

    # Reserve a left margin for the rotated row labels, then place one
    # label per row at the vertical center of that row's axes
    fig.tight_layout(rect=[0.045, 0.0, 1.0, 0.955])
    for row, row_label in [(0, "HEA"), (1, "HVA")]:
        pos = axes[row, 0].get_position()
        yc = 0.5 * (pos.y0 + pos.y1)
        fig.text(0.025, yc, row_label, rotation=90,
                 va="center", ha="center",
                 fontsize=14, fontweight="bold")

    if out:
        outdir = os.path.dirname(out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("[INFO] saved: {0}".format(out))

    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot seed-averaged energy and B_eff trajectories "
                    "for the primary update rules."
    )
    p.add_argument("--hea_csv", type=str, required=True,
                   help="Path to the HEA trajectory CSV (v3 suite)")
    p.add_argument("--hva_csv", type=str, required=True,
                   help="Path to the HVA trajectory CSV (v3 suite)")
    p.add_argument("--n_qubits", type=int, default=10)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--variants", type=str,
                   default=",".join(PRIMARY_VARIANTS),
                   help="Comma-separated variant names to plot")
    p.add_argument("--beff_col", type=str, default="",
                   help="B_eff column name (auto-detected if empty)")
    p.add_argument("--band", type=str, default="sd",
                   choices=["sd", "iqr", "none"],
                   help="Across-seed band: mean+-1SD, median IQR, or none")
    p.add_argument("--step0_tol", type=float, default=1e-9,
                   help="Tolerance for the step-0 identity audit (A3)")
    p.add_argument("--out", type=str,
                   default="fig_mismatch_trajectories.pdf")
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    args = parse_args()
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    try:
        fig = make_mismatch_figure(
            hea_csv=args.hea_csv,
            hva_csv=args.hva_csv,
            n_qubits=args.n_qubits,
            depth=args.depth,
            variants=variants,
            beff_col=args.beff_col,
            band=args.band,
            step0_tol=args.step0_tol,
            out=args.out,
        )
        plt.close(fig)
    except AuditError as exc:
        print("[ABORT] {0}".format(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
