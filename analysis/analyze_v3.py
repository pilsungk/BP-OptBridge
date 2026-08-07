#!/usr/bin/env python3
# analyze_v3.py
#
# Adjudication pipeline for the v3 six-variant experiment.
#
# Input : trajectory_per_seed.csv produced by bp_lso_pcgrad_tfim_v3.py
#         (HEA) or bp_lso_pcgrad_tfim_hva_v3.py (HVA), schema version 4.
# Output: schema_audit.txt, q1_summary.csv, q1_correlations.csv,
#         q1_verdict_by_n.csv, q1_verdict.txt, q2_paired.csv,
#         q2_verdict.txt, coupling_by_n.csv, quadrant_by_n.csv,
#         paper_table3.csv .. paper_table7.csv (two panels for Table 4),
#         paper_fig1_data.csv,
#         t0_scale_by_n.csv, manuscript_numbers.json, figures/*
#
# Usage:
#   python analyze_v3.py --input trajectory_per_seed.csv \
#       --outdir analysis_hea --ansatz hea
#
# The verdict thresholds are declared below and printed into the verdict
# files, so that adjudication is mechanical and pre-committed rather than
# post-hoc.

import argparse
import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEGACY = ["vanilla", "pcgrad", "lso_pcgrad"]
NEW = ["pcgrad_nm", "lso_grid", "vanilla_ls_nm"]
ALL6 = LEGACY + NEW
RUN_KEYS = ["n_qubits", "depth", "variant", "seed"]
ROW_KEYS = RUN_KEYS + ["step"]

EXPECTED_PROBES = {
    "vanilla": (0, 0),
    "pcgrad": (0, 0),
    "pcgrad_nm": (0, 0),
    "lso_pcgrad": (7, 8),
    "lso_grid": (5, 5),
    "vanilla_ls_nm": (5, 5),
}

REQUIRED_COLUMNS = [
    "n_qubits",
    "depth",
    "variant",
    "seed",
    "step",
    "energy",
    "pred_decrease_lin",
    "U_step",
    "Q_step",
    "grad_norm_l2",
    "B_eff_mean",
    "Q_mean",
    "norm_ratio",
    "pc_norm_ratio",
    "probe_eval_count",
    "nm_fallback",
    "degenerate_anchor",
    "cos_to_vanilla",
    "selected_s",
]

# Pre-committed verdict thresholds. These are printed into verdict files.
CORR_RETAINED = 0.95
CORR_COLLAPSED = 0.85
HARM_RESCUE_FRACTION = 0.50
EQUIV_MARGIN = 0.02
N_BOOT = 5000
BOOT_SEED = 20260802
SCHEMA_VERSION = 4
NUM_TOL = 1e-9


def n_params(ansatz: str, n: int, d: int) -> int:
    if ansatz == "hea":
        return 2 * n * d
    if ansatz == "hva":
        return (2 * n - 1) * d
    raise ValueError("unknown ansatz: " + str(ansatz))


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(ROW_KEYS).reset_index(drop=True)
    if all(c in df.columns for c in RUN_KEYS + ["energy"]):
        g = df.groupby(RUN_KEYS, sort=False)
        df["dE_act"] = g["energy"].transform(lambda s: s - s.shift(-1))
    return df


def transition_frame(df: pd.DataFrame) -> pd.DataFrame:
    run_max = df.groupby(RUN_KEYS, sort=False)["step"].transform("max")
    return df[df["step"] < run_max].copy()


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    run_max = df.groupby(RUN_KEYS, sort=False)["step"].transform("max")
    return df[df["step"] == run_max].copy()


def safe_corr(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(
    values: Sequence[float],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Tuple[float, float]:
    """Paired bootstrap CI for one condition, resampling seed pairs."""
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def hierarchical_boot_ci(
    values_by_condition: Mapping[Tuple[int, int], Sequence[float]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Tuple[float, float]:
    """Hierarchical bootstrap across conditions and paired seeds.

    Conditions are resampled first. Within each selected condition, paired
    seed differences are resampled. The replicate statistic is the mean of
    the resampled condition means, so every (n, d) condition receives equal
    weight, matching the manuscript aggregation protocol.
    """
    clean: Dict[Tuple[int, int], np.ndarray] = {}
    for key, values in values_by_condition.items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            clean[key] = v
    keys = list(clean.keys())
    if not keys:
        return (np.nan, np.nan)

    rng = np.random.RandomState(seed)
    reps = np.empty(n_boot, dtype=float)
    n_cond = len(keys)
    for b in range(n_boot):
        sampled_idx = rng.randint(0, n_cond, size=n_cond)
        condition_means: List[float] = []
        for idx in sampled_idx:
            v = clean[keys[idx]]
            seed_idx = rng.randint(0, len(v), size=len(v))
            condition_means.append(float(v[seed_idx].mean()))
        reps[b] = float(np.mean(condition_means))
    return (
        float(np.percentile(reps, 2.5)),
        float(np.percentile(reps, 97.5)),
    )


def condition_balanced_mean(
    values_by_condition: Mapping[Tuple[int, int], Sequence[float]],
) -> float:
    means = []
    for values in values_by_condition.values():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            means.append(float(v.mean()))
    return float(np.mean(means)) if means else np.nan


def _audit_line(lines: List[str], label: str, ok: bool, detail: str) -> None:
    lines.append("{0}: {1}  [{2}]".format(label, detail, "OK" if ok else "FAIL"))


def schema_audit(
    df: pd.DataFrame,
    out: str,
    input_path: str,
    expected_states: int,
    expected_seeds: int,
) -> Tuple[str, bool]:
    lines = ["SCHEMA AUDIT", "=" * 60]
    checks: List[bool] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    ok = len(missing) == 0
    checks.append(ok)
    _audit_line(
        lines,
        "required columns",
        ok,
        "none missing" if ok else "missing " + ", ".join(missing),
    )

    found_variants = sorted(df["variant"].dropna().astype(str).unique()) if "variant" in df.columns else []
    exact_variants = set(found_variants) == set(ALL6)
    checks.append(exact_variants)
    _audit_line(
        lines,
        "exact six variants",
        exact_variants,
        ", ".join(found_variants),
    )

    if "schema_version" in df.columns:
        versions = sorted(
            pd.to_numeric(df["schema_version"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        schema_ok = versions == [SCHEMA_VERSION]
        detail = "CSV column found " + ", ".join(str(v) for v in versions)
    else:
        config_path = os.path.join(os.path.dirname(os.path.abspath(input_path)), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                version = int(cfg.get("schema_version", -1))
                schema_ok = version == SCHEMA_VERSION
                detail = "config.json found {0}".format(version)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                schema_ok = False
                detail = "could not read config.json: " + str(exc)
        else:
            schema_ok = False
            detail = "neither CSV schema_version nor adjacent config.json found"
    checks.append(schema_ok)
    _audit_line(lines, "schema version", schema_ok, detail)

    if all(c in df.columns for c in ROW_KEYS):
        dup_count = int(df.duplicated(ROW_KEYS).sum())
        dup_ok = dup_count == 0
        checks.append(dup_ok)
        _audit_line(lines, "duplicate row keys", dup_ok, str(dup_count))

        run_sizes = df.groupby(RUN_KEYS, sort=False).size()
        state_count_ok = bool((run_sizes == expected_states).all())
        checks.append(state_count_ok)
        bad_runs = int((run_sizes != expected_states).sum())
        _audit_line(
            lines,
            "states per run",
            state_count_ok,
            "expected {0}, bad runs {1}".format(expected_states, bad_runs),
        )

        expected_steps = set(range(expected_states))
        step_bad = 0
        for _, s in df.groupby(RUN_KEYS, sort=False)["step"]:
            observed = set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist())
            if observed != expected_steps:
                step_bad += 1
        step_ok = step_bad == 0
        checks.append(step_ok)
        _audit_line(
            lines,
            "step index coverage",
            step_ok,
            "expected 0..{0}, bad runs {1}".format(expected_states - 1, step_bad),
        )

        seed_counts = (
            df.groupby(["n_qubits", "depth", "variant"], sort=False)["seed"]
            .nunique()
        )
        if expected_seeds > 0:
            seeds_ok = bool((seed_counts == expected_seeds).all())
            bad_cells = int((seed_counts != expected_seeds).sum())
            detail = "expected {0}, bad cells {1}".format(expected_seeds, bad_cells)
        else:
            seeds_ok = bool(seed_counts.nunique() == 1)
            detail = "inferred counts " + ", ".join(str(x) for x in sorted(seed_counts.unique()))
        checks.append(seeds_ok)
        _audit_line(lines, "seeds per (n,d,variant)", seeds_ok, detail)

    if not missing:
        null_counts = df[REQUIRED_COLUMNS].isna().sum()
        # dE_act is not part of REQUIRED_COLUMNS and final-state NaNs are expected.
        null_total = int(null_counts.sum())
        null_ok = null_total == 0
        checks.append(null_ok)
        _audit_line(lines, "missing required values", null_ok, str(null_total))

        tr = transition_frame(df)
        for var in ALL6:
            s = tr[tr["variant"] == var]
            exp_lo, exp_hi = EXPECTED_PROBES[var]
            counts = pd.to_numeric(s["probe_eval_count"], errors="coerce")
            probe_ok = len(s) > 0 and bool(counts.between(exp_lo, exp_hi).all())
            checks.append(probe_ok)
            detail = "observed {0}/{1}, expected [{2},{3}]".format(
                int(counts.min()) if len(counts) else -1,
                int(counts.max()) if len(counts) else -1,
                exp_lo,
                exp_hi,
            )
            _audit_line(lines, "probe budget " + var, probe_ok, detail)

        nm = tr[tr["variant"] == "pcgrad_nm"]
        if len(nm):
            nr = pd.to_numeric(nm["norm_ratio"], errors="coerce").to_numpy(dtype=float)
            dev = float(np.nanmax(np.abs(nr - 1.0)))
            norm_ok = np.isfinite(dev) and dev < NUM_TOL
            checks.append(norm_ok)
            _audit_line(
                lines,
                "pcgrad_nm norm matching",
                norm_ok,
                "max|norm_ratio-1|={0:.3e}".format(dev),
            )
            lines.append(
                "pcgrad_nm diagnostics: fallbacks={0}, degenerate anchors={1}".format(
                    int(pd.to_numeric(nm["nm_fallback"], errors="coerce").fillna(0).sum()),
                    int(pd.to_numeric(nm["degenerate_anchor"], errors="coerce").fillna(0).sum()),
                )
            )

        pc = tr[tr["variant"] == "pcgrad"]
        if len(pc):
            nr = pd.to_numeric(pc["norm_ratio"], errors="coerce").to_numpy(dtype=float)
            pr = pd.to_numeric(pc["pc_norm_ratio"], errors="coerce").to_numpy(dtype=float)
            dev = float(np.nanmax(np.abs(nr - pr)))
            pc_ok = np.isfinite(dev) and dev < NUM_TOL
            checks.append(pc_ok)
            _audit_line(
                lines,
                "pcgrad norm_ratio consistency",
                pc_ok,
                "max|norm_ratio-pc_norm_ratio|={0:.3e}".format(dev),
            )

    overall_ok = bool(checks) and all(checks)
    lines.append("")
    lines.append("OVERALL: " + ("OK" if overall_ok else "FAIL"))
    txt = "\n".join(lines)
    with open(os.path.join(out, "schema_audit.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return txt, overall_ok


def final_frame(df: pd.DataFrame) -> pd.DataFrame:
    fin = final_rows(df)
    return fin.pivot_table(
        index=["n_qubits", "depth", "seed"],
        columns="variant",
        values="energy",
        aggfunc="first",
    )


def classify_predictability(cp: float, cn: float) -> Tuple[str, str]:
    if not np.isfinite(cp) or not np.isfinite(cn):
        return ("insufficient", "insufficient correlation data")
    if cp >= CORR_RETAINED and cn >= CORR_RETAINED:
        return (
            "intact_both",
            "predictability remains vanilla-like for both full and norm-matched projection",
        )
    if cp <= CORR_COLLAPSED and cn >= CORR_RETAINED:
        return (
            "norm_match_recovers",
            "norm matching restores predictability; this is consistent with step-size inflation as the primary contributor",
        )
    if cn <= CORR_COLLAPSED:
        return (
            "degraded_after_norm_match",
            "predictability remains degraded after norm matching; directional rotation remains associated with the loss",
        )
    return (
        "partial_mixed",
        "partial or intermediate outcome; neither pre-committed threshold pattern is met",
    )


def q1_analysis(df: pd.DataFrame, out: str):
    tr = transition_frame(df)
    fin = final_frame(df)

    rows = []
    for (n, d), grp in fin.groupby(level=["n_qubits", "depth"]):
        row = {"n": n, "d": d}
        for var in ["pcgrad", "pcgrad_nm", "lso_pcgrad"]:
            if var in grp.columns and "vanilla" in grp.columns:
                row["dE_" + var] = float((grp[var] - grp["vanilla"]).mean())
        rows.append(row)
    per_nd = pd.DataFrame(rows)

    corr_rows = []
    verdict_rows = []
    for n, g_n in tr.groupby("n_qubits"):
        row = {"n": n}
        for var in ALL6:
            s = g_n[g_n["variant"] == var].dropna(subset=["dE_act"])
            row["corr_" + var] = safe_corr(s["pred_decrease_lin"], s["dE_act"])
        for var in ["pcgrad", "pcgrad_nm"]:
            s = g_n[g_n["variant"] == var]
            if len(s):
                row["cos_" + var] = float(pd.to_numeric(s["cos_to_vanilla"], errors="coerce").mean())
                row["pcr_mean_" + var] = float(pd.to_numeric(s["pc_norm_ratio"], errors="coerce").mean())
                row["pcr_med_" + var] = float(pd.to_numeric(s["pc_norm_ratio"], errors="coerce").median())
        corr_rows.append(row)

        cp = row.get("corr_pcgrad", np.nan)
        cn = row.get("corr_pcgrad_nm", np.nan)
        code, text = classify_predictability(cp, cn)
        verdict_rows.append(
            {
                "n": n,
                "corr_pcgrad": cp,
                "corr_pcgrad_nm": cn,
                "verdict_code": code,
                "verdict_text": text,
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    by_n = pd.DataFrame(verdict_rows)
    per_nd.to_csv(os.path.join(out, "q1_summary.csv"), index=False)
    corr_df.to_csv(os.path.join(out, "q1_correlations.csv"), index=False)
    by_n.to_csv(os.path.join(out, "q1_verdict_by_n.csv"), index=False)

    lines = [
        "Q1 VERDICT -- directional rotation versus norm inflation",
        "=" * 60,
        "thresholds: retained corr >= {0}, collapsed corr <= {1}, rescue if harm_nm <= {2} x harm_pc".format(
            CORR_RETAINED,
            CORR_COLLAPSED,
            HARM_RESCUE_FRACTION,
        ),
        "Interpretive rule: single-mechanism wording is used only when all n values receive the same verdict.",
        "",
    ]

    harm_pc = per_nd.get("dE_pcgrad")
    harm_nm = per_nd.get("dE_pcgrad_nm")
    if harm_pc is not None and harm_nm is not None:
        hp = float(harm_pc.mean())
        hn = float(harm_nm.mean())
        lines.append(
            "condition-balanced mean dE_final vs vanilla: pcgrad {0:+.4f}, pcgrad_nm {1:+.4f}".format(
                hp, hn
            )
        )

        cp = float(corr_df["corr_pcgrad"].mean()) if "corr_pcgrad" in corr_df else np.nan
        cn = float(corr_df["corr_pcgrad_nm"].mean()) if "corr_pcgrad_nm" in corr_df else np.nan
        lines.append(
            "mean across n of pooled corr(pred, act): pcgrad {0:.4f}, pcgrad_nm {1:.4f}".format(
                cp, cn
            )
        )
        lines.append("")
        lines.append("Per-n predictability adjudication:")
        for _, r in by_n.sort_values("n").iterrows():
            lines.append(
                "  n={0}: pcgrad={1:.4f}, pcgrad_nm={2:.4f} -> {3}".format(
                    int(r["n"]),
                    float(r["corr_pcgrad"]),
                    float(r["corr_pcgrad_nm"]),
                    r["verdict_text"],
                )
            )

        codes = [c for c in by_n["verdict_code"].tolist() if c != "insufficient"]
        lines.append("")
        if not codes:
            lines.append(
                "-> Predictability correlations are insufficient for threshold adjudication; report the available values without a mechanism claim."
            )
        elif len(set(codes)) == 1 and len(codes) == len(by_n):
            code = codes[0]
            if code == "intact_both":
                lines.append(
                    "-> Predictability remains intact for both rules at every tested n; the collapse axis is not supported in this dataset."
                )
            elif code == "norm_match_recovers":
                lines.append(
                    "-> Across all tested n, norm matching restores predictability; this is consistent with step-size inflation as the primary contributor."
                )
            elif code == "degraded_after_norm_match":
                lines.append(
                    "-> Across all tested n, predictability remains degraded after norm matching; directional rotation remains associated with the loss."
                )
            else:
                lines.append(
                    "-> All tested n show intermediate outcomes; report the correlations without a binary mechanism claim."
                )
        else:
            lines.append(
                "-> SCALE-DEPENDENT OR MIXED outcome: the per-n verdicts are not uniform, so no single-mechanism conclusion is warranted."
            )

        if hp > 0.0:
            if hn <= 0.0:
                lines.append(
                    "-> Energy harm is fully rescued or reversed by norm matching; this is consistent with norm inflation contributing materially to the harm."
                )
            elif hn <= HARM_RESCUE_FRACTION * hp + NUM_TOL:
                lines.append(
                    "-> Energy harm is substantially attenuated by norm matching; this is consistent with norm inflation as a major contributor."
                )
            else:
                lines.append(
                    "-> Energy harm persists after norm matching; directional quality or trajectory effects remain important."
                )
        else:
            lines.append(
                "-> Full PCGrad is not harmful on average across conditions; interpret per-condition outcomes rather than invoking harm rescue."
            )

    txt = "\n".join(lines)
    with open(os.path.join(out, "q1_verdict.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return per_nd, corr_df, by_n, txt


def grade_q2_outcome(lo: float, hi: float,
                     margin: float = EQUIV_MARGIN) -> str:
    """Pure, pre-committed grading of the Q2 interval versus the margin.

    Returns one of: "equivalence", "separation_grid",
    "separation_vls", "small_grid", "small_vls", "inconclusive".
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if -margin <= lo and hi <= margin:
        return "equivalence"
    if hi < -margin:
        return "separation_grid"
    if lo > margin:
        return "separation_vls"
    if hi < 0.0:
        return "small_grid"
    if lo > 0.0:
        return "small_vls"
    return "inconclusive"


def q2_analysis(df: pd.DataFrame, out: str):
    fin = final_frame(df)
    rows = []
    raw_by_condition: Dict[Tuple[int, int], np.ndarray] = {}
    perq_by_condition: Dict[Tuple[int, int], np.ndarray] = {}

    for (n, d), grp in fin.groupby(level=["n_qubits", "depth"]):
        if "lso_grid" not in grp.columns or "vanilla_ls_nm" not in grp.columns:
            continue
        diff = (grp["lso_grid"] - grp["vanilla_ls_nm"]).dropna().to_numpy(dtype=float)
        if len(diff) == 0:
            continue
        diff_per_q = diff / float(n)
        raw_by_condition[(int(n), int(d))] = diff
        perq_by_condition[(int(n), int(d))] = diff_per_q

        lo, hi = boot_ci(diff, seed=BOOT_SEED + int(n) * 100 + int(d))
        lo_q, hi_q = boot_ci(diff_per_q, seed=BOOT_SEED + 10000 + int(n) * 100 + int(d))
        row = {
            "n": n,
            "d": d,
            "n_seeds": int(len(diff)),
            "mean_grid_minus_vls": float(diff.mean()),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "mean_grid_minus_vls_per_qubit": float(diff_per_q.mean()),
            "ci95_lo_per_qubit": lo_q,
            "ci95_hi_per_qubit": hi_q,
            "grid_strictly_better": int((diff < 0).sum()),
            "grid_strictly_better_fraction": float((diff < 0).mean()),
        }
        if "lso_pcgrad" in grp.columns:
            row["grid_minus_lso"] = float((grp["lso_grid"] - grp["lso_pcgrad"]).mean())
        rows.append(row)

    q2 = pd.DataFrame(rows)
    q2.to_csv(os.path.join(out, "q2_paired.csv"), index=False)

    overall_mean = condition_balanced_mean(raw_by_condition)
    overall_ci = hierarchical_boot_ci(raw_by_condition)
    overall_mean_per_q = condition_balanced_mean(perq_by_condition)
    overall_ci_per_q = hierarchical_boot_ci(
        perq_by_condition,
        seed=BOOT_SEED + 1,
    )

    lines = [
        "Q2 VERDICT -- projected direction versus matched-budget line search",
        "=" * 60,
        "grading rule at margin m = {0}: EQUIVALENCE if the entire CI lies within [-m, +m]; ".format(
            EQUIV_MARGIN
        )
        + "SEPARATION if the entire CI lies beyond +-m; direction consistent but practical size "
        + "UNRESOLVED if the CI excludes 0 but crosses the margin band; INCONCLUSIVE if the CI covers "
        + "0 and extends beyond +-m (too wide to grade)",
        "overall CI: hierarchical bootstrap over (n,d) conditions and paired seeds; conditions receive equal weight",
        "per-condition CI: paired bootstrap over seeds",
        "",
    ]

    if raw_by_condition:
        lo, hi = overall_ci
        lo_q, hi_q = overall_ci_per_q
        n_pairs = int(sum(len(v) for v in raw_by_condition.values()))
        lines.append(
            "overall condition-balanced mean (grid - vls) = {0:+.4f}, hierarchical CI95 [{1:+.4f}, {2:+.4f}], conditions={3}, pairs={4}".format(
                overall_mean,
                lo,
                hi,
                len(raw_by_condition),
                n_pairs,
            )
        )
        lines.append(
            "per-qubit sensitivity mean = {0:+.5f}, hierarchical CI95 [{1:+.5f}, {2:+.5f}]".format(
                overall_mean_per_q,
                lo_q,
                hi_q,
            )
        )

        grade = grade_q2_outcome(lo, hi, EQUIV_MARGIN)
        if grade == "equivalence":
            lines.append(
                "-> PRACTICAL EQUIVALENCE: the entire interval lies within the +-margin, providing evidence of no practically meaningful difference; this is consistent with the benefit being largely attributable to the shared search or step-norm adaptation."
            )
        elif grade == "separation_grid":
            lines.append(
                "-> lso_grid is clearly better with practical separation: the projected direction contributes beyond matched step-size control."
            )
        elif grade == "separation_vls":
            lines.append(
                "-> vanilla_ls_nm is clearly better with practical separation: the projected direction is a liability even when gated."
            )
        elif grade == "small_grid":
            lines.append(
                "-> lso_grid advantage is statistically consistent in direction, but its practical size is UNRESOLVED: the CI crosses the +-margin."
            )
        elif grade == "small_vls":
            lines.append(
                "-> vanilla_ls_nm advantage is statistically consistent in direction, but its practical size is UNRESOLVED: the CI crosses the +-margin."
            )
        else:
            lines.append(
                "-> INCONCLUSIVE: the interval both covers 0 and extends beyond the +-margin, so it is too wide to grade at the pre-committed margin (underpowered for this comparison); inspect condition-level estimates."
            )

        if "grid_minus_lso" in q2.columns:
            max_abs = float(q2["grid_minus_lso"].abs().max())
            mean_abs = float(q2["grid_minus_lso"].abs().mean())
            lines.append(
                "faithfulness to original LSO: max condition |lso_grid-lso_pcgrad|={0:.4f}, mean absolute difference={1:.4f}".format(
                    max_abs,
                    mean_abs,
                )
            )

    stats = {
        "overall_mean": overall_mean,
        "overall_ci95": [overall_ci[0], overall_ci[1]],
        "overall_mean_per_qubit": overall_mean_per_q,
        "overall_ci95_per_qubit": [overall_ci_per_q[0], overall_ci_per_q[1]],
        "n_conditions": len(raw_by_condition),
        "n_pairs": int(sum(len(v) for v in raw_by_condition.values())),
    }

    txt = "\n".join(lines)
    with open(os.path.join(out, "q2_verdict.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return q2, stats, txt


def coupling_and_quadrants(df: pd.DataFrame, out: str) -> pd.DataFrame:
    tr = transition_frame(df).dropna(subset=["dE_act"])

    rows = []
    for (n, var), s in tr.groupby(["n_qubits", "variant"]):
        sq = np.sqrt(pd.to_numeric(s["Q_step"], errors="coerce").clip(lower=0.0))
        rows.append(
            {
                "n": n,
                "variant": var,
                "corr_U_sqrtQ": safe_corr(s["U_step"], sq),
                "corr_U_dE": safe_corr(s["U_step"], s["dE_act"]),
                "corr_sqrtQ_dE": safe_corr(sq, s["dE_act"]),
                "corr_pred_dE": safe_corr(s["pred_decrease_lin"], s["dE_act"]),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out, "coupling_by_n.csv"), index=False)

    q_rows = []
    # All six variants are published symmetrically. In particular,
    # pcgrad fulfills the manuscript Section 6.3 statement that the
    # blind-PCGrad quadrant reversal data are available, and
    # pcgrad_nm provides a free auxiliary Q1 diagnostic: whether the
    # reversal survives norm matching localizes it to direction
    # quality versus step size.
    quadrant_variants = list(ALL6)
    for var in quadrant_variants:
        var_df = tr[tr["variant"] == var]
        if var_df.empty:
            continue
        for n, s in var_df.groupby("n_qubits"):
            mu = float(s["U_step"].median())
            mq = float(s["Q_step"].median())
            hi_vals = s[(s["U_step"] > mu) & (s["Q_step"] <= mq)]["dE_act"]
            lo_vals = s[(s["U_step"] <= mu) & (s["Q_step"] > mq)]["dE_act"]
            hi_u_lo_q = float(hi_vals.mean())
            lo_u_hi_q = float(lo_vals.mean())
            denom = hi_u_lo_q
            ratio = lo_u_hi_q / denom if np.isfinite(denom) and abs(denom) > 1e-15 else np.nan
            q_rows.append(
                {
                    "variant": var,
                    "n": n,
                    "U_median": mu,
                    "Q_median": mq,
                    "n_highU_lowQ": int(len(hi_vals)),
                    "n_lowU_highQ": int(len(lo_vals)),
                    "highU_lowQ": hi_u_lo_q,
                    "lowU_highQ": lo_u_hi_q,
                    "abs_diff": lo_u_hi_q - hi_u_lo_q,
                    "ratio": ratio,
                }
            )

    qd = pd.DataFrame(q_rows)
    qd.to_csv(os.path.join(out, "quadrant_by_n.csv"), index=False)
    return qd

def quadrant_table(tr: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Median-split quadrant table at an arbitrary grouping granularity.

    Mirrors the quadrant logic of coupling_and_quadrants exactly
    (within-variant medians, > / <= split conventions, identical
    output fields), with the grouping unit given by group_cols.
    """
    out_names = {"n_qubits": "n", "depth": "d", "phase": "phase"}
    q_rows = []
    for var in ALL6:
        var_df = tr[tr["variant"] == var]
        if var_df.empty:
            continue
        for keys, s in var_df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            mu = float(s["U_step"].median())
            mq = float(s["Q_step"].median())
            hi_vals = s[(s["U_step"] > mu) & (s["Q_step"] <= mq)]["dE_act"]
            lo_vals = s[(s["U_step"] <= mu) & (s["Q_step"] > mq)]["dE_act"]
            hi_u_lo_q = float(hi_vals.mean())
            lo_u_hi_q = float(lo_vals.mean())
            denom = hi_u_lo_q
            ratio = lo_u_hi_q / denom if np.isfinite(denom) and abs(denom) > 1e-15 else np.nan
            row = {"variant": var}
            for col, key in zip(group_cols, keys):
                row[out_names.get(col, col)] = key
            row.update(
                {
                    "U_median": mu,
                    "Q_median": mq,
                    "n_highU_lowQ": int(len(hi_vals)),
                    "n_lowU_highQ": int(len(lo_vals)),
                    "highU_lowQ": hi_u_lo_q,
                    "lowU_highQ": lo_u_hi_q,
                    "abs_diff": lo_u_hi_q - hi_u_lo_q,
                    "ratio": ratio,
                }
            )
            q_rows.append(row)
    return pd.DataFrame(q_rows)


def quadrant_sensitivity(df: pd.DataFrame, out: str) -> None:
    """Sensitivity checks for the quadrant analysis (manuscript Sec. 9.4).

    Repeats the median-split quadrant classification at finer grouping
    units, (n, d) and (n, d, phase), and prints a robustness summary
    counting the groups in which the low-U/high-Q regime yields the
    larger mean realized decrease.
    """
    tr = transition_frame(df).dropna(subset=["dE_act"]).copy()
    step = tr["step"].astype(int)
    tr["phase"] = np.where(step <= 20, "early",
                           np.where(step <= 60, "mid", "late"))

    for group_cols, fname, label in [
        (["n_qubits", "depth"], "quadrant_by_nd.csv", "(n,d)"),
        (["n_qubits", "depth", "phase"], "quadrant_by_ndphase.csv", "(n,d,phase)"),
    ]:
        qt = quadrant_table(tr, group_cols)
        qt.to_csv(os.path.join(out, fname), index=False)
        for var in ALL6:
            s = qt[qt["variant"] == var]
            fin = s[np.isfinite(s["abs_diff"])]
            k = int((fin["abs_diff"] > 0).sum())
            m = int(len(fin))
            if m < len(s):
                print("  WARNING: {0} group(s) dropped due to empty quadrant".format(len(s) - m))
            print(
                "quadrant sensitivity {0} {1}: lowU_highQ > highU_lowQ in {2}/{3} groups".format(
                    label, var, k, m
                )
            )

def t0_scale(df: pd.DataFrame, ansatz: str, out: str) -> pd.DataFrame:
    t0 = df[df["step"] == 0]
    rows = []
    for (n, d, var), s in t0.groupby(["n_qubits", "depth", "variant"]):
        if var != "vanilla":
            continue
        k = n_params(ansatz, int(n), int(d))
        rows.append(
            {
                "n": n,
                "d": d,
                "g2_over_K": float((s["grad_norm_l2"] ** 2).mean() / k),
                "B_eff_t0": float(s["B_eff_mean"].mean()),
                "Q_t0": float(s["Q_mean"].mean()),
            }
        )
    t0df = pd.DataFrame(rows)
    t0df.to_csv(os.path.join(out, "t0_scale_by_n.csv"), index=False)
    return t0df


def make_figures(
    df: pd.DataFrame,
    t0df: pd.DataFrame,
    qd: pd.DataFrame,
    out: str,
    fig_format: str,
) -> None:
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)
    tr = transition_frame(df)

    if len(t0df) and t0df["n"].nunique() >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for d, s in t0df.groupby("d"):
            s = s.sort_values("n")
            ax.semilogy(s["n"], s["g2_over_K"], "o-", label="d={0}".format(d))
        ax.set_xlabel("qubits n")
        ax.set_ylabel("t=0  ||g||^2 / K  (vanilla)")
        ax.set_title("Onset-trend evidence: per-parameter gradient scale")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "t0_scale." + fig_format))
        plt.close(fig)

    if "pc_norm_ratio" in tr.columns:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for var in ["pcgrad", "pcgrad_nm"]:
            s = tr[tr["variant"] == var]
            if len(s):
                ax.hist(
                    s["pc_norm_ratio"],
                    bins=60,
                    alpha=0.6,
                    density=True,
                    label=var,
                )
        ax.axvline(1.0, linewidth=1)
        ax.set_xlabel("raw pc_norm_ratio along own trajectory")
        ax.set_ylabel("density")
        ax.set_title("State dependence of the raw projection-norm ratio")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "pc_norm_ratio." + fig_format))
        plt.close(fig)

    if len(qd):
        for var, s in qd.groupby("variant"):
            s = s.sort_values("n").reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            x = np.arange(len(s))
            w = 0.35
            ax.bar(x - w / 2, s["highU_lowQ"], w, label="high-U / low-Q")
            ax.bar(x + w / 2, s["lowU_highQ"], w, label="low-U / high-Q")
            for xi, row in s.iterrows():
                y = max(float(row["highU_lowQ"]), float(row["lowU_highQ"]), 0.0)
                ax.text(
                    xi,
                    y,
                    "diff={0:+.4f}".format(float(row["abs_diff"])),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(["n={0}".format(int(n)) for n in s["n"]])
            ax.set_ylabel("mean dE_act")
            ax.set_title("Quadrant trade-off under {0}".format(var))
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                os.path.join(figdir, "quadrant_absdiff_{0}.{1}".format(var, fig_format))
            )
            plt.close(fig)


def finite_or_none(value):
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value



PRIMARY = list(LEGACY)
CONTROLS = list(NEW)


def paper_tables(df: pd.DataFrame, out: str, ansatz: str) -> None:
    """Emit the manuscript's Tables 3-7 and Fig. 1 data from v3 records.

    All quantities follow the manuscript protocols: condition-level
    aggregation before averaging over conditions (Table 3), pooled
    step-level correlations by system size (Tables 4-5), Section 9.4
    phase boundaries (Table 6), and within-variant median splits
    (Table 7). Winner cases are counted among the PRIMARY three rules
    only, so the x/360 denominator keeps its published meaning.
    """
    tr = transition_frame(df).dropna(subset=["dE_act"])
    fin = final_frame(df)

    # ---- Table 3 ----------------------------------------------------
    rows3 = []
    have = [v for v in PRIMARY if v in fin.columns]
    prim = fin[have].dropna()
    winners = prim.idxmin(axis=1)
    n_cases = int(len(prim))
    n_conds = int(prim.reset_index()[["n_qubits", "depth"]]
                  .drop_duplicates().shape[0])

    fin_b = final_rows(df).pivot_table(
        index=["n_qubits", "depth", "seed"], columns="variant",
        values="B_eff_mean", aggfunc="first")

    for method in ["pcgrad", "lso_pcgrad"]:
        if method not in fin.columns:
            continue
        de_cond = (fin[method] - fin["vanilla"]).groupby(
            level=["n_qubits", "depth"]).mean()
        db_cond = (fin_b[method] - fin_b["vanilla"]).groupby(
            level=["n_qubits", "depth"]).mean()
        rows3.append({
            "ansatz": ansatz,
            "method": method,
            "mean_dB_eff": float(db_cond.mean()),
            "mean_dE_final": float(de_cond.mean()),
            "worse_conditions": "{0}/{1}".format(
                int((de_cond > 0).sum()), n_conds),
            "winner_cases": "{0}/{1}".format(
                int((winners == method).sum()), n_cases),
        })
    print("table3 note: vanilla wins {0}/{1} primary-rule cases "
          "(reference row, not part of the manuscript table)".format(
              int((winners == "vanilla").sum()), n_cases))
    pd.DataFrame(rows3).to_csv(
        os.path.join(out, "paper_table3.csv"), index=False)

    # ---- Tables 4 (two panels) and 5, and Fig. 1 data ---------------
    def coupling_row(sub):
        sq = np.sqrt(pd.to_numeric(sub["Q_step"],
                                   errors="coerce").clip(lower=0.0))
        return {
            "corr_U_sqrtQ": safe_corr(sub["U_step"], sq),
            "corr_U_dE": safe_corr(sub["U_step"], sub["dE_act"]),
            "corr_sqrtQ_dE": safe_corr(sq, sub["dE_act"]),
            "corr_pred_dE": safe_corr(sub["pred_decrease_lin"],
                                      sub["dE_act"]),
        }

    for panel, variants, fname in [
            ("A", PRIMARY, "paper_table4_panelA.csv"),
            ("B", CONTROLS, "paper_table4_panelB.csv")]:
        rows4 = []
        for n, g_n in tr.groupby("n_qubits"):
            row = {"ansatz": ansatz, "n": int(n)}
            for var in variants:
                s = g_n[g_n["variant"] == var]
                row[var] = (coupling_row(s)["corr_U_sqrtQ"]
                            if len(s) else np.nan)
            rows4.append(row)
        pd.DataFrame(rows4).to_csv(os.path.join(out, fname), index=False)

    rows5 = []
    for n, g_n in tr[tr["variant"] == "lso_pcgrad"].groupby("n_qubits"):
        row = {"ansatz": ansatz, "n": int(n)}
        row.update(coupling_row(g_n))
        rows5.append(row)
    t5 = pd.DataFrame(rows5)
    t5.to_csv(os.path.join(out, "paper_table5.csv"), index=False)
    t5[["ansatz", "n", "corr_U_sqrtQ"]].to_csv(
        os.path.join(out, "paper_fig1_data.csv"), index=False)

    # ---- Table 6 (phase-resolved, Section 9.4 boundaries) -----------
    lso = tr[tr["variant"] == "lso_pcgrad"].copy()
    step = lso["step"].astype(int)
    lso["phase"] = np.where(step <= 20, "early",
                            np.where(step <= 60, "mid", "late"))
    # Manuscript protocol (Table 6 caption): correlations are computed
    # within each (n, depth) and then averaged over circuit depths at
    # the given system size -- not pooled across depths.
    rows6 = []
    for n, g_n in lso.groupby("n_qubits"):
        row = {"ansatz": ansatz, "n": int(n)}
        for ph in ["early", "mid", "late"]:
            per_depth = []
            for d, g_d in g_n[g_n["phase"] == ph].groupby("depth"):
                per_depth.append(safe_corr(g_d["U_step"], g_d["dE_act"]))
            per_depth = [c for c in per_depth if np.isfinite(c)]
            row[ph] = float(np.mean(per_depth)) if per_depth else np.nan
        row["d_le"] = row["late"] - row["early"]
        rows6.append(row)
    pd.DataFrame(rows6).to_csv(
        os.path.join(out, "paper_table6.csv"), index=False)

    # ---- Table 7 (quadrant trade-off, LSO-PCGrad) -------------------
    rows7 = []
    for n, s in lso.groupby("n_qubits"):
        mu = s["U_step"].median()
        mq = s["Q_step"].median()
        hi_lo = s[(s["U_step"] > mu) & (s["Q_step"] <= mq)]["dE_act"].mean()
        lo_hi = s[(s["U_step"] <= mu) & (s["Q_step"] > mq)]["dE_act"].mean()
        rows7.append({
            "ansatz": ansatz, "n": int(n),
            "highU_lowQ": float(hi_lo), "lowU_highQ": float(lo_hi),
            "ratio": float(lo_hi / hi_lo) if hi_lo else np.nan,
        })
    pd.DataFrame(rows7).to_csv(
        os.path.join(out, "paper_table7.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ansatz", choices=["hea", "hva"], default="hea")
    ap.add_argument("--fig-format", default="pdf", choices=["pdf", "png"])
    ap.add_argument(
        "--expected-states",
        type=int,
        default=101,
        help="expected number of recorded states per run",
    )
    ap.add_argument(
        "--expected-seeds",
        type=int,
        default=30,
        help="expected seeds per (n,d,variant); use 0 to infer consistency only",
    )
    ap.add_argument(
        "--strict-audit",
        action="store_true",
        help="stop immediately when the schema audit fails",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.input)

    audit_txt, audit_ok = schema_audit(
        df,
        args.outdir,
        input_path=args.input,
        expected_states=args.expected_states,
        expected_seeds=args.expected_seeds,
    )
    print(audit_txt)
    if args.strict_audit and not audit_ok:
        raise SystemExit("Schema audit failed; analysis stopped by --strict-audit.")

    print()
    per_nd, corr_df, by_n, q1txt = q1_analysis(df, args.outdir)
    print(q1txt)
    print()
    q2, q2_stats, q2txt = q2_analysis(df, args.outdir)
    print(q2txt)

    qd = coupling_and_quadrants(df, args.outdir)
    quadrant_sensitivity(df, args.outdir)
    t0df = t0_scale(df, args.ansatz, args.outdir)
    paper_tables(df, args.outdir, args.ansatz)
    make_figures(df, t0df, qd, args.outdir, args.fig_format)

    nums = {
        "schema_ok": audit_ok,
        "q1_mean_dE_condition_balanced": {
            c: float(per_nd[c].mean())
            for c in per_nd.columns
            if c.startswith("dE_")
        },
        "q1_predictability_by_n": by_n.to_dict(orient="records"),
        "q2": q2_stats,
        "ansatz": args.ansatz,
        "thresholds": {
            "corr_retained": CORR_RETAINED,
            "corr_collapsed": CORR_COLLAPSED,
            "harm_rescue_fraction": HARM_RESCUE_FRACTION,
            "equivalence_margin": EQUIV_MARGIN,
            "n_boot": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
        },
    }
    with open(
        os.path.join(args.outdir, "manuscript_numbers.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(finite_or_none(nums), f, indent=2, allow_nan=False)

    print("\nSaved outputs in:", args.outdir)


if __name__ == "__main__":
    main()
