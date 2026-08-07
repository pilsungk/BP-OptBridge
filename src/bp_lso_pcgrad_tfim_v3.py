#!/usr/bin/env python3
# bp_lso_pcgrad_tfim_v3.py
#
# LSO-PCGrad experiment on TFIM with HEA (v3: matched-control ablations)
#
# LSO-PCGrad:
#   Lookahead Strength-Optimized PCGrad
#
# Example:
# python bp_lso_pcgrad_tfim_v3.py --n_qubits 4,6,8,10 --depths 4,6,8 --seeds 30 --steps 100 --lr 0.05
#
# Resume an interrupted run (config must match, schema_version checked):
# python bp_lso_pcgrad_tfim_v3.py --n_qubits 4,6,8,10 --depths 4,6,8 --seeds 30 --steps 100 --lr 0.05 --outdir <existing run dir>
#
# Purpose:
#   Compare six update rules in a BP-relevant optimization setting:
#     1) vanilla       : use the raw full gradient
#     2) pcgrad        : apply PCGrad to Hamiltonian-term gradient vectors
#                        (uncapped aggregate)
#     3) pcgrad_nm     : norm-matched blind PCGrad; same direction as
#                        pcgrad but rescaled to the vanilla gradient norm.
#                        Isolates direction change from norm inflation.
#     4) lso_pcgrad    : interpolate between vanilla and PCGrad using
#                        a probe-based scalar strength alpha
#                        (3 initial probes + quadratic vertex + extra
#                        candidate; 7-8 probe evaluations per step)
#     5) lso_grid      : LSO on a fixed candidate grid
#                        alpha in {0, 0.25, 0.5, 0.75, 1.0};
#                        exactly 5 probe evaluations per step
#     6) vanilla_ls_nm : norm-matched vanilla line-search control;
#                        for each grid alpha, probe s(alpha)*u_van with
#                        s(alpha) = ||u_van + alpha*delta_capped|| / ||u_van||,
#                        so each candidate has the same norm as the
#                        corresponding lso_grid candidate; exactly 5
#                        probe evaluations per step. Direction is the
#                        only difference from lso_grid.
#
#   The matched-budget ablation pair is lso_grid vs vanilla_ls_nm
#   (5 energy evaluations each). The original lso_pcgrad uses 7-8
#   evaluations and is kept for continuity with the main results.
#
# Main objects:
#   For each optimization step, let
#     g_anchor = sum of Hamiltonian-term gradients
#     g_pc     = PCGrad aggregate built from term gradients
#     delta    = g_pc - g_anchor
#
#   Then the actual update direction is
#     g_update = g_anchor + alpha * delta
#
# Variant definitions:
#   - vanilla:
#       alpha = 0, so g_update = g_anchor
#
#   - pcgrad:
#       alpha = 1, so g_update = g_pc
#
#   - lso_pcgrad:
#       choose alpha in [0, alpha_max] by minimizing a local probe objective
#
# Local probe problem:
#   alpha* = argmin_{alpha in [0, alpha_max]}
#            E(theta - lr * (g_anchor + alpha * delta))
#
# Practical alpha selection:
#   alpha* is approximated by
#     1) probing at alpha = 0, alpha_mid, alpha_max
#     2) fitting a quadratic through the probed points
#     3) testing the fitted vertex plus optional extra sampled points
#     4) selecting the alpha with the lowest probed energy
#
# Diagnostics recorded at each step:
#   Standard BP-DI diagnostics:
#     - R_mean
#     - N_eff_mean
#     - B_eff_mean
#     - Q_mean
#     - var_bridge_actual
#     - var_bridge_ratio
#
#   LSO-specific diagnostics:
#     - alpha
#     - s_dir
#     - probe
#     - probe_score
#
#   Step-level optimizer-usefulness diagnostics:
#     These are computed from directional term contributions
#       c_t = g_t^T g_update
#     where g_t is the gradient vector of Hamiltonian term t.
#
#     Recorded quantities include
#       - S_step             : signed survival ratio along the chosen update
#       - N_eff_step         : effective term count along the chosen update
#       - U_step             : optimization-usefulness score
#       - Q_step             : directional pre-cancellation activity scale
#       - dir_sum_step       : sum_t c_t
#       - pred_delta_lin     : first-order predicted energy change
#       - pred_decrease_lin  : first-order predicted energy decrease
#
# Interpretation:
#   - B_eff tracks gradient signal survival under destructive interference
#   - U_step tracks whether the chosen update direction is actually useful
#     for coherent descent across Hamiltonian terms
#
# Outputs:
#   - trajectory_per_seed.csv
#       Step-by-step optimization trace for every seed and variant
#
#   - final_compare.csv
#       Final-step comparison between vanilla and each intervention variant
#
#   - summary.json
#       Compact summary of run outputs
#
#   - figures/*.pdf
#       Trajectory plots for energy, BP-DI diagnostics, alpha, probe scores,
#       and step-level usefulness metrics

import os
import csv
import json
import time
import argparse
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_N_QUBITS = "4,6"
DEFAULT_DEPTHS = "4,6"
DEFAULT_SEEDS = 10
DEFAULT_STEPS = 100
DEFAULT_LR = 0.05
DEFAULT_OUTDIR_ROOT = "./runs"
DEFAULT_VARIANTS = "vanilla,pcgrad,pcgrad_nm,lso_pcgrad,lso_grid,vanilla_ls_nm"

H_FIELD = 1.0
RNG_INIT_LOW = -np.pi
RNG_INIT_HIGH = np.pi
EPS = 1e-15

SCHEMA_VERSION = 4

# Fallback guard for pcgrad_nm: if ||u_pc|| / ||u_van|| falls below this
# ratio, rescaling would amplify numerical direction noise; fall back to
# the vanilla direction and record nm_fallback = 1.
NM_RATIO_MIN = 1e-6

# Fixed candidate grid shared by lso_grid and vanilla_ls_nm.
GRID_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Set True by --selfcheck; enables runtime invariant assertions
# (norm matching, argmin correctness, bridge identity).
SELFCHECK = False

TRAJECTORY_KEYS = [
    "n_qubits", "depth", "variant", "seed", "step", "energy",
    "R_mean", "N_eff_mean", "B_eff_mean", "Q_mean",
    "grad_norm_l2", "var_bridge_actual", "var_bridge_ratio",
    "alpha", "s_dir", "probe", "probe_score",
    "S_step", "N_eff_step", "U_step", "Q_step",
    "dir_sum_step", "pred_delta_lin", "pred_decrease_lin",
    "update_norm", "step_norm", "norm_ratio", "pc_norm_ratio",
    "selected_s", "probe_eval_count", "nm_fallback",
    "degenerate_anchor", "cos_to_vanilla",
]

# Column semantics (identical schema for every variant):
#   norm_ratio    : ||u_actual|| / ||u_van|| of the update actually taken
#                   (1.0 for vanilla; 1.0 for pcgrad_nm unless fallback).
#   pc_norm_ratio : ||u_pc|| / ||u_van|| of the raw uncapped PCGrad
#                   aggregate BEFORE any rescaling; recorded by pcgrad
#                   and pcgrad_nm, 0.0 for other variants.
#   alpha         : variant-specific path parameter; interpolation
#                   strength for lso variants, the reference lambda that
#                   generated the selected norm for vanilla_ls_nm, and a
#                   nominal constant (0 or 1) for vanilla/pcgrad/
#                   pcgrad_nm.
#   degenerate_anchor : computed centrally as ||u_van|| <= EPS at the
#                   current parameter point, for every variant.

INT_ROW_KEYS = set([
    "n_qubits", "depth", "seed", "step",
    "probe_eval_count", "nm_fallback", "degenerate_anchor",
])
FLOAT_ROW_KEYS = set(TRAJECTORY_KEYS) - INT_ROW_KEYS - set(["variant"])

RESUME_CONFIG_KEYS = [
    "schema_version", "n_qubits", "depths", "seeds", "steps", "lr",
    "variants", "lso_cfg", "h_field", "grid_lambdas", "nm_ratio_min",
]


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def make_timestamped_outdir(root):
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(root, "bp_lso_pcgrad_tfim_" + ts)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    return outdir


def wrap_angles(x):
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def ry_mat(t):
    c = np.cos(t / 2.0)
    s = np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_mat(t):
    return np.diag([
        np.exp(-1j * t / 2.0),
        np.exp(1j * t / 2.0),
    ]).astype(np.complex128)


def apply_gate_sv(psi, gate, qubit, n):
    shape = [2] * n
    psi_t = psi.reshape(shape)
    ax = n - 1 - qubit
    psi_t = np.tensordot(gate, psi_t, axes=([1], [ax]))
    psi_t = np.moveaxis(psi_t, 0, ax)
    return psi_t.reshape(2 ** n)


_CNOT_CACHE = {}


def get_cnot(ctrl, tgt, n):
    key = (ctrl, tgt, n)
    if key in _CNOT_CACHE:
        return _CNOT_CACHE[key]
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        if (k >> ctrl) & 1:
            U[k ^ (1 << tgt), k] = 1.0
        else:
            U[k, k] = 1.0
    _CNOT_CACHE[key] = U
    return U


def build_h_obc_matrix(n, h=1.0):
    H = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    def kron_all(ops):
        r = ops[0]
        for o in ops[1:]:
            r = np.kron(r, o)
        return r

    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= kron_all(ops)

    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H += h * kron_all(ops)

    return H


def pauli_terms_for_h_obc(n, h=1.0):
    terms = []
    for i in range(n - 1):
        terms.append(("zz", i, i + 1, -1.0))
    for i in range(n):
        terms.append(("x", i, None, h))
    return terms


def build_h_term_matrix(n, term):
    kind, i, j, coeff = term
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    ops = [I2] * n
    if kind == "zz":
        ops[i] = Z
        ops[j] = Z
    elif kind == "x":
        ops[i] = X
    else:
        raise ValueError("Unknown term kind: " + str(kind))

    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return float(coeff), M


def build_h_term_matrices(n, h=1.0):
    mats = []
    for term in pauli_terms_for_h_obc(n, h):
        coeff, M = build_h_term_matrix(n, term)
        mats.append((term, coeff, M))
    return mats


def n_params_for_hea(n, depth):
    return 2 * n * depth


def final_state_sv_hea(params, n, depth):
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    idx = 0

    for _layer in range(depth):
        for q in range(n):
            psi = apply_gate_sv(psi, ry_mat(params[idx]), q, n)
            idx += 1
            psi = apply_gate_sv(psi, rz_mat(params[idx]), q, n)
            idx += 1
        for q in range(n - 1):
            psi = get_cnot(q, q + 1, n) @ psi
        if n > 1:
            psi = get_cnot(n - 1, 0, n) @ psi

    return psi


def energy_sv(params, n, depth, H_mat):
    psi = final_state_sv_hea(params, n, depth)
    e = np.real(np.vdot(psi, H_mat @ psi))
    return float(e)


def grad_termwise_sv(params, n, depth, term_mats):
    npar = len(params)
    nterms = len(term_mats)
    out = np.zeros((npar, nterms), dtype=np.float64)

    shift = np.pi / 2.0
    pref = 0.5

    for k in range(npar):
        pp = params.copy()
        pm = params.copy()
        pp[k] += shift
        pm[k] -= shift

        psi_p = final_state_sv_hea(pp, n, depth)
        psi_m = final_state_sv_hea(pm, n, depth)

        for t_idx, (_, h_coeff, M) in enumerate(term_mats):
            ep = np.real(np.vdot(psi_p, M @ psi_p))
            em = np.real(np.vdot(psi_m, M @ psi_m))
            out[k, t_idx] = h_coeff * pref * (ep - em)

    return out


def signed_cancellation_ratio(a):
    denom = float(np.sum(np.abs(a)))
    numer = float(np.abs(np.sum(a)))
    if denom <= EPS:
        return 0.0
    return numer / denom


def effective_term_count(a):
    abs_a = np.abs(a)
    denom = float(np.sum(abs_a ** 2))
    if denom <= EPS:
        return 0.0
    numer = float(np.sum(abs_a) ** 2)
    return numer / denom


def diagnostics_from_grad_terms(grad_terms):
    npar = grad_terms.shape[0]
    R_k = np.zeros(npar, dtype=np.float64)
    N_eff_k = np.zeros(npar, dtype=np.float64)
    B_eff_k = np.zeros(npar, dtype=np.float64)
    Q_k = np.zeros(npar, dtype=np.float64)
    grad_full = np.sum(grad_terms, axis=1)

    for k in range(npar):
        a = grad_terms[k]
        R_k[k] = signed_cancellation_ratio(a)
        N_eff_k[k] = effective_term_count(a)
        B_eff_k[k] = R_k[k] * np.sqrt(N_eff_k[k]) if N_eff_k[k] > EPS else 0.0
        Q_k[k] = float(np.sum(a ** 2))

    vb_actual = float(np.mean((B_eff_k ** 2) * Q_k))
    vb_approx = float(np.mean(B_eff_k ** 2) * np.mean(Q_k))

    return {
        "grad_full": grad_full,
        "R_mean": float(np.mean(R_k)),
        "N_eff_mean": float(np.mean(N_eff_k)),
        "B_eff_mean": float(np.mean(B_eff_k)),
        "Q_mean": float(np.mean(Q_k)),
        "grad_norm_l2": float(np.linalg.norm(grad_full)),
        "var_bridge_actual": vb_actual,
        "var_bridge_approx": vb_approx,
        "var_bridge_ratio": float(vb_actual / vb_approx) if vb_approx > EPS else 0.0,
    }

def step_usefulness_from_update(grad_terms, update_grad, lr):
    # grad_terms: shape (npar, nterms)
    # update_grad: shape (npar,)
    # directional term contributions along the chosen update direction
    c = grad_terms.T @ update_grad

    c_sum = float(np.sum(c))
    c_abs_sum = float(np.sum(np.abs(c)))
    c_sq_sum = float(np.sum(c ** 2))

    if c_abs_sum > EPS:
        S_step = c_sum / c_abs_sum
    else:
        S_step = 0.0

    if c_sq_sum > EPS:
        N_eff_step = (c_abs_sum ** 2) / c_sq_sum
        U_step = c_sum / np.sqrt(c_sq_sum)
    else:
        N_eff_step = 0.0
        U_step = 0.0

    # First-order predicted energy change:
    # E(theta - lr * u) approx E(theta) - lr * grad^T u
    pred_delta_lin = float(-lr * c_sum)
    pred_decrease_lin = float(lr * c_sum)

    return {
        "S_step": float(S_step),
        "N_eff_step": float(N_eff_step),
        "U_step": float(U_step),
        "Q_step": float(c_sq_sum),
        "dir_sum_step": float(c_sum),
        "pred_delta_lin": float(pred_delta_lin),
        "pred_decrease_lin": float(pred_decrease_lin),
    }

def pcgrad_project(task_grads):
    proj = [g.copy() for g in task_grads]
    m = len(proj)

    for i in range(m):
        gi = proj[i]
        for j in range(m):
            if i == j:
                continue
            gj = proj[j]
            dot = float(np.dot(gi, gj))
            gj_norm2 = float(np.dot(gj, gj))
            if dot < 0.0 and gj_norm2 > EPS:
                gi = gi - (dot / gj_norm2) * gj
        proj[i] = gi

    return np.sum(np.stack(proj, axis=0), axis=0)


def clip_delta(delta, ref_norm, cap_ratio):
    delta_norm = float(np.linalg.norm(delta))
    max_norm = cap_ratio * ref_norm
    if delta_norm <= max_norm or delta_norm <= EPS:
        return delta.copy()
    return delta * (max_norm / delta_norm)


def probe_energy_dir(params, n, depth, H_mat, upd, lr):
    # Energy at the trial point theta - lr * upd, without committing.
    new_params = wrap_angles(params - lr * upd)
    return energy_sv(new_params, n, depth, H_mat)


def probe_energy(params, n, depth, H_mat, g_anchor, delta, lr, alpha):
    upd = g_anchor + alpha * delta
    return probe_energy_dir(params, n, depth, H_mat, upd, lr)


def fit_quadratic_vertex(x0, y0, x1, y1, x2, y2):
    xs = np.array([x0, x1, x2], dtype=np.float64)
    ys = np.array([y0, y1, y1 if abs(x2 - x1) <= EPS and abs(x1 - x0) > EPS else y2], dtype=np.float64)

    if abs(x2 - x1) <= EPS or abs(x1 - x0) <= EPS or abs(x2 - x0) <= EPS:
        return None

    coeff = np.polyfit(xs, ys, 2)
    a = float(coeff[0])
    b = float(coeff[1])

    if abs(a) <= 1e-14:
        return None

    xv = -b / (2.0 * a)
    return float(xv)


def build_lso_update(params, n, depth, H_mat, grad_terms, lr, lso_cfg):
    g_anchor = np.sum(grad_terms, axis=1)
    task_grads = [grad_terms[:, t].copy() for t in range(grad_terms.shape[1])]
    g_pc = pcgrad_project(task_grads)

    delta = g_pc - g_anchor
    anchor_norm = float(np.linalg.norm(g_anchor))
    delta = clip_delta(delta, anchor_norm, lso_cfg["delta_cap_ratio"])
    delta_norm = float(np.linalg.norm(delta))

    if anchor_norm > EPS and delta_norm > EPS:
        s_dir = float(np.dot(g_anchor, delta) / (anchor_norm * delta_norm))
    else:
        s_dir = 0.0

    alpha_max = float(lso_cfg["alpha_max"])
    alpha_mid = float(lso_cfg["alpha_mid_frac"] * alpha_max)

    a0 = 0.0
    a1 = alpha_mid
    a2 = alpha_max

    probe_calls = 0

    f0 = probe_energy(params, n, depth, H_mat, g_anchor, delta, lr, a0)
    f1 = probe_energy(params, n, depth, H_mat, g_anchor, delta, lr, a1)
    f2 = probe_energy(params, n, depth, H_mat, g_anchor, delta, lr, a2)
    probe_calls += 3

    candidate_alphas = [a0, a1, a2]

    xv = fit_quadratic_vertex(a0, f0, a1, f1, a2, f2)
    if xv is not None and 0.0 <= xv <= alpha_max:
        candidate_alphas.append(float(xv))

    if lso_cfg["extra_probe_frac"] > 0.0:
        a3 = float(lso_cfg["extra_probe_frac"] * alpha_max)
        if 0.0 < a3 < alpha_max:
            candidate_alphas.append(a3)

    best_alpha = 0.0
    best_energy = f0

    seen = set()
    for alpha in candidate_alphas:
        key = round(alpha, 12)
        if key in seen:
            continue
        seen.add(key)
        val = probe_energy(params, n, depth, H_mat, g_anchor, delta, lr, alpha)
        probe_calls += 1
        if val < best_energy:
            best_energy = val
            best_alpha = float(alpha)

    update_grad = g_anchor + best_alpha * delta

    probe_delta = float(best_energy - f0)
    probe_score = float((f0 - best_energy) / max(lr * max(anchor_norm * anchor_norm, EPS), EPS))

    return {
        "update_grad": update_grad,
        "alpha": float(best_alpha),
        "s_dir": float(s_dir),
        "probe": float(probe_delta),
        "probe_score": float(probe_score),
        "probe_eval_count": int(probe_calls),
    }


def build_pcgrad_nm_update(grad_terms):
    # Norm-matched blind PCGrad: same direction as the uncapped PCGrad
    # aggregate, rescaled to the vanilla gradient norm. Isolates the
    # effect of the projected direction from norm inflation.
    g_anchor = np.sum(grad_terms, axis=1)
    task_grads = [grad_terms[:, t].copy() for t in range(grad_terms.shape[1])]
    g_pc = pcgrad_project(task_grads)

    anchor_norm = float(np.linalg.norm(g_anchor))
    pc_norm = float(np.linalg.norm(g_pc))

    if anchor_norm <= EPS:
        # Vanilla gradient itself is numerically zero; nothing to match.
        # The degenerate_anchor flag is recorded centrally in run_single.
        return {
            "update_grad": np.zeros_like(g_anchor),
            "alpha": 1.0,
            "s_dir": 0.0,
            "probe": 0.0,
            "probe_score": 0.0,
            "probe_eval_count": 0,
            "pc_norm_ratio": 0.0,
            "nm_fallback": 0,
        }

    ratio = pc_norm / anchor_norm

    if ratio < NM_RATIO_MIN:
        # Rescaling a near-zero projected direction would amplify
        # numerical noise; fall back to the vanilla direction.
        update_grad = g_anchor.copy()
        nm_fallback = 1
    else:
        update_grad = g_pc * (anchor_norm / pc_norm)
        nm_fallback = 0

    if SELFCHECK and nm_fallback == 0:
        got = float(np.linalg.norm(update_grad))
        assert abs(got - anchor_norm) <= 1e-9 * max(anchor_norm, EPS), \
            "pcgrad_nm norm mismatch: {0} vs {1}".format(got, anchor_norm)

    return {
        "update_grad": update_grad,
        "alpha": 1.0,
        "s_dir": 0.0,
        "probe": 0.0,
        "probe_score": 0.0,
        "probe_eval_count": 0,
        "pc_norm_ratio": float(ratio),
        "nm_fallback": int(nm_fallback),
    }


def build_lso_grid_update(params, n, depth, H_mat, grad_terms, lr, lso_cfg):
    # LSO on a fixed candidate grid: probes u_van + lambda * delta_capped
    # at lambda in GRID_LAMBDAS and picks the lowest probed energy.
    # Exactly len(GRID_LAMBDAS) probe evaluations; ties resolved toward
    # the smaller lambda by strict-inequality argmin in ascending order.
    g_anchor = np.sum(grad_terms, axis=1)
    anchor_norm = float(np.linalg.norm(g_anchor))

    if anchor_norm <= EPS:
        # All capped displacements vanish; both matched-control variants
        # terminate identically with zero probes and a zero update.
        return {
            "update_grad": np.zeros_like(g_anchor),
            "alpha": 0.0,
            "s_dir": 0.0,
            "probe": 0.0,
            "probe_score": 0.0,
            "probe_eval_count": 0,
        }

    task_grads = [grad_terms[:, t].copy() for t in range(grad_terms.shape[1])]
    g_pc = pcgrad_project(task_grads)

    delta = g_pc - g_anchor
    delta = clip_delta(delta, anchor_norm, lso_cfg["delta_cap_ratio"])
    delta_norm = float(np.linalg.norm(delta))

    if delta_norm > EPS:
        s_dir = float(np.dot(g_anchor, delta) / (anchor_norm * delta_norm))
    else:
        s_dir = 0.0

    energies = []
    for lam in GRID_LAMBDAS:
        val = probe_energy(params, n, depth, H_mat, g_anchor, delta, lr, lam)
        energies.append(float(val))

    best_idx = 0
    for i in range(1, len(GRID_LAMBDAS)):
        if energies[i] < energies[best_idx]:
            best_idx = i

    if SELFCHECK:
        assert energies[best_idx] <= min(energies) + 1e-15, \
            "lso_grid argmin violation"

    best_lambda = float(GRID_LAMBDAS[best_idx])
    best_energy = energies[best_idx]
    f0 = energies[0]

    update_grad = g_anchor + best_lambda * delta

    probe_delta = float(best_energy - f0)
    probe_score = float((f0 - best_energy) / max(lr * max(anchor_norm * anchor_norm, EPS), EPS))

    return {
        "update_grad": update_grad,
        "alpha": best_lambda,
        "s_dir": float(s_dir),
        "probe": float(probe_delta),
        "probe_score": float(probe_score),
        "probe_eval_count": int(len(GRID_LAMBDAS)),
    }


def build_vanilla_ls_update(params, n, depth, H_mat, grad_terms, lr, lso_cfg):
    # Norm-matched vanilla line-search control: for each grid lambda,
    # probe s(lambda) * u_van with
    #   s(lambda) = ||u_van + lambda * delta_capped|| / ||u_van||,
    # so that each candidate has exactly the same norm as the
    # corresponding lso_grid candidate. The candidate set differs from
    # lso_grid only in direction. Exactly len(GRID_LAMBDAS) probe
    # evaluations; ties resolved toward the smaller lambda.
    g_anchor = np.sum(grad_terms, axis=1)
    anchor_norm = float(np.linalg.norm(g_anchor))

    if anchor_norm <= EPS:
        # All capped displacements are zero as well; use the vanilla
        # zero update and mark the step as degenerate.
        return {
            "update_grad": np.zeros_like(g_anchor),
            "alpha": 0.0,
            "s_dir": 0.0,
            "probe": 0.0,
            "probe_score": 0.0,
            "probe_eval_count": 0,
            "selected_s": 0.0,
        }

    task_grads = [grad_terms[:, t].copy() for t in range(grad_terms.shape[1])]
    g_pc = pcgrad_project(task_grads)

    delta = g_pc - g_anchor
    delta = clip_delta(delta, anchor_norm, lso_cfg["delta_cap_ratio"])
    delta_norm = float(np.linalg.norm(delta))

    if delta_norm > EPS:
        s_dir = float(np.dot(g_anchor, delta) / (anchor_norm * delta_norm))
    else:
        s_dir = 0.0

    energies = []
    s_values = []
    for lam in GRID_LAMBDAS:
        u_ref = g_anchor + lam * delta
        ref_norm = float(np.linalg.norm(u_ref))
        s = ref_norm / anchor_norm
        cand = s * g_anchor

        if SELFCHECK:
            got = float(np.linalg.norm(cand))
            assert abs(got - ref_norm) <= 1e-9 * max(ref_norm, EPS), \
                "vanilla_ls_nm candidate norm mismatch: {0} vs {1}".format(got, ref_norm)

        val = probe_energy_dir(params, n, depth, H_mat, cand, lr)
        energies.append(float(val))
        s_values.append(float(s))

    best_idx = 0
    for i in range(1, len(GRID_LAMBDAS)):
        if energies[i] < energies[best_idx]:
            best_idx = i

    if SELFCHECK:
        assert energies[best_idx] <= min(energies) + 1e-15, \
            "vanilla_ls_nm argmin violation"

    best_lambda = float(GRID_LAMBDAS[best_idx])
    best_s = s_values[best_idx]
    best_energy = energies[best_idx]
    f0 = energies[0]

    update_grad = best_s * g_anchor

    probe_delta = float(best_energy - f0)
    probe_score = float((f0 - best_energy) / max(lr * max(anchor_norm * anchor_norm, EPS), EPS))

    return {
        "update_grad": update_grad,
        "alpha": best_lambda,
        "s_dir": float(s_dir),
        "probe": float(probe_delta),
        "probe_score": float(probe_score),
        "probe_eval_count": int(len(GRID_LAMBDAS)),
        "selected_s": best_s,
    }


def build_update_gradient(params, n, depth, H_mat, grad_terms, variant, lr, lso_cfg):
    task_grads = [grad_terms[:, t].copy() for t in range(grad_terms.shape[1])]
    g_anchor = np.sum(grad_terms, axis=1)

    if variant == "vanilla":
        return {
            "update_grad": g_anchor,
            "alpha": 0.0,
            "s_dir": 0.0,
            "probe": 0.0,
            "probe_score": 0.0,
        }

    if variant == "pcgrad":
        u_pc = pcgrad_project(task_grads)
        anchor_norm = float(np.linalg.norm(g_anchor))
        pc_norm = float(np.linalg.norm(u_pc))
        pc_ratio = pc_norm / anchor_norm if anchor_norm > EPS else 0.0
        return {
            "update_grad": u_pc,
            "alpha": 1.0,
            "s_dir": 0.0,
            "probe": 0.0,
            "probe_score": 0.0,
            "pc_norm_ratio": float(pc_ratio),
        }

    if variant == "pcgrad_nm":
        return build_pcgrad_nm_update(grad_terms)

    if variant == "lso_grid":
        return build_lso_grid_update(
            params=params,
            n=n,
            depth=depth,
            H_mat=H_mat,
            grad_terms=grad_terms,
            lr=lr,
            lso_cfg=lso_cfg,
        )

    if variant == "vanilla_ls_nm":
        return build_vanilla_ls_update(
            params=params,
            n=n,
            depth=depth,
            H_mat=H_mat,
            grad_terms=grad_terms,
            lr=lr,
            lso_cfg=lso_cfg,
        )

    if variant == "lso_pcgrad":
        return build_lso_update(
            params=params,
            n=n,
            depth=depth,
            H_mat=H_mat,
            grad_terms=grad_terms,
            lr=lr,
            lso_cfg=lso_cfg,
        )

    raise ValueError("Unknown variant: " + str(variant))


def run_single(n, depth, variant, seed, steps, lr, H_mat, term_mats, lso_cfg):
    rng = np.random.RandomState(seed)
    npar = n_params_for_hea(n, depth)
    params = rng.uniform(RNG_INIT_LOW, RNG_INIT_HIGH, size=npar).astype(np.float64)

    rows = []

    for step in range(steps + 1):
        e = energy_sv(params, n, depth, H_mat)
        grad_terms = grad_termwise_sv(params, n, depth, term_mats)
        diag = diagnostics_from_grad_terms(grad_terms)

        update_info = build_update_gradient(
            params=params,
            n=n,
            depth=depth,
            H_mat=H_mat,
            grad_terms=grad_terms,
            variant=variant,
            lr=lr,
            lso_cfg=lso_cfg,
        )

        step_diag = step_usefulness_from_update(
            grad_terms, update_info["update_grad"], lr
        )

        # Common v3 metadata, identical schema for every variant.
        # Note: at the final recorded state (step == steps) the update
        # direction is computed but never applied, so update_norm and
        # step_norm at that step describe a hypothetical update;
        # transition-level analyses must restrict to t = 0..steps-1.
        u = update_info["update_grad"]
        update_norm = float(np.linalg.norm(u))
        anchor_norm = float(np.linalg.norm(diag["grad_full"]))

        if anchor_norm > EPS and update_norm > EPS:
            cos_to_vanilla = float(
                np.dot(diag["grad_full"], u) / (anchor_norm * update_norm)
            )
        else:
            cos_to_vanilla = 0.0

        default_norm_ratio = (
            float(update_norm / anchor_norm) if anchor_norm > EPS else 0.0
        )

        if SELFCHECK:
            lhs = float(np.dot(diag["grad_full"], u))
            rhs = float(step_diag["U_step"] * np.sqrt(max(step_diag["Q_step"], 0.0)))
            tol = 1e-9 * max(1.0, abs(lhs))
            assert abs(lhs - rhs) <= tol, \
                "bridge identity violation: {0} vs {1}".format(lhs, rhs)
            assert abs(step_diag["pred_decrease_lin"] - lr * lhs) <= tol, \
                "pred_decrease mismatch"

        rows.append({
            "n_qubits": n,
            "depth": depth,
            "variant": variant,
            "seed": seed,
            "step": step,
            "energy": e,
            "R_mean": diag["R_mean"],
            "N_eff_mean": diag["N_eff_mean"],
            "B_eff_mean": diag["B_eff_mean"],
            "Q_mean": diag["Q_mean"],
            "grad_norm_l2": diag["grad_norm_l2"],
            "var_bridge_actual": diag["var_bridge_actual"],
            "var_bridge_ratio": diag["var_bridge_ratio"],
            "alpha": update_info["alpha"],
            "s_dir": update_info["s_dir"],
            "probe": update_info["probe"],
            "probe_score": update_info["probe_score"],
            "S_step": step_diag["S_step"],
            "N_eff_step": step_diag["N_eff_step"],
            "U_step": step_diag["U_step"],
            "Q_step": step_diag["Q_step"],
            "dir_sum_step": step_diag["dir_sum_step"],
            "pred_delta_lin": step_diag["pred_delta_lin"],
            "pred_decrease_lin": step_diag["pred_decrease_lin"],
            "update_norm": update_norm,
            "step_norm": float(lr * update_norm),
            "norm_ratio": default_norm_ratio,
            "pc_norm_ratio": float(update_info.get("pc_norm_ratio", 0.0)),
            "selected_s": float(update_info.get("selected_s", 0.0)),
            "probe_eval_count": int(update_info.get("probe_eval_count", 0)),
            "nm_fallback": int(update_info.get("nm_fallback", 0)),
            "degenerate_anchor": int(anchor_norm <= EPS),
            "cos_to_vanilla": cos_to_vanilla,
        })

        if step == steps:
            break

        params = wrap_angles(params - lr * update_info["update_grad"])

    return rows


def save_config(outdir, cfg):
    path = os.path.join(outdir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("Saved:", path)


def assert_resume_config_compatible(outdir, current_cfg):
    path = os.path.join(outdir, "config.json")
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        old_cfg = json.load(f)

    for key in RESUME_CONFIG_KEYS:
        if key not in old_cfg:
            raise ValueError("Existing config is missing key: " + key)
        if key not in current_cfg:
            raise ValueError("Current config is missing key: " + key)
        if old_cfg[key] != current_cfg[key]:
            raise ValueError(
                "Resume config mismatch for key '{0}': old={1}, current={2}".format(
                    key, old_cfg[key], current_cfg[key]
                )
            )

    print("Resume config check passed:", path)


def write_trajectory_csv_atomic(outdir, rows):
    path = os.path.join(outdir, "trajectory_per_seed.csv")
    tmp_path = path + ".tmp"

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJECTORY_KEYS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)
    print("Wrote clean trajectory:", path, "rows=", len(rows))


def append_trajectory_csv(outdir, rows):
    path = os.path.join(outdir, "trajectory_per_seed.csv")
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJECTORY_KEYS)
        if need_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def cast_loaded_trajectory_row(row):
    out = dict(row)

    for k in INT_ROW_KEYS:
        if k in out and out[k] != "":
            out[k] = int(out[k])

    for k in FLOAT_ROW_KEYS:
        if k in out and out[k] != "":
            out[k] = float(out[k])

    return out


def job_key_from_row(row):
    return (
        int(row["n_qubits"]),
        int(row["depth"]),
        str(row["variant"]),
        int(row["seed"]),
    )


def load_existing_trajectory(outdir, final_step):
    path = os.path.join(outdir, "trajectory_per_seed.csv")
    if not os.path.exists(path):
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        header_line = f.readline().strip()
    header = [h.strip() for h in header_line.split(",")] if header_line else []
    if header and header != TRAJECTORY_KEYS:
        raise ValueError(
            "Existing trajectory header does not match the current "
            "schema (version {0}); refusing to resume into a mixed-"
            "schema file: {1}".format(SCHEMA_VERSION, path)
        )

    raw_rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(cast_loaded_trajectory_row(row))

    grouped = {}
    for row in raw_rows:
        grouped.setdefault(job_key_from_row(row), []).append(row)

    clean_rows = []
    dropped_jobs = 0
    dropped_rows = 0
    deduped_rows = 0
    required_steps = set(range(int(final_step) + 1))

    for job_key in sorted(grouped.keys()):
        rows = grouped[job_key]

        by_step = {}
        for row in rows:
            step = int(row["step"])
            if 0 <= step <= int(final_step):
                by_step[step] = row

        if not required_steps.issubset(set(by_step.keys())):
            dropped_jobs += 1
            dropped_rows += len(rows)
            continue

        clean_job_rows = [by_step[s] for s in sorted(required_steps)]
        clean_rows.extend(clean_job_rows)

        if len(rows) != len(clean_job_rows):
            deduped_rows += len(rows) - len(clean_job_rows)

    changed = (len(clean_rows) != len(raw_rows))

    print(
        "Loaded existing trajectory:",
        path,
        "raw_rows=", len(raw_rows),
        "clean_rows=", len(clean_rows),
        "dropped_jobs=", dropped_jobs,
        "dropped_rows=", dropped_rows,
        "deduped_rows=", deduped_rows,
    )

    if changed:
        write_trajectory_csv_atomic(outdir, clean_rows)

    return clean_rows


def completed_job_keys_from_rows(rows, final_step):
    done = set()
    for row in rows:
        if int(row["step"]) == int(final_step):
            done.add(job_key_from_row(row))
    return done


def build_final_compare(all_rows, variants, final_step):
    grouped = {}
    for row in all_rows:
        if row["step"] != final_step:
            continue
        key = (row["n_qubits"], row["depth"], row["variant"])
        grouped.setdefault(key, []).append(row)

    compare_rows = []
    by_nd = {}
    for (n, d, variant), rows in grouped.items():
        by_nd.setdefault((n, d), {})[variant] = rows

    for (n, d), vd in sorted(by_nd.items()):
        if "vanilla" not in vd:
            continue
        a = vd["vanilla"]

        for variant_b in variants:
            if variant_b == "vanilla":
                continue
            if variant_b not in vd:
                continue

            b = vd[variant_b]

            compare_rows.append({
                "n_qubits": n,
                "depth": d,
                "variant_a": "vanilla",
                "variant_b": variant_b,
                "energy_mean_a": float(np.mean([x["energy"] for x in a])),
                "energy_mean_b": float(np.mean([x["energy"] for x in b])),
                "R_mean_a": float(np.mean([x["R_mean"] for x in a])),
                "R_mean_b": float(np.mean([x["R_mean"] for x in b])),
                "N_eff_mean_a": float(np.mean([x["N_eff_mean"] for x in a])),
                "N_eff_mean_b": float(np.mean([x["N_eff_mean"] for x in b])),
                "B_eff_mean_a": float(np.mean([x["B_eff_mean"] for x in a])),
                "B_eff_mean_b": float(np.mean([x["B_eff_mean"] for x in b])),
                "Q_mean_a": float(np.mean([x["Q_mean"] for x in a])),
                "Q_mean_b": float(np.mean([x["Q_mean"] for x in b])),
                "VB_mean_a": float(np.mean([x["var_bridge_actual"] for x in a])),
                "VB_mean_b": float(np.mean([x["var_bridge_actual"] for x in b])),
                "alpha_mean_a": float(np.mean([x["alpha"] for x in a])),
                "alpha_mean_b": float(np.mean([x["alpha"] for x in b])),
                "s_dir_mean_b": float(np.mean([x["s_dir"] for x in b])),
                "U_step_mean_a": float(np.mean([x["U_step"] for x in a])),
                "U_step_mean_b": float(np.mean([x["U_step"] for x in b])),
                "delta_U_step": float(np.mean([x["U_step"] for x in b]) - np.mean([x["U_step"] for x in a])),
                "probe_score_mean_b": float(np.mean([x["probe_score"] for x in b])),
                "delta_energy": float(np.mean([x["energy"] for x in b]) - np.mean([x["energy"] for x in a])),
                "delta_B_eff": float(np.mean([x["B_eff_mean"] for x in b]) - np.mean([x["B_eff_mean"] for x in a])),
                "delta_Q": float(np.mean([x["Q_mean"] for x in b]) - np.mean([x["Q_mean"] for x in a])),
                "delta_VB": float(np.mean([x["var_bridge_actual"] for x in b]) - np.mean([x["var_bridge_actual"] for x in a])),
            })

    return compare_rows


def save_final_compare_csv(outdir, rows):
    path = os.path.join(outdir, "final_compare.csv")
    keys = [
        "n_qubits", "depth", "variant_a", "variant_b",
        "energy_mean_a", "energy_mean_b",
        "R_mean_a", "R_mean_b",
        "N_eff_mean_a", "N_eff_mean_b",
        "B_eff_mean_a", "B_eff_mean_b",
        "Q_mean_a", "Q_mean_b",
        "VB_mean_a", "VB_mean_b",
        "alpha_mean_a", "alpha_mean_b",
        "s_dir_mean_b", "probe_score_mean_b",
        "U_step_mean_a", "U_step_mean_b", "delta_U_step",
        "delta_energy", "delta_B_eff", "delta_Q", "delta_VB",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("Saved:", path)


def build_summary(all_rows, final_compare):
    return {
        "final_compare": final_compare,
        "n_total_rows": len(all_rows),
    }


def save_summary(outdir, summary):
    path = os.path.join(outdir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", path)


def plot_metric_vs_step(outdir, all_rows, metric_key, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    grouped = {}
    for row in all_rows:
        key = (row["n_qubits"], row["depth"], row["variant"])
        grouped.setdefault(key, {}).setdefault(row["step"], []).append(row[metric_key])

    for key in sorted(grouped.keys()):
        n, d, variant = key
        steps = sorted(grouped[key].keys())
        vals = [float(np.mean(grouped[key][s])) for s in steps]
        ax.plot(steps, vals, marker=None, label="n={0}, d={1}, {2}".format(n, d, variant))

    ax.set_xlabel("Optimization step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    path = os.path.join(outdir, "figures", filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser(
        description="LSO-PCGrad experiment on TFIM HEA."
    )
    parser.add_argument("--n_qubits", type=str, default=DEFAULT_N_QUBITS)
    parser.add_argument("--depths", type=str, default=DEFAULT_DEPTHS)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    parser.add_argument("--outdir_root", type=str, default=DEFAULT_OUTDIR_ROOT)

    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_mid_frac", type=float, default=0.5)
    parser.add_argument("--extra_probe_frac", type=float, default=0.25)
    parser.add_argument("--delta_cap_ratio", type=float, default=1.0)

    parser.add_argument(
        "--outdir", type=str, default="",
        help="Existing run directory to resume; config must match."
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="Enable runtime invariant assertions (norm matching, "
             "argmin correctness, bridge identity)."
    )

    args = parser.parse_args()

    global SELFCHECK
    SELFCHECK = bool(args.selfcheck)

    n_qubits_list = parse_int_list(args.n_qubits)
    depths = parse_int_list(args.depths)
    seeds = int(args.seeds)
    steps = int(args.steps)
    lr = float(args.lr)
    variants = parse_str_list(args.variants)

    lso_cfg = {
        "alpha_max": float(args.alpha_max),
        "alpha_mid_frac": float(args.alpha_mid_frac),
        "extra_probe_frac": float(args.extra_probe_frac),
        "delta_cap_ratio": float(args.delta_cap_ratio),
    }

    grid_variants = set(["lso_grid", "vanilla_ls_nm"]) & set(variants)
    if grid_variants and abs(lso_cfg["alpha_max"] - 1.0) > 1e-12:
        raise ValueError(
            "Variants {0} use the fixed grid {1}, which assumes "
            "alpha_max = 1.0; got alpha_max = {2}.".format(
                sorted(grid_variants), GRID_LAMBDAS, lso_cfg["alpha_max"]
            )
        )

    cfg = {
        "schema_version": SCHEMA_VERSION,
        "n_qubits": n_qubits_list,
        "depths": depths,
        "seeds": seeds,
        "steps": steps,
        "lr": lr,
        "outdir_root": args.outdir_root,
        "hamiltonian": "TFIM_OBC",
        "h_field": H_FIELD,
        "variants": variants,
        "optimizer_rule": "six-variant ablation: vanilla, pcgrad, "
                          "pcgrad_nm (norm-matched), lso_pcgrad, "
                          "lso_grid and vanilla_ls_nm (matched budget)",
        "lso_cfg": lso_cfg,
        "grid_lambdas": GRID_LAMBDAS,
        "nm_ratio_min": NM_RATIO_MIN,
        "selfcheck": SELFCHECK,
    }

    resume = bool(args.outdir)
    if resume:
        outdir = args.outdir
        if not os.path.isdir(outdir):
            raise ValueError("Resume outdir does not exist: " + outdir)
        assert_resume_config_compatible(outdir, cfg)
    else:
        outdir = make_timestamped_outdir(args.outdir_root)

    cfg["timestamp_outdir"] = outdir
    save_config(outdir, cfg)

    t0 = time.time()

    all_rows = []
    if resume:
        all_rows = load_existing_trajectory(outdir, steps)
    completed_jobs = completed_job_keys_from_rows(all_rows, steps)
    if resume:
        print("Completed jobs loaded:", len(completed_jobs))

    h_cache = {}
    term_cache = {}

    total_jobs = len(n_qubits_list) * len(depths) * len(variants) * seeds
    done = len(completed_jobs)

    for n, depth, variant, seed in product(n_qubits_list, depths, variants, range(seeds)):
        job_key = (int(n), int(depth), str(variant), int(seed))
        if job_key in completed_jobs:
            continue

        if n not in h_cache:
            h_cache[n] = build_h_obc_matrix(n, h=H_FIELD)
            term_cache[n] = build_h_term_matrices(n, h=H_FIELD)

        rows = run_single(
            n=n,
            depth=depth,
            variant=variant,
            seed=seed,
            steps=steps,
            lr=lr,
            H_mat=h_cache[n],
            term_mats=term_cache[n],
            lso_cfg=lso_cfg,
        )

        all_rows.extend(rows)
        append_trajectory_csv(outdir, rows)
        completed_jobs.add(job_key)

        final_row = rows[-1]
        done += 1
        print(
            "[{0}/{1}] n={2} d={3} variant={4} seed={5} "
            "E={6:.6f} R={7:.4f} N_eff={8:.4f} B_eff={9:.4f} "
            "Q={10:.4e} alpha={11:.4f} s_dir={12:.4f} probe={13:.4e}".format(
                done,
                total_jobs,
                n,
                depth,
                variant,
                seed,
                final_row["energy"],
                final_row["R_mean"],
                final_row["N_eff_mean"],
                final_row["B_eff_mean"],
                final_row["Q_mean"],
                final_row["alpha"],
                final_row["s_dir"],
                final_row["probe"],
            )
        )

    write_trajectory_csv_atomic(outdir, all_rows)

    final_compare = build_final_compare(all_rows, variants, steps)
    summary = build_summary(all_rows, final_compare)

    save_final_compare_csv(outdir, final_compare)
    save_summary(outdir, summary)

    plot_metric_vs_step(
        outdir, all_rows, "energy", "Energy",
        "energy_vs_step.pdf", "Energy vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "B_eff_mean", "B_eff_mean",
        "B_eff_vs_step.pdf", "B_eff vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "Q_mean", "Q_mean",
        "Q_vs_step.pdf", "Q vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "var_bridge_actual", "E[B_eff^2 Q]",
        "var_bridge_vs_step.pdf", "Variance-bridge quantity vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "grad_norm_l2", "Gradient L2 norm",
        "grad_norm_vs_step.pdf", "Gradient norm vs optimization step"
    )
    alpha_rows = [
        r for r in all_rows
        if r["variant"] in ("lso_pcgrad", "lso_grid", "vanilla_ls_nm")
    ]
    plot_metric_vs_step(
        outdir, alpha_rows, "alpha", "alpha",
        "alpha_vs_step.pdf",
        "Selected path parameter alpha vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "s_dir", "s_dir",
        "s_dir_vs_step.pdf", "Directional score vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "probe_score", "probe_score",
        "probe_score_vs_step.pdf", "Probe score vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "U_step", "U_step",
        "U_step_vs_step.pdf", "U_step vs optimization step"
    )
    plot_metric_vs_step(
        outdir, all_rows, "update_norm", "Update norm",
        "update_norm_vs_step.pdf", "Update-direction norm vs optimization step"
    )

    dt = time.time() - t0
    print("=" * 72)
    print("DONE: {0:.1f}s ({1:.2f}h)".format(dt, dt / 3600.0))
    print("Results in: {0}".format(outdir))
    print("=" * 72)


if __name__ == "__main__":
    main()
