# BP-OptBridge

Code and data for the paper:

> **Organization versus Scale in Variational Quantum Optimization:
> A Step-Level Diagnostic Framework**
> Pilsung Kang (Dankook University)
> arXiv: (to be added)

This is the companion repository to the destructive-interference
diagnostics of barren plateaus introduced in arXiv:2605.01319
([BP-DI repository](https://github.com/pilsungk/BP-DI)).

## Overview

The paper introduces step-level diagnostics for variational quantum
optimization: an exact bridge `g^T u = U_step * sqrt(Q_step)` that
decomposes the first-order predicted energy decrease into a
norm-invariant organization factor (`U_step`) and a usable scale
factor (`sqrt(Q_step)`), with `Q_step` itself referred to as the
directional activity. Six update rules are compared on TFIM with HEA
and gate-local HVA circuits. Three primary rules are vanilla gradient
descent, a deterministic PCGrad-style Hamiltonian-term projection, and
probe-gated LSO-PCGrad. Three matched control rules support the
pre-specified adjudication reported in the paper: norm-matched PCGrad,
a capped grid variant of LSO-PCGrad, and a pointwise norm-matched
vanilla line search.

Rule names in the CSV outputs map to the paper as follows.

| CSV `variant`   | Paper name                              |
|-----------------|-----------------------------------------|
| `vanilla`       | vanilla gradient descent                |
| `pcgrad`        | (blind) PCGrad-style projection         |
| `lso_pcgrad`    | LSO-PCGrad                              |
| `pcgrad_nm`     | norm-matched PCGrad                     |
| `lso_grid`      | capped grid LSO-PCGrad (grid)           |
| `vanilla_ls_nm` | norm-matched vanilla line search (vls)  |

## Repository layout

- `src/` -- experiment scripts (exact statevector simulation)
  - `bp_lso_pcgrad_tfim_v3.py` (HEA)
  - `bp_lso_pcgrad_tfim_hva_v3.py` (HVA)

  Both scripts support checkpoint/resume via
  `--outdir <existing run>` with a config-compatibility check.
- `analysis/` -- adjudication and table pipeline
  - `analyze_v3.py`
- `results/` -- configs, raw trajectories, and full analysis outputs
  for the two runs used in the paper
  - `results/hea/`, `results/hva/`, each containing `config.json`,
    `trajectory_per_seed.csv.gz`, and `analysis/`

## Reproducing the experiments

```
pip install -r requirements.txt

# HEA
python src/bp_lso_pcgrad_tfim_v3.py \
    --n_qubits 4,6,8,10 --depths 4,6,8 --seeds 30 --steps 100 --lr 0.05

# HVA
python src/bp_lso_pcgrad_tfim_hva_v3.py \
    --n_qubits 4,6,8,10 --depths 4,6,8 --seeds 30 --steps 100 --lr 0.05

# Analysis (one command per ansatz)
python analysis/analyze_v3.py --input <run dir>/trajectory_per_seed.csv.gz \
    --outdir <run dir>/analysis --ansatz hea
python analysis/analyze_v3.py --input <run dir>/trajectory_per_seed.csv.gz \
    --outdir <run dir>/analysis --ansatz hva
```

Runs are deterministic (fixed seeds, no sampling noise). The full HEA
and HVA grids take roughly 82 and 20 hours of wall-clock time on a
desktop CPU. Results in the paper were generated with Python 3.12.3,
numpy 2.3.3 (OpenBLAS 0.3.30), pandas 2.3.3, matplotlib 3.10.7. The
primary-rule trajectories were additionally verified bit-exactly
across two machines.

## Output-to-paper mapping

All numbers reported in the paper regenerate from the analysis command
above. In each `results/<ansatz>/analysis/` directory:

- `paper_table3.csv` .. `paper_table7.csv` -- Tables 3-7
  (Table 4 has two panels, primary rules and control rules)
- `paper_fig1_data.csv` -- Figure 1
- `q1_summary.csv`, `q1_correlations.csv`, `q1_verdict_by_n.csv`,
  `q1_verdict.txt` -- Table 8 and the Q1 adjudication
- `q2_paired.csv`, `q2_verdict.txt` -- Table 9 and the Q2 adjudication
- `quadrant_by_n.csv`, `quadrant_by_nd.csv`,
  `quadrant_by_ndphase.csv` -- quadrant analysis and its
  grouping-granularity sensitivity checks
- `coupling_by_n.csv`, `t0_scale_by_n.csv`,
  `manuscript_numbers.json`, `schema_audit.txt` -- supporting
  diagnostics and the pre-committed schema audit

## Implementation notes

- The PCGrad variant is deterministic: Hamiltonian terms are processed
  in a fixed order and each projection is applied against the current
  (already-modified) components, unlike the randomized original of
  Yu et al. (2020). See the Methods section of the paper.
- Task gradients supplied to the projection are coefficient-weighted
  Hamiltonian-term gradients (including the sign of each coefficient).
- LSO-PCGrad caps the displacement `u_pc - u_van` at the norm of the
  vanilla gradient (cap ratio 1.0) before probe-based strength
  selection.
- The `alpha` column in output CSVs is the projection strength, denoted
  lambda in the paper.
- In the quadrant CSVs, `abs_diff` is the signed difference
  `lowU_highQ - highU_lowQ`, not an absolute value.

## License

MIT (see LICENSE).
