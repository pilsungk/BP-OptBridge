# BP-OptBridge

Code and data for the paper:

> **From Trainability Diagnostics to Optimization Claims: Boundaries and
> Controls in Variational Quantum Optimization**
>
> Pilsung Kang (Dankook University)
>
> arXiv: (to be added)

The paper builds on the parameter-level gradient-suppression diagnostics of
barren plateaus introduced in arXiv:2605.01319
([BP-DI repository](https://github.com/pilsungk/BP-DI)).

## Overview

The paper studies the gap between trainability diagnostics and optimization
outcomes at the level of the individual optimizer step. Treating
coefficient-weighted Hamiltonian-term gradients as task-like components gives
an exact bridge

```
g^T u = U_step * sqrt(Q_step)
```

which decomposes the first-order directional derivative into a signed termwise
organization factor `U_step` and a directional-activity magnitude
`sqrt(Q_step)`. Rewriting that bridge in standard first-order geometry shows
that these two factors are not independent optimization axes, and that at fixed
state and update norm the raw gradient already maximizes first-order descent of
the summed objective.

Six update rules are compared on the transverse-field Ising model with a
hardware-efficient ansatz and a gate-local Hamiltonian variational ansatz. Three
primary rules are vanilla gradient descent, a deterministic PCGrad-style
Hamiltonian-term projection, and probe-gated LSO-PCGrad. Three matched control
rules support the prespecified adjudication: norm-matched PCGrad, a capped grid
variant of LSO-PCGrad, and a pointwise norm-matched vanilla line search.

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

  Both scripts support checkpoint/resume via `--outdir <existing run>` with a
  config-compatibility check.
- `analysis/` -- adjudication and table pipeline
  - `analyze_v3.py`
- `analysis_e1/` -- the prespecified association analysis of Section 6 and the
  two post hoc analyses reported alongside it, with their outputs
- `figures/` -- plotting code for the manuscript figure of Section 5
- `results/` -- configs, raw trajectories, and full analysis outputs for the
  two runs used in the paper
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

Runs are deterministic (fixed seeds, no sampling noise). The full HEA and HVA
grids take roughly 82 and 20 hours of wall-clock time on a desktop CPU. Results
in the paper were generated with Python 3.12.3, numpy 2.3.3 (OpenBLAS 0.3.30),
pandas 2.3.3, matplotlib 3.10.7.

## Section 6: the prespecified association analysis

Section 6 asks whether the resolved step-level coordinates retain incremental
association with realized descent once standard first-order geometry is
controlled. The criteria were fixed before any association was computed, and
estimability was assessed before the association statistics were examined.

```
cd analysis_e1

# Stage 1: estimability precheck, which fixes the analysis route
python e1_precheck_v1_6.py \
    --hea ../results/hea/trajectory_per_seed.csv.gz \
    --hva ../results/hva/trajectory_per_seed.csv.gz \
    --outdir e1_precheck_out_v1_6

# Stage 2: the association analysis itself
python e1_association_v1_1.py \
    --hea ../results/hea/trajectory_per_seed.csv.gz \
    --hva ../results/hva/trajectory_per_seed.csv.gz \
    --outdir e1_association_out_v1_1

# Post hoc, reported separately and changing no conclusion
python e1_posthoc_variation_budget.py \
    --assoc e1_association_out_v1_1/cell_associations.csv \
    --audit e1_association_out_v1_1/log_variance_audit.csv \
    --outdir e1_posthoc_out

python e1_posthoc_higher_order_v1_2.py \
    --hea ../results/hea/trajectory_per_seed.csv.gz \
    --hva ../results/hva/trajectory_per_seed.csv.gz \
    --lr 0.05 --outdir posthoc_higher_order_out
```

Both post hoc scripts sit outside the frozen protocol. They are reported in the
paper as post hoc observations and do not alter the prespecified adjudication.

### Reading `cell_associations.csv`

This file reports a raw association estimate for **every** cell, including
cells that failed the estimability criterion. Only rows with
`cell_estimable = True` enter the adjudication or any downstream aggregation.
The same rule applies to `posthoc_higher_order_cells.csv`. Quoting a value from
a cell that failed the screen reproduces a number the paper does not use.

## Section 5 figure

```
python figures/plot_mismatch_trajectories.py \
    --hea_csv results/hea/trajectory_per_seed.csv.gz \
    --hva_csv results/hva/trajectory_per_seed.csv.gz \
    --n_qubits 10 --depth 8 \
    --out fig_mismatch_n10_d8.pdf
```

The script prints per-rule seed counts, step ranges, and final-step means so the
figure can be cross-checked against Table 3. Any `(n,d)` condition on the grid
can be plotted with `--n_qubits` and `--depth`.

## Output-to-paper mapping

Every number in the paper regenerates from the commands above.

In each `results/<ansatz>/analysis/`:

| File | Paper item |
|---|---|
| `paper_table3.csv` | Table 3, summary of optimization outcomes |
| `q1_summary.csv`, `q1_correlations.csv`, `q1_verdict_by_n.csv`, `q1_verdict.txt` | Table 6 and the Q1 adjudication |
| `q2_paired.csv`, `q2_verdict.txt` | Table 7 and the Q2 adjudication |
| `manuscript_numbers.json`, `schema_audit.txt` | sealed headline values and the pre-committed schema audit |

In `analysis_e1/`:

| File | Paper item |
|---|---|
| `e1_association_out_v1_1/log_variance_audit.csv` | Table 4, log-variance decomposition |
| `e1_association_out_v1_1/cell_associations.csv` | Table 5, prespecified association summary |
| `e1_association_out_v1_1/reproducibility.csv`, `gate_a_verdict.txt` | the adjudication of Section 6.3 and 6.4 |
| `e1_precheck_out_v1_6/` | the estimability precheck that fixed the analysis route |
| `e1_posthoc_out/` | the variation-budget diagnostic of Section 6.6 |
| `posthoc_higher_order_out/` | the higher-order sensitivity analysis of Section 6.6 |

Tables 1 and 2 are definitional and have no data source. Figure 1 is drawn in
TikZ and has no data file. Figure 2 is produced by `figures/`.

The analysis pipeline also writes `paper_table4_panelA.csv`,
`paper_table4_panelB.csv`, `paper_table5.csv`, `paper_table6.csv`,
`paper_table7.csv`, `paper_fig1_data.csv`, `quadrant_by_*.csv`,
`coupling_by_n.csv`, and `t0_scale_by_n.csv`. These belong to analyses that were
removed during the revision of the manuscript and do not correspond to anything
in the submitted paper. They are kept because the pipeline still produces them,
and their filenames refer to an earlier table numbering.

## Implementation notes

- The PCGrad variant is deterministic: Hamiltonian terms are processed in a
  fixed order and each projection is applied against the current
  (already-modified) components, unlike the randomized original of Yu et al.
  (2020). See the Methods section of the paper.
- Task gradients supplied to the projection are coefficient-weighted
  Hamiltonian-term gradients (including the sign of each coefficient).
- LSO-PCGrad caps the displacement `u_pc - u_van` at the norm of the vanilla
  gradient (cap ratio 1.0) before probe-based strength selection.
- The `alpha` column in output CSVs is the projection strength, denoted lambda
  in the paper.
- The estimability screen rejects a seed when the maximum variance inflation
  factor exceeds 10 or the condition number exceeds 30. This is a hard cutoff on
  a computed quantity, so a seed sitting very close to the threshold can change
  sides between numerical environments. Cells in which every seed is estimable
  reproduce exactly; a cell with a borderline seed can differ in the third
  decimal. No adjudicated value in the paper sits near the materiality
  threshold, so this does not affect any verdict.

## License

MIT (see LICENSE).
