# Exact-estimator paired bootstrap — self-contained local bundle

This closes the last statistical gap in the revision: **point estimates and
confidence intervals now come from one and the same estimator.**

## The problem it fixes

The main revision grid (2576 cells) reported point estimates from the pipeline's
primary quality-matching rule, but computed its percentile intervals inside the
bootstrap with a *vectorised caliper-band ranking statistic*. Those are different
estimators, so the intervals were not intervals for the reported numbers. A
reviewer is entitled to ask why. This bundle re-runs the primary rule inside
every replicate so the question cannot be asked.

## What the primary rule actually is

Reproduced term for term from `revq.matching._greedy_wo`, and verified
bit-identical against `match_pairs(... strategy="nn_wo", eps_sd=0.10)` on a full
run:

| property | value |
|---|---|
| pair criterion | `\|Q_noisy - Q_clean\| <= 0.10 * SD(Q)`, caliper recomputed per replicate |
| clean side | **1:1** — a clean sample is consumed by the first noisy sample that claims it |
| noisy side | up to **5** clean partners per noisy sample, taken in ascending |dQ| order |
| order | greedy, ascending sample-id over the noisy samples |
| candidate window | the 512 nearest clean candidates per noisy sample |
| unmatched | a noisy sample with no unused clean candidate in the caliper contributes no pairs |

Two consequences worth stating in the manuscript, because they are counter-intuitive:

1. 8,934 noisy samples yield **36,048** pairs, drawn from 36,048 *distinct* clean
   samples — not 8,934 pairs, and not 5× reuse of each clean sample.
2. Only **7,249** of the noisy samples are matched at all. Early noisy samples
   exhaust the clean pool inside the narrow caliper band, so coverage is ~81%,
   not 100%.

**Verified equivalence.** On CIFAR-100 S20 seed 0, DINO-density conditioning:

```
reference  match_pairs(..., eps_sd=0.10) : 36,048 pairs, matched AUROC 0.69053762
this script                              : 36,048 pairs, matched AUROC 0.69053762
identical pair sets: True        AUC difference: 0.00e+00
```

Two plausible-looking alternatives were implemented and **rejected on
measurement**, because both are different statistics:

| variant | pairs | matched AUROC |
|---|---|---|
| strict 1:1 optimal assignment | 8,934 | 0.69594 |
| optimal assignment, each clean reused ≤ 5× | 44,666 | 0.69234 |
| **the pipeline's actual rule** | **36,048** | **0.69054** |

## Scope

* dataset: CIFAR-100 symmetric-20%
* seeds: all ten on disk
* detectors: automatically restricted to those with **global AUROC ≥ 0.5** — a
  detector below chance cannot exhibit a "reversal", so it is not a cell the
  reversal claim rests on. On these runs this excludes `forgetting`.
* conditioning variables: the five **label-free** ones —
  `dino_density`, `dino_density_fullpool`, `dino_knndist`, `dino_augcons`, `random`
* B = 2000, paired (one draw drives both the global and the matched statistic)

That is 10 seeds × 7 eligible detectors × 5 covariates = 350 cells, whose
eligibility-filtered subset is the set the manuscript's reversal claim covers.

## Running it

```bash
python exact_bootstrap_c100s20.py                    # full run, B=2000
python exact_bootstrap_c100s20.py --B 30 --workers 4 # ~2 minute smoke test
python exact_bootstrap_c100s20.py --seeds 0 --B 200  # one seed
```

No dependencies beyond `numpy`, `pandas`, `scipy`. `data/` must sit next to the
script (or pass `--data`).

**Runtime — read this before launching.** 165 ms of CPU per replicate, i.e.
5.5 CPU-minutes per cell at B=2000. Total cost is
`cells x B x 165 ms` of *CPU time*, and the host's CPU quota (not the worker
count) sets the wall clock:

| scope | B | CPU time | wall clock @ 1 core |
|---|---|---|---|
| 60 cells (one conditioning variable, 6 eligible detectors, 10 seeds) | 500 | 82 min | ~2.8 h |
| 60 cells | 1000 | 2.8 h | ~5.5 h |
| 60 cells | 2000 | 5.5 h | ~11 h |
| all 350 cells | 2000 | 32 h | ~64 h |

`--workers` does **not** help when the container is CPU-quota limited: this
container allows 0.5 core in total, so extra workers only share that half core.
Use `--B 30 --workers 4` (~2 min) as a smoke test and pick scope to fit the
available wall clock. `out/exact_bootstrap.partial.csv` is flushed after every
seed, so a long run is never all-or-nothing and can be resumed seed by seed with
`--seeds`.

A vectorised rewrite of the greedy pass was attempted and **rejected**: it ran at
68 ms/replicate (2.4x faster) but reproduced only the pair *count*, not the pair
*sets*, because two noisy samples inside one vectorised block can claim the same
clean candidate. Bit-identical output is the requirement, so the loop is kept.

## Outputs (`out/`)

| file | contents |
|---|---|
| `exact_bootstrap.csv` | one row per (detector, seed, quality) |
| `exact_bootstrap.partial.csv` | incremental flush, same schema |
| `summary.json` | eligible-cell counts, interval verdicts, failures |
| `summary_by_quality.csv` | per-covariate roll-up |

Per-cell columns include `auc_global`, `auc_primary`, `delta_point`, `ci_low`,
`ci_high` (interval for the matched AUROC), `delta_ci_low`, `delta_ci_high`
(paired interval for Delta), `ci_entirely_above_chance`,
`ci_entirely_below_chance`, `n_valid_bootstrap`, `n_failed_overlap`,
`coverage_primary`, `coverage_bootstrap_mean`, `n_pairs_primary`,
`n_pairs_bootstrap_mean`, `common_support_lo/hi`, `caliper_point`, and
`eligible_global_above_chance`.

## `data/`

`seed{0..9}.npz`, 4.5 MB each (44 MB total). Each holds the frozen per-run
arrays for one CIFAR-100 S20 run at `sample_id` ascending order:

* `mask` — the evaluation-only corruption indicator `z_i`
* `sig__<name>` — 17 per-sample score vectors: the eight detector scores
  (`combined`, `combined_cl`, `ema_loss`, `confidence`, `aum`, `forgetting`,
  `neighbor`, `confident_learning`) and the quality covariates
  (`knn_agreement`, `proto_margin`, `dino_density`, `dino_density_fullpool`,
  `dino_knndist`, `dino_augcons`, `dino_augcons_sd`, `native_density`, `random`)
* `y_observed`, `sample_ids`

All 45,000 entries per array are the run's training split (the 45k/5k split of
the 50k CIFAR-100 train set), in the same order as the frozen evaluation table.

## Manuscript consequence

If every eligible interval lies entirely above chance, the conservative wording

> no interval-supported reversal was observed

can be replaced by

> all N eligible primary-estimator intervals remained entirely above chance

using the exact N from `summary.json`. Only write that if
`n_ci_entirely_below_chance == 0` and `n_ci_entirely_above_chance == n_eligible_cells`.
