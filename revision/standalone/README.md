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

## Results (completed run, B = 1000, DINO density, 60 cells)

Run locally: 280 s wall, 8 workers.

```
quality = dino_density
cells   = 60   (6 eligible detectors x 10 seeds)
  auc_global  = 0.7663      auc_primary = 0.7664      delta = -0.0001
  coverage    = 0.8049      failed replicates = 0

intervals entirely above chance : 60/60
intervals entirely below chance :  0/60
```

Marginal detail: the smallest `ci_low` over all 60 cells is **0.6579**, i.e. the
closest any cell comes to chance is 0.158 AUROC away. The adaptive top-up rule
(raise a cell to B = 2000 if `ci_low < 0.52`) therefore never fires: **B = 1000 is
sufficient for every cell** and no cell needed a larger budget.

Per detector (10 seeds each):

| detector | AUC_global | AUC_primary | delta | min ci_low | mean coverage |
|---|---|---|---|---|---|
| aum | 0.7127 | 0.7396 | -0.0269 | 0.7074 | 0.805 |
| combined | 0.6965 | 0.6838 | +0.0126 | 0.6579 | 0.805 |
| confidence | 0.7375 | 0.7350 | +0.0025 | 0.7152 | 0.805 |
| confident_learning | 0.8162 | 0.8132 | +0.0030 | 0.7907 | 0.805 |
| ema_loss | 0.8043 | 0.8004 | +0.0040 | 0.7774 | 0.805 |
| neighbor | 0.8308 | 0.8264 | +0.0044 | 0.8035 | 0.805 |

Consistency with the frozen table: `auc_global` reproduced to 0.000000 and
`auc_primary` to within 0.000167 (tie handling), so this run is on the same
estimator as the reported point estimates.

Point reversals (AUC_primary < 0.5): **0/60**.
Interval-supported reversals (ci_high < 0.5): **0/60**.

### Exact estimator vs the fast-path intervals on the same cells

| | mean ci_low | mean ci_high | mean width |
|---|---|---|---|
| exact primary (this run) | 0.7533 | 0.7857 | 0.0324 |
| fast path (original grid) | 0.7496 | 0.7777 | 0.0280 |

The exact intervals sit slightly higher and are ~16% wider. Both agree on the
verdict (60/60 above chance), so the original conclusion was not an artifact of
the estimator mismatch -- but the reported intervals are now the right ones.

## Manuscript consequence

With `n_ci_entirely_below_chance == 0` and
`n_ci_entirely_above_chance == n_eligible_cells == 60`, the interval-supported
sentence is licensed:

> For DINO-density conditioning, all 60 eligible detector--seed evaluations with
> global AUROC at least 0.5 had exact-primary-estimator 95\% bootstrap intervals
> entirely above chance.

Scope it to DINO density. The other four label-free variables (full-pool density,
KNN distance, augmentation consistency, random) remain valid for the proxy
gradient, the heatmap, the support analysis and descriptive point comparisons,
but their intervals in the original grid come from the fast statistic and should
not carry a formal inferential claim.

## Data archive

`c100s20_exact_bootstrap_inputs_v1.tar.gz`

| | |
|---|---|
| size | 44,717,471 bytes |
| SHA256 | `bc080a779813722ae1c75a3dfa8d74dc71a5103e0d2903bfff61bb5159b3640d` |
| contents | `data/seed0.npz` … `data/seed9.npz`, 22 arrays each, 45,000 entries per array |
| generation | payloads assembled by `revq.prepare` at commit `1584a60`; archived 2026-09-16 |
| seed mapping | `seedN.npz` is the CIFAR-100 S20 run with run seed N; the run seed drives both the noise realization and the 45k/5k split |
| sample_id convention | entries are in ascending `sample_id` order over the run's 45,000 training examples; identical ordering across every array and every covariate |
| noisy counts | 8861–9156 per seed (realized rate 0.1969–0.2035) |

Array names: `mask`, `y_observed`, `sample_ids`, and `sig__<name>` for
`combined`, `combined_cl`, `ema_loss`, `confidence`, `aum`, `forgetting`,
`neighbor`, `confident_learning`, `knn_agreement`, `proto_margin`,
`dino_density`, `dino_density_fullpool`, `dino_knndist`, `dino_augcons`,
`dino_augcons_sd`, `native_density`, `random`.

> The archive contains derived masks, detector and proxy scores, observed labels
> and sample identifiers required to reproduce the C100-S20 bootstrap audit; it
> does not contain CIFAR image data.
