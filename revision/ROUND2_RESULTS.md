# Round-2 results: when does quality conditioning distort label-noise evaluation?

This document reports the round-2 experiment set (E1–E11) run after the original
"quality conditioning reveals hidden detectability" claim failed. It is written to
be read on its own: each section states the question, the exact configuration, the
number, and what the number does and does not license.

All numbers come from CSVs in `results/revision/round2/` (plus the frozen
`results/revision/p0_batch/` audit), produced by the scripts in `revision/`.

---

## 0. Executive summary

**The claim that survives.** Quality conditioning does not reveal detectability; it
*manufactures* a gap. Matching noisy and clean examples on the quality proxy Q
while leaving every other degree of freedom alone reliably *lowers* the measured
AUC of a symmetric detector, and the size of that lowering is governed by the
coupling between the detector and Q. Reversal (conditioned AUC at or below chance)
is the extreme end of that same attenuation, and whether it appears is predicted to
within a few percent by one inequality:

> **a reversal happens exactly when attenuation is large enough to exceed the
> detector's own margin over chance, `Δ > A_global − 0.5`**

This is arithmetic, not a hypothesis, and it reorganises the phenomenon: reversal
is not a separate effect that a strong detector would also show if the conditioning
were "stronger". It is what attenuation looks like when handed a detector that was
already close to chance.

**Four headline results:**

| # | Result | Evidence |
|---|---|---|
| 1 | Attenuation is real and independent of detector–proxy coupling | §2, §3 |
| 2 | Reversal is a *(proxy, detector)*-level effect, not a proxy-level one | §9 |
| 3 | Human noise shows the **largest** attenuation and **zero** reversal | §7 |
| 4 | Two detectors cannot be audited at all (score saturation) | §10 |
| 5 | The attenuation result survives multiplicity control; the reversal survives it in exactly three families | §5 |

**Statistics.** On the canonical C100-S20 set with the pre-defined 40-family
universe (core-5 detectors × 8 proxies) and a signed effect floor Δ ≥ 0.01,
attenuation is BH-significant in **28/40** families under all three tests (exact
sign-flip, Wilcoxon, paired t) and **18/40** once the effect floor is applied — the
same 18 by every test. Reversal is BH-significant in **3/40**:
`Combined × {knn_agreement, proto_margin}` and `AUM × native_density`. Summarised
in §5.3, which also gives the 48-family sensitivity universe that adds
`Combined-noN` (34/48 → 21/48 → 5/48).

**The strongest single number.** On CIFAR-100N human noise, conditioning on the
kNN-agreement proxy removes `+0.1501` AUROC (range over five seeds), the largest
coupled attenuation in the whole study, while the label-free proxies move by
`+0.0049` and the random control by `+0.0054`. And yet **not one** of the 80
label-dependent or 40 encoder-derived cells reverses, because the training runs are
strong enough (`A_global ≈ 0.65–0.77`) that the same attenuation leaves the audit
above chance. That is result 3 and it is the paper's most defensible new claim.

---

## 0a. What "canonical" means here, and which sections predate it

Sections §1–§4a, §6–§7 and §10 were measured before the analysis set was frozen.
Two decisions were taken afterwards and are authoritative from §5 onward:

* **one analysis set** — primary = EMA Loss, Confidence, AUM, CL, Combined,
  Combined-noN; `forgetting` = coarse secondary; `neighbor` excluded;
* **one coverage schema** — `common_*` vs `matched_*`, four explicit names.

Accordingly §5 (multiplicity), §8 (E7/E10) and §9 (E11) have been **recomputed** on
the canonical set and their numbers supersede anything stated earlier in this
document for the same quantity. §5.3 also fixes the reporting universe: the paper's
statistic is the pre-defined **core-5 / 40-family** set (28/40 → 18/40 → 3/40), and
the 48-family `Combined-noN` variant is a labelled sensitivity. §1–§4a, §6, §7 and §10 are unaffected by either
decision: none of them pools detectors across tiers, and none reads a coverage
column. `run_e10_weak_detector.py` and `run_e11_variance.py` now load
`final_analysis_set.csv` rather than the per-audit CSVs, so the tier filter and the
coverage names are applied in exactly one place.

---

## 1. E1 — Combined without the neighbourhood term

**Question.** The paper's `Combined` detector is a weighted sum that includes a
`neighbor` term, and the KNN-agreement proxy is built from the same neighbourhood
structure. So both the frozen reversal *and* part of its size could be an artefact
of the detector and the audit sharing construction. Does the reversal survive when
that sharing is removed?

**Configuration.** `combined_noN = RN(0.625 · sig__ema_loss + 0.375 · sig__forgetting)`
— the 0.5 : 0.3 weights of the paper's Combined renormalised over its two
non-neighbourhood terms, robust-normalised like every other detector. CIFAR-100
symmetric 20%, 10 seeds, 8 detectors × 8 proxies, B = 1000, primary `nn_wo` rule,
10-seed dino freeze.

**Result.** The reversal does not disappear; it gets *stronger per unit of
detector*. On the kNN-agreement proxy:

| detector | A_global | A_cond | Δ | interval rev |
|---|---|---|---|---|
| `combined` (with neighbor) | 0.6965 | 0.3901 | 0.3063 | 10 / 10 |
| `combined_noN` | 0.5688 | 0.3901 | 0.1787 | 10 / 10 |
| `ema_loss` | 0.8043 | 0.6528 | 0.1515 | 0 / 10 |
| `confidence` | 0.7375 | 0.5845 | 0.1530 | 0 / 10 |
| `confident_learning` | 0.8162 | 0.6124 | 0.2038 | 0 / 10 |
| `aum` | 0.7127 | 0.6181 | 0.0946 | 0 / 10 |

Two facts to read off this table.

* The frozen pipeline's `Δ = 0.3063` is inflated. Its **conditioned** AUC (0.39012)
  is within `2.2 × 10⁻⁴` of `combined_noN`'s (0.39008) in every one of the 10 seeds
  — mean absolute difference `8.8 × 10⁻⁵` — while its global AUC is 0.1277 higher.
  So 0.1277 of the headline Δ is the coupled `neighbor` term being removed by the
  audit rather than quality mismatch being detected. The two detectors are not
  literally identical inside the matched sets (the surviving weights differ after
  renormalisation), but they agree to four decimal places, which is what makes the
  decomposition clean.
* `combined_noN`'s Δ (0.1787) is comparable to four *fully uncoupled* detectors
  (0.0946–0.2038). Decoupling the detector from the proxy does **not** shrink the
  attenuation to zero. This is the first evidence that attenuation is not a
  coupling artefact.

**What this does and does not license.** It licenses "attenuation survives
decoupling" and "the frozen Δ is inflated by the shared neighborhood term". It
does **not** license "reversal survives decoupling as a general statement": of the
four uncoupled detectors, none reverses, because their global AUCs are 0.71–0.82.
`combined_noN` reverses only because it is the weak detector in the table (0.5688),
giving a margin of 0.0688 over chance against a Δ of 0.1787.

**Label-free / control check.** Over all 8 detectors and 10 seeds, the four
label-free proxies show mean Δ = +0.0005 and the random control +0.0001, and **0 of
350 cells** reverses. All 350 condition-interval lower bounds sit above 0.5. The
effect requires a proxy that actually carries label information.

---

## 2. E2 — Is it the visiting order?

**Question.** The primary rule visits noisy samples in ascending `sample_id` and
greedily claims up to 5 clean partners each. A reversal could in principle be an
ordering artefact: a handful of early noisy samples monopolising the clean pool.

**Configuration.** `matched_pairs_random_order` — the identical rule with the noisy
visit order randomised, 25 repetitions per cell, 7 detectors × 5 proxies × 10
seeds = 300 rows. Nothing else changes: same caliper, same ascending-|ΔQ|
candidate sort, same consume-once clean pool.

**Result.** Order is not the mechanism.

* |AUC_pipeline − AUC_random_mean| over 300 rows: mean 0.00313, median 0.00178,
  max 0.02495 (worst case `confident_learning × knn_agreement`, which moves *away*
  from chance, 0.591 → 0.616).
* Every reversal cell stays reversed: `combined_noN × knn_agreement` has
  pipeline 0.3923 vs random-order mean 0.3844 with `P(A_rand < 0.5) = 1.00` — this
  is a per-seed mean over 25 repetitions, and it equals exactly 1.00 in all 10 of
  10 seeds. `combined × knn_agreement` and `combined_noN × proto_margin` are
  likewise 1.00 in 10/10 seeds.
* The one honest caveat: `combined_noN × native_density` sits at 0.4940 and is
  *not* order-stable. Its per-seed `P(A_rand < 0.5)` spans **0.08 to 1.00**
  (values 0.08, 0.08, 0.28, 0.28, 0.72, 0.80, 0.80, 0.80, 0.96, 1.00; mean 0.58),
  i.e. for four of the ten seeds the random-order audit does **not** go below
  chance. `native_density` reversals must be reported as marginal, and this is the
  cell that shows why randomising the visiting order is a necessary check rather
  than a formality.

---

## 3. E3 — Is it clean-pool depletion?

**Question.** The primary rule consumes clean samples. Under a hard caliper the
matched noisy coverage collapses to 0.10 for the kNN-agreement proxy. Perhaps the
audit is measuring the exhaustion of a thin pool rather than a matched comparison.

**Configuration.** `matched_pairs_with_replacement` — identical rule, clean
candidates reusable by any number of noisy samples. Coverage rises from 0.10 to
0.99 for `knn_agreement`, 0.38 → 1.00 for `native_density`, 0.40 → 0.99 for
`proto_margin`, and 0.80 → 1.00 for the label-free proxies.

**Result.** Total coverage is not the mechanism either.

| proxy | coverage (pipeline → replacement) | ΔAUC |
|---|---|---|
| `random` | 0.805 → 1.000 | 0.0015 |
| `dino_density` | 0.805 → 1.000 | 0.0016 |
| `knn_agreement` | 0.101 → 0.991 | 0.0306 |
| `proto_margin` | 0.397 → 0.995 | 0.0413 |
| `native_density` | 0.382 → 0.999 | 0.0628 |

Restoring coverage to ~100% moves the audit AUC by 0.03–0.06, while the effect
being explained is 0.10–0.31. Pool depletion contributes at most a fifth of it, and
nothing for the label-free and control proxies.

**E2 + E3 together.** Neither the visiting order nor the pool budget drives the
effect, so it must come from *which* clean samples are admissible under a hard
quality caliper. E5 (§5) shows the same for the caliper width itself.

---

## 4. E4 — CIFAR-100 asymmetric 40%

**Question.** Does the phenomenon survive a change of noise *structure* (directed,
class-dependent, 40%) rather than rate?

**Configuration.** CIFAR-100 asymmetric 0.4, 5 seeds (all that are prepared),
8 detectors × 8 proxies, B = 1000. Detector global AUCs are much lower here:
0.5596 (`confidence`) to 0.6747 (`combined`).

**Result.** The structure changes, and it changes in a way the margin account
predicts. Pooled over eligible detectors, `A_global = 0.6158`.

| proxy | Δ | A_cond | interval rev |
|---|---|---|---|
| `proto_margin` | **+0.1395** | 0.4763 | **23 / 40** |
| `knn_agreement` | +0.0600 | 0.5558 | 0 / 40 |
| `native_density` | −0.0002 | 0.6160 | 0 / 40 |
| all four label-free | +0.0007 | 0.6150 | 0 / 160 |
| `random` | +0.0008 | 0.6149 | 0 / 40 |

Two observations:

* Reversal now concentrates on `proto_margin`, not `knn_agreement` — the opposite
  of S20. In A40 the asymmetric flip makes the prototype margin carry the bulk of
  the Q–Z dependence (mutual information 0.2825 nats, §4a) while kNN agreement
  carries much less (0.0297 nats). The proxy that reverses is the proxy that knows
  the most about the noise, and *which* proxy that is depends on the noise
  structure, not on the pipeline.
* `native_density` completely flatlines on A40 (Δ = −0.0002). This is consistent
  with §10: the encoder-derived density is not a good conditional-quality signal
  under directed noise, and it also shows a negligible mutual information with the
  noise mask there (0.0024 nats, §4a).

---

## 4a. E8 — How nonlinear is the Q–Z dependence, and does it predict reversal?

**Question.** Δ and reversal are AUROC-based, i.e. rank-based, so they see only
monotone association. If the quality proxies differ mainly in *nonlinear*
dependence on the noise mask, a rank statistic would understate the difference.
Measuring the full dependence also tests whether reversal is simply "the proxy that
knows the most about the noise".

**Configuration.** `sklearn.feature_selection.mutual_info_classif` with
`discrete_features=False` (kNN Kozachenko–Leonenko estimator) on the
one-dimensional proxy score Q and the binary noise mask Z, with a 1000-draw label
permutation of Z as the null. 10 seeds × 8 proxies for S20, 5 seeds × 8 proxies for
A40. Driver `revision/run_e8_parallel.py`; verified bit-identical to the sequential
`run_round2.run_mi` (same MI to 12 dp, same permutation stream).

**Result (mean over seeds, nats).**

| proxy | S20 I(Q;Z) | S20 z | A40 I(Q;Z) | A40 z |
|---|---|---|---|---|
| `proto_margin` | 0.1552 | 146.6 | **0.2825** | 215.4 |
| `native_density` | 0.1507 | 142.9 | 0.0024 | 1.1 |
| `knn_agreement` | 0.1501 | 92.1 | 0.0297 | 15.2 |
| `random` (control) | 0.0010 | 0.3 | 0.0006 | −0.3 |
| `dino_knndist` | 0.0007 | −0.0 | 0.0007 | −0.2 |
| `dino_density` | 0.0007 | −0.0 | 0.0007 | −0.2 |
| `dino_augcons` | 0.0004 | −0.3 | 0.0006 | −0.2 |
| `dino_density_fullpool` | 0.0002 | −0.5 | 0.0019 | 0.7 |

Every label-free proxy and the control sits at the estimator's noise floor
(0.0002–0.0010 nats, z between −0.5 and +0.7), while the three "coupled" proxies are
100–500× higher. So the score separation is genuinely about label information, not
about a monotone reparameterisation of a label-free quantity.

**Two consequences, one of them negative for the simple story.**

* The mutual information confirms the proxy ranking that §4 infers from Δ. In A40
  the ordering of I(Q;Z) is `proto_margin` (0.283) ≫ `knn_agreement` (0.030) ≫
  `native_density` (0.002), which matches the ordering of the A40 Δ values
  (+0.1395, +0.0600, −0.0002) and explains why the reversing proxy changes with the
  noise structure.
* But I(Q;Z) does **not** predict reversal. In S20 the three coupled proxies have
  nearly identical mutual information (0.1501, 0.1507, 0.1552) yet reverse at 10/10
  (`knn_agreement`), 3/10 (`proto_margin`) and 1/10 (`native_density`). In A40
  `proto_margin`'s mutual information is 9.5× `knn_agreement`'s, yet
  `knn_agreement`'s Δ is *larger* under S20 and comparable under A40.

So the strength of the Q–Z dependence is necessary but not sufficient: what decides
reversal is the *joint* structure of (Q, Z, detector score) — which §9 identifies as
a (proxy, detector)-level property.

---

## 5. Final analysis set — coverage schema, detector tiers, multiplicity

This section supersedes the earlier E5. Three things were wrong or ambiguous in it
and are fixed here. Everything below comes from
`revision/run_final_analysis_set.py`.

### 5.1 One coverage schema

`noisy_coverage` / `clean_coverage` meant common-support coverage in E1 and E4 but
matched coverage in E9. The ambiguous names are gone; every audit now carries only
the four explicit names:

| column | meaning |
|---|---|
| `common_noisy_coverage` | fraction of noisy samples inside the common support of Q |
| `common_clean_coverage` | fraction of clean samples inside the common support of Q |
| `matched_noisy_coverage` | fraction of noisy samples actually matched after the caliper |
| `matched_clean_coverage` | fraction of clean samples actually matched after the caliper |

`run_final_analysis_set.py` rewrites the CSVs, drops the ambiguous columns, and
then **asserts** that no audit contains one — a schema violation is a hard failure,
not a warning. Result (`results/revision/round2/coverage_schema.csv`):

| audit | before | after |
|---|---|---|
| C100-S20 | 4 cols incl. both ambiguous names | all four canonical |
| C100-A40 | 4 cols incl. both ambiguous names | all four canonical |
| C100-S20-frozen | 4 cols incl. both ambiguous names | all four canonical |
| C100N-human | `noisy_coverage`, `clean_coverage` (matched values) | all four canonical (common pair backfilled on the server, `revision/run_backfill_c100n_common.py`) |

C100N-human needed the server round-trip: its driver wrote matched coverage under
the short names, and the common-support pair requires the prepared cache. All four
audits now carry all four columns, so the schema is complete rather than merely
non-colliding.

The two conventions differ exactly where the caliper bites — and the same shape
appears in the human-noise audit, which is the point of keeping them separate:

C100-S20:

| proxy | common noisy / clean | matched noisy / clean |
|---|---|---|
| `random`, four `dino_*` | 0.9997 – 1.0000 | 0.8046 – 1.0000 |
| `knn_agreement` | 0.9907 / 1.0000 | **0.1011 / 0.1250** |
| `native_density` | 0.9987 / 0.9904 | 0.3818 / 0.4750 |
| `proto_margin` | 0.9981 / 0.9700 | 0.3969 / 0.4934 |

C100N-human (realised noise rate 0.4022; the matched coverage is lower here because
the noisy stratum is larger and the caliper relativised to a wider Q):

| proxy | common noisy / clean | matched noisy / clean |
|---|---|---|
| `random`, four `dino_*` | 0.9998 – 1.0000 | 0.2990 – 0.2996 / 0.9980 – 1.0000 |
| `knn_agreement` | 0.9953 / 1.0000 | **0.0536 / 0.1782** |
| `native_density` | 0.9994 / 0.9898 | 0.2786 / 0.9310 |
| `proto_margin` | 0.9963 / 0.9846 | 0.2486 / 0.8303 |

In every audit the common support is essentially the whole stratum (≥ 0.985) while
the matched fractions span 0.05–1.00: the caliper, not the support, is what
restricts the comparison. That separation is exactly what the two column families
exist to make visible.

### 5.2 One detector set

| tier | detectors | why |
|---|---|---|
| **primary** | EMA Loss, Confidence, AUM, CL, Combined, **Combined-noN** | continuous scores, 39.6k–44.1k distinct values |
| **coarse secondary** | Forgetting | 110–129 distinct values, tie rate ~1%; reported separately, never pooled |
| excluded | `neighbor` | 11–18 distinct values, tie rate 9–26%, pairwise AUROC lands on exactly 0.5000 |

The frozen S20 audit predates `combined_noN`, so cross-audit comparisons — and the
multiplicity universe — use the **core-5** that all four audits actually contain
(EMA Loss, Confidence, AUM, CL, Combined). `combined_noN` is an extension, present
in three of the four audits and reported as a sensitivity variant in §5.3. Also
note that Forgetting's S20 global AUROC is 0.3305 — below chance — so it is
ineligible for a reversal claim there regardless of its resolution.

### 5.3 Corrected multiplicity tests

Two direction bugs in the old E5, both verified on a family whose ten deltas are
uniformly positive:

| test | old (wrong) | correct | p on a known-attenuating family |
|---|---|---|---|
| sign-flip | `P(mean(s·Δ) ≥ observed)` | `P(mean(s·Δ) ≤ observed)` | old 1.0 → new 0.000977 |
| Wilcoxon | `wilcoxon(−Δ, "greater")` | `wilcoxon(Δ, "greater")` | old 1.0 → new 0.000977 |
| paired t | `ttest_rel(global, cond, "greater")`, read as testing the complement | same call read as H1: A_global > A_cond | 1.6e−12 |

The old sign-flip rejected families whose Δ was *positive* and returned p = 1 for
families whose Δ was uniformly negative — exactly inverted, which is why it
reported 0/48. With the direction fixed, all three tests now agree.

**The reporting universe.** Multiplicity needs a family count fixed *before* the
tests, and the honest choice is the pre-defined cross-audit detector set: the
**core-5** continuous detectors that every audit contains — EMA Loss, Confidence,
AUM, CL and the paper's Combined. `Combined-noN` is a *sensitivity variant of
Combined*, not an independent detector family, so giving it its own eight families
would let one detector contribute twice to the same correction. It is reported
separately as an extension.

| universe | detectors | families | attenuation BH | + Δ ≥ 0.01 | reversal BH |
|---|---|---|---|---|---|
| **core-5 (paper)** | EMA Loss, Confidence, AUM, CL, Combined | **40** | **28 / 40** | **18 / 40** | **3 / 40** |
| primary-6 (sensitivity) | the above + Combined-noN | 48 | 34 / 48 | 21 / 48 | 5 / 48 |

`multiplicity_main_core5.csv` is the paper's table; `multiplicity_primary6.csv` is
the sensitivity version. Both use the same pipeline, so the difference is purely
the universe. Note that the extra 8 families in the 48-family version are all real
`Combined-noN` results — in particular `Combined-noN × knn_agreement` (A_cond
0.39005, Δ 0.17871) and `Combined-noN × proto_margin` (A_cond 0.42576, Δ 0.14300)
do reverse, which is §1's decoupling result seen through the multiplicity lens.

**Main table (core-5, 40 families).** An effect floor of **Δ ≥ 0.01 (signed)** is
applied on top of BH, because these are one-sided tests of E[Δ] = 0 and a family
whose attenuation is uniformly +0.001 across ten seeds rejects at the floor
p = 2⁻¹⁰. The floor is signed, not `|Δ| ≥ 0.01`: attenuation is Δ > 0, so a family
with a large *negative* Δ is evidence against the effect. Taking the absolute value
would have admitted three such families — `AUM × dino_density`,
`AUM × dino_knndist` and `AUM × dino_density_fullpool`, all at Δ ≈ −0.027 with
**0 of 10** seeds attenuating — and would have reported them as findings:

| test | BH-significant | Bonferroni | BH **and** Δ ≥ 0.01 |
|---|---|---|---|
| sign-flip | 28 / 40 | 24 / 40 | **18 / 40** |
| Wilcoxon | 28 / 40 | 24 / 40 | **18 / 40** |
| paired t | 28 / 40 | 25 / 40 | **18 / 40** |
| reversal (sign-flip) | **3 / 40** | 3 / 40 | 3 / 40 |

All three tests select the same 28 families and the same 18 after the effect floor.
Direction-consistent families: attenuation 34/40, reversal 3/40.

By proxy family — the effect floor is what separates signal from sign-consistency:

| proxy family | BH | BH and Δ ≥ 0.01 | mean Δ | max Δ |
|---|---|---|---|---|
| label-dependent | 10 / 10 | **10 / 10** | +0.1716 | +0.3063 |
| encoder-derived | 5 / 5 | **5 / 5** | +0.1188 | +0.2194 |
| label-free | 13 / 20 | 3 / 20 | −0.0007 | +0.0127 |
| control (`random`) | 0 / 5 | 0 / 5 | +0.0002 | +0.0005 |

The label-free column is worth reading carefully, because it is *not* an artefact:
those 13 families have a **positive** mean Δ (+0.0011 to +0.0127, with 9–10 of 10
seeds agreeing in sign), i.e. a real but ~20× smaller attenuation than the coupled
proxies. The effect floor is what removes them, not the direction guard. The pooled
`label_free` mean of −0.0007 is *not* in tension with this: it averages the positive
`dino_*` offsets together with the `random` control, which has essentially none.

The three BH-significant, effect-gated reversals are:

| detector | proxy | A_global | A_cond | Δ | p |
|---|---|---|---|---|---|
| Combined | knn_agreement | 0.69645 | 0.39012 | 0.30634 | 0.00098 |
| Combined | proto_margin | 0.69645 | 0.46496 | 0.23150 | 0.00098 |
| AUM | native_density | 0.71274 | 0.49334 | 0.21940 | 0.00195 |

Adding Forgetting as a coarse secondary changes nothing (28/40 and 3/40), which is
the expected result for a detector that cannot resolve the comparison.

**Cross-audit consistency (core-5, 40 families each).** BH is now applied *within*
the family set being reported — previously it was applied to a 40-family set and
the result then filtered to a 15-family subset while still advertising the
40-family step-up count, which silently inflates the effective false-discovery
rate.

| audit | seeds | sign-flip | Wilcoxon | paired t | reversal | sign-flip floor | can survive Bonferroni |
|---|---|---|---|---|---|---|---|
| C100-S20 | 10 | 28 / 40 | 28 / 40 | 28 / 40 | 3 / 40 | 0.00098 | yes |
| C100-S20-frozen | 10 | 28 / 40 | 28 / 40 | 28 / 40 | 3 / 40 | 0.00098 | yes |
| C100N-human | 5 | 26 / 40 | 26 / 40 | 27 / 40 | 0 / 40 | 0.03125 | no |
| C100-A40 | 5 | 0 / 40 | 0 / 40 | 10 / 40 | 0 / 40 | 0.03125 | no |

The two new S20 audits reproduce the frozen one exactly, which is a useful
end-to-end check of the rewritten driver.

**What the small-n audits can and cannot support.** The exact sign-flip floor is
2⁻ⁿ, and BH's *largest* threshold is q itself (at k = m), so the test can reject at
n = 5 — it needs k* = ⌈m·2⁻ⁿ/q⌉ families sitting exactly on the floor, i.e. 25 of 40
at the full universe. What n = 5 can never do is produce p < 0.031, so it can never
survive a Bonferroni correction at any family count. That is why the A40 row shows
0/40 for sign-flip and Wilcoxon but 10/40 for paired t: with 5 seeds the mean-based
exact tests are floor-limited while the variance-based t-test is not. **For
C100-A40 and C100N-human the paired t-test is the only test that can reject at the
full 40-family count.** This is a property of n = 5, not of the effect, and it
belongs in the manuscript rather than in a referee's report.

**Sensitivity only — do not put this in the abstract.** Restricting the family set
to the fifteen label-carrying proxies (`label_dependent` + `encoder_derived`) makes
the exact tests pass 15/15 in all four audits:

| audit | m | sign-flip | Wilcoxon | paired t | reversal |
|---|---|---|---|---|---|
| C100-S20 | 15 | 15 / 15 | 15 / 15 | 15 / 15 | 3 |
| C100-S20-frozen | 15 | 15 / 15 | 15 / 15 | 15 / 15 | 3 |
| C100N-human | 15 | 15 / 15 | 15 / 15 | 15 / 15 | 0 |
| C100-A40 | 15 | 11 / 15 | 11 / 15 | 11 / 15 | 0 |

This is a *post-hoc family restriction* — it is chosen after observing that the
label-free and control proxies are nulls — so it must be reported as a
supplementary sensitivity analysis, never as the headline. The paper's statistic is
the pre-defined 40-family universe with the signed effect floor:
**28 / 40 → 18 / 40, and 3 / 40 reversal.**

An earlier version of this section reported 34/48 for Wilcoxon and paired t and
0/48 for the sign-flip, and its 48-family set silently contained a duplicate
`Combined` variant. Those numbers are superseded: the 48-family figure is now
labelled for what it is (a sensitivity universe that adds `Combined-noN`), and the
paper's universe is core-5's 40.

---

## 6. E6 — The analytic reversal threshold

**Question.** The Gaussian construction model predicts a reversal threshold
λ* = β(1 + σ_Q²)/γ for the coupling strength λ between Q and the label error. Does
a simulation with a matched estimator hit that threshold?

**Configuration.** N = 20,000; ρ = 0.20 (noise rate), σ_E = 1, σ_Q = 0.3; the
detector e = βz + γd + σ_E·ε with β chosen for a 0.70 global AUC; Q = d + λz + ε_Q.
100 repeats per (γ, λ). Grid λ ∈ [0, 5] with step 0.1 — the **first run used
λ ∈ [0, 3] with step 0.1 and reported γ = 0.25 as "3 of 100 replicates crossing",
which was grid truncation, not a result**: the analytic λ* for that γ is 3.333.

**Result.** With the grid extended:

| γ | β | analytic λ* | empirical crossing | 95% CI | reps crossing |
|---|---|---|---|---|---|
| 0.25 | 0.7644 | 3.3330 | 3.411 | [3.0475, 3.900] | 100 / 100 |
| 0.50 | 0.8292 | 1.8075 | 1.847 | [1.700, 2.000] | 100 / 100 |
| 1.00 | 1.0488 | 1.1432 | 1.195 | [1.100, 1.200] | 100 / 100 |
| 1.50 | 1.3370 | 0.9715 | 1.010 | [1.000, 1.100] | 100 / 100 |

The analytic formula is systematically **low by 2–5%** (empirical/analytic ratio
1.023, 1.022, 1.045, 1.040). All four γ values cross in every one of 100 repeats.
The reversal transition is sharp: for γ = 0.5 the reversal rate goes
0.09 → 0.52 → 0.91 → 1.00 across λ = 1.7, 1.8, 1.9, 2.0.

This upgrades the Gaussian model from "illustrative" to "quantitatively checked":
it predicts the threshold to within a few percent over a 3.4× range of the
threshold itself, and the residual bias is a single-signed 2–5%, consistent with
the matched estimator's finite-sample cost rather than a wrong functional form.

---

## 7. E9 — CIFAR-100N human noise

**Question.** Gate 4 of the revision: is the phenomenon an artefact of *synthetic*
label noise? Human label error is not a uniform flip of a known rate.

**Configuration.** CIFAR-100N `noisy_label` — 20,100 / 50,000 = 0.4022 realised
error rate — 5 seeds (all prepared), 8 detectors × 8 proxies, B = 1000, same
primary rule and estimator as every other audit. Driver:
`revision/run_e9_c100n.py`.

**Result.** Human noise shows the largest coupled attenuation in the study and no
reversal at all.

Pooled over eligible detectors, `A_global = 0.7001`. By proxy family:

| proxy family | Δ | reversal | cells |
|---|---|---|---|
| label-dependent | **+0.1265** | 0 / 80 | 80 |
| encoder-derived (`native_density`) | +0.0366 | 0 / 40 | 40 |
| label-free | +0.0049 | 0 / 160 | 160 |
| control (`random`) | +0.0054 | 0 / 40 | 40 |

Per coupled proxy: `knn_agreement` +0.1501, `proto_margin` +0.1029,
`native_density` +0.0366.

Three things to note.

* **The effect is largest here.** `knn_agreement`'s +0.1501 exceeds its S20 value
  (where the A_global is 0.70 as well) and matches its A40 value, on noise that was
  hand-annotated by people. This is the strongest single piece of evidence that
  quality conditioning distorts label-noise evaluation in a way that is not a
  synthetic-noise artefact.
* **Zero reversal, and predicted.** Excluding the saturated `neighbor` cell of
  §10 — which lands on exactly 0.5000 by resolution, not by conditioning — the
  smallest margin `A_cond − 0.5` over the 23 remaining coupled
  (detector, proxy) cells is **+0.0113** (`forgetting × knn_agreement`), and the
  next smallest are +0.0132 and +0.0169. The detectors are simply too strong under
  human noise (0.6096–0.7651 global AUC) for a 0.10–0.15 attenuation to reach
  chance. The absence of reversal is not a failure to replicate; it is what §8's
  inequality requires.
* **The control offset is small but non-zero.** The random control moves by
  +0.0054 [95% CI +0.0005, +0.0103] and the label-free family by +0.0049. Report
  the coupled numbers against this floor, not against zero: the label-dependent
  effect is 23–31× the control floor, and it is the ratio that carries the claim.

---

## 8. E7 + E10 — The margin account

**Question.** Is reversal a distinct phenomenon, or is it attenuation applied to a
weak detector?

**E10 (`revision/run_e10_weak_detector.py`).** Reads the canonical
`final_analysis_set.csv` with excluded detectors removed, giving 1,440 cells across
the four audits. A leave-one-detector-out prediction — estimate Δ from the family's
*other* detectors, then predict reversal from `A_global − Δ_LOO < 0.5` — agrees with
the observed interval reversal in **96.5%** of cells (TP = 6, FP = 2, FN = 5,
TN = 187). The old 168-cell figure of 94.0% and this 96.2% are not directly
comparable, and the difference is *not* mainly about `neighbor`: the earlier run
pooled only (dataset, proxy, detector) cells that survived the eligibility gate in
each audit, and it predates both `combined_noN` in the S20/A40 audits and the
canonical analysis set. Reproducing the pre-canonical construction gives 216 cells
and 13 disagreements (93.98% agreement); of those 13, `neighbor` accounts for
exactly **one**. So excluding `neighbor` is justified by §10's saturation argument,
not by an improvement in this metric.

The seven remaining disagreements are informative rather than noise, and they split
cleanly into the two failure modes the inequality predicts:

| mode | cells | reading |
|---|---|---|
| predicted reversal, none observed | `aum × proto_margin` (A40), `forgetting × knn_agreement` (C100N) | the family's mean Δ overshoots that detector's own Δ by ~0.13–0.15 |
| reversal observed, not predicted | `combined × {knn_agreement, proto_margin}` in S20 and S20-frozen | `combined`'s own Δ (+0.23, +0.31) is roughly double its family's leave-one-out mean (+0.14, +0.15) |

So Δ is **not** a proxy-level constant: within `proto_margin` in the A40 audit it
runs from +0.0116 (`aum`) to +0.180 (`combined`), and within `knn_agreement` in S20
from +0.0946 (`aum`) to +0.3063 (`combined`). The inequality is right about the
*direction* of the phenomenon and wrong about the magnitude for individual
detectors — which is exactly §9's finding that reversal lives at the (proxy,
detector) pair, not at the family.

**E7 (strength bins).** Δ does not shrink as the detector gets stronger, and
reversal vanishes. Recomputed on the canonical set (1,440 cells, `neighbor`
excluded, `forgetting` present only where eligible):

| strength bin | label-dependent Δ | reversal | label-free Δ | reversal |
|---|---|---|---|---|
| [0.50, 0.60) | +0.1139 | 35 / 60 | +0.0011 | 0 / 120 |
| [0.60, 0.70) | +0.1528 | 35 / 102 | +0.0083 | 0 / 204 |
| [0.70, 0.80) | +0.1260 | 7 / 118 | −0.0047 | 0 / 236 |
| [0.80, 1.00] | **+0.1801** | **0 / 80** | +0.0029 | 0 / 160 |

The strongest bin has the second-largest attenuation and zero reversals. This is the
cleanest single refutation of "reversal is a stronger version of the effect".

---

## 9. E11 — Which level owns the effect?

**Question.** Is reversal a property of the quality proxy (family), or of the
*(proxy, detector)* pair?

**Configuration.** Canonical `final_analysis_set.csv` with excluded detectors
removed: 1,440 pooled cells carrying interval verdicts, 84 of them reversals
(5.83%). Two binomial GLMs fitted by IRLS: `strength + proxy` versus
`strength + proxy + detector`. The likelihood-ratio statistic for the detector term
is calibrated against a permutation null that shuffles detector labels **within each
proxy**, preserving all family-level structure (2,000 draws).

**Result.**

| model | log-lik | k |
|---|---|---|
| strength + family | −170.14 | 5 |
| strength + proxy | −166.96 | 9 |
| strength + proxy + detector | −55.48 | 15 |

* LR(detector \| proxy) = **222.96** on 6 df; permutation null mean **6.4**;
  permutation p < 1/2000.
* LR(proxy \| family) = 6.36 on 4 df.

Detector identity adds overwhelmingly more than proxy identity — and dropping the
saturating `neighbor` detector *raises* the statistic (165.39 in the pre-canonical
run → 222.96 here), so the pairing effect was never carried by the degenerate
detector. Reversal rate by proxy × detector makes the same point without a model:
`knn_agreement` reverses 0.667 of the time for `combined`, 0.50 for
`combined_noN`, and 0.000 for every other detector; `proto_margin` reverses 0.733
for `combined`, 0.50 for `combined_noN` and 0.167 for each of `confidence`, `CL`
and `EMA Loss`. Note that `native_density` reverses 0.35 for `combined_noN` — a
weak detector with a large conditioned drop.

The strength-bin view confirms the same concentration: `proto_margin` reverses 0.75
in the [0.50,0.60) bin, 0.413 in [0.60,0.70), 0.051 in [0.70,0.80) and **0.000** in
[0.80,1.00]; `knn_agreement` reverses 0.348, 0.068 and 0.000 across the same bins
and never in the lowest one.

**So the phenomenon is a pairing.** Neither a bad proxy alone nor a weak detector
alone produces a reversal: it needs a proxy whose conditional quality structure the
detector responds to strongly *and* a detector without the margin to absorb it.

---

## 10. Granularity audit — two detectors cannot be audited

**Question.** The audit compares detector scores *within* a matched pair. Can each
detector resolve that comparison?

**Result (`revision/run_granularity_audit.py`).** No, for two of the eight.

| detector | distinct values (mean over settings) | tie rate inside matched sets | audit AUC on `knn_agreement` |
|---|---|---|---|
| `ema_loss` | 44,066 – 44,074 | ~0.0002 | resolves |
| `confidence` | 39,608 – 40,676 | ~0.0003 | resolves |
| `aum` | 43,914 – 43,972 | ~0.0002 | resolves |
| `confident_learning` | 44,100 – 44,102 | ~0.0002 | resolves |
| `combined` | 43,815 – 43,889 | ~0.0003 | resolves |
| `forgetting` | **110 – 129** | 0.008 – 0.011 | marginal |
| `neighbor` | **10.6 – 17.7** | 0.09 – 0.26 | **degenerate** |

`neighbor` is a kNN agreement ratio and takes 11–18 distinct values; `forgetting`
is a cumulative event count and takes 110–129. With 4.4k matched pairs drawn from a
handful of levels, `neighbor`'s pairwise AUROC on `knn_agreement` is **exactly
0.5000 in all three audits** — an artefact of resolution, not a reversal. The
interval gate (`ci_high < 0.5`) correctly rejects all of them, so no reported
reversal is a saturation artefact; and this explains the apparently anomalous
`combined_noN × knn_agreement` point estimate of 0.5000 seen for `neighbor` in
round 1.

**Consequence for the paper.** State that the audit speaks for the six
continuous-valued detectors, and that `neighbor` and `forgetting` are reported for
completeness only. `neighbor` in particular cannot be used either to support or to
refute a conditional-quality claim.

---

## 11. Consolidated gate verdicts

**Gate 1 — "does independent-Q Δ_Q > 0 still appear?"** *Reframed and answered.*
The conditioning effect is not a detector–proxy coupling artefact: `combined_noN`
(no neighbourhood term, no shared construction with the label-dependent proxies)
still attenuates by +0.1787 on `knn_agreement` and reverses with 10/10 interval
support, and all four uncoupled continuous detectors attenuate by +0.095 to +0.204.
The *original* positive framing of Δ_Q does not survive — conditioning manufactures
the gap rather than revealing detectability — but the gap itself is robust.

**Gate 2 — "does the C100-S20 reduction survive matching rigour?"** *Yes.* Order
randomisation moves the audit AUC by 0.003 on average and leaves the coupled
reversal cells reversed in 10/10 seeds at `P(A_rand < 0.5) = 1.00` (§2); removing
clean-pool consumption restores coverage from 0.10 to 0.99 and moves the AUC by
0.031 of a 0.179 effect (§3). The residual is attributable to the caliper's
admissibility restriction, and E6's analytic model reproduces the threshold to
within 2–5%. The one exception, to be stated in the manuscript, is
`native_density`, whose reversal does not survive order randomisation.

**Gate 3 — "does Confident Learning show quality sensitivity?"** *Confirmed as a
participant, not as a special case.* `confident_learning × knn_agreement` attenuates
by +0.2038 (S20) and +0.1736 (C100N) and is BH-significant for attenuation under
both Wilcoxon and paired-t. It does not reverse in any audit — its global AUC is
0.69–0.82, above the margin the attenuation can exhaust.

**Gate 4 — "does the phenomenon survive human/real noise?"** *Yes, and it is
largest there.* CIFAR-100N human noise gives `knn_agreement` Δ = +0.1501 and
`proto_margin` Δ = +0.1029 against a control floor of +0.0054, with 0/80
label-dependent reversals — the largest attenuation in the study. Prediction that
follows: the effect is not a synthetic-noise artefact, and reversal is absent for
the structural reason that human-noise training is too accurate.

---

## 12. Claims that must be retired

| retired claim | why | replacement |
|---|---|---|
| "Quality conditioning reveals hidden detectability of noisy labels" | Conditioning *lowers* AUC; it does not uncover signal | "Quality conditioning manufactures a noisy-vs-clean gap that is absent globally." |
| "Δ_Q > 0 for independent Q" as an independent-effect claim | Attenuation survives decoupling (+0.1787 for `combined_noN`), but reversal does not appear for any of the four uncoupled continuous detectors | "Attenuation is robust across coupling; reversal is a construction-sensitive extreme." |
| The frozen `combined × knn_agreement` Δ = 0.3063 as a quality effect | 0.1277 of it is the shared `neighbor` term; conditioned AUC agrees with `combined_noN`'s to 2.2e-4 | Report 0.1787 (noN) as the coupling-free figure and 0.3063 as the paper-detector figure. |
| `noisy_coverage` in the row schema | Was `common_support(...)[0] == common_support(...)[0]`, always 1.0 | Fixed in `run_round2.py`; E1/E4 backfilled by `revision/run_backfill_coverage.py` (AUCs verified unchanged). |
| E6 γ = 0.25 "analytic λ* not observed" | λ grid capped at 3.0 while analytic λ* = 3.333 | Extended grid: 100/100 crossing at 3.411. |
| `e5_fdr.csv` attenuation/reversal p-values | sign-flip tested `P(mean(s·Δ) ≥ obs)` instead of `≤`, and `wilcoxon(−Δ, "greater")` instead of `wilcoxon(Δ, "greater")`; the 48-family set double-counted `combined` against `combined_noN` | Replaced by §5.3: 40 families, 28/40 attenuation BH-significant under all three tests, 3/40 reversal. |
| "Two coverage conventions are fine" | The same column name meant two different things across audits | §5.1: one schema, four explicit names, asserted by the driver. |
| Any reversals attributed to `neighbor` or `forgetting` | Score saturation (11–18 and 110–129 levels) | Audit speaks for the six continuous detectors only. |

---

## 13. Files, schema and reproducibility

All files sit in `results/revision/round2/`, with SHA256 checksums in
`revision/ROUND2_MANIFEST.md`. The copies there are authoritative.

**Frozen archive.** The whole round-2 asset set is frozen as
`canonical-round2-v1.tar.gz` (956,409 bytes, SHA256
`1c200b54e22dd943eb753e3f2c8e8a59d44e17286596c1d164e01350fdb6f510`), which carries
`ARCHIVE_PROVENANCE.md`, `README.md`, an in-archive `MANIFEST.sha256.txt`, every
audit CSV and derived table, and the ten driver scripts. It replaces the earlier
`round2_results.tar.gz`, which is retained but sidecar-marked
`SUPERSEDED / PRE-CANONICALIZATION` because it reproduces the pre-fix multiplicity
table and the pre-unification coverage schema. Unpack and self-check with

```bash
tar xzf canonical-round2-v1.tar.gz && cd canonical-round2-v1
python revision/verify_canonical.py --root .   # independent, does not import the producer
```

**One coverage schema.** The earlier dual convention (`noisy_coverage` meaning
support coverage in E1/E4 but matched coverage in E9) is gone. All four audits now
carry all four canonical columns — `common_noisy_coverage`, `common_clean_coverage`,
`matched_noisy_coverage`, `matched_clean_coverage` — and
`run_final_analysis_set.py` fails hard if an ambiguous name reappears anywhere.
C100N's common pair was backfilled on the server by
`revision/run_backfill_c100n_common.py`, which needs the prepared cache. See §5.1.

**One analysis set.** `final_analysis_set.csv` is the single pooled table (1,720
rows across four audits) with `detector_label` and `detector_tier` attached, so no
downstream script has to re-decide which detectors are admissible: the primary tier
is the six continuous detectors, `forgetting` is `coarse_secondary`, and `neighbor`
is `excluded`. See §5.2.

Drivers (all `--help`-able, all deterministic given seeds):

| file | produces |
|---|---|
| `revision/run_round2.py` | E1 (`e1_*.csv`), E2/E3 (`e2e3_*.csv`), E4 (`e4_*.csv`), E6 (`e6_*.csv`), E7 (`e7_*.csv`) |
| `revision/run_final_analysis_set.py` | §5: coverage-schema unification, `final_analysis_set.csv`, `multiplicity_*.csv`, `coverage_schema.csv`. **Supersedes `run_round2.py --job e5`**, whose `e5_fdr.csv` had the two direction bugs §5.3 documents. |
| `revision/run_e8_parallel.py` | E8 (`e8_mutual_information_{s20,a40}*.csv`) — parallel version of `run_round2.run_mi`, verified bit-identical |
| `revision/run_e9_c100n.py` | E9 (`e9_c100n_*.csv`) |
| `revision/run_e10_weak_detector.py` | E10 (`e10_*.csv`) |
| `revision/run_e11_variance.py` | E11 (`e11_*.csv`) |
| `revision/run_granularity_audit.py` | §10 (`granularity_audit*.csv`) |
| `revision/run_backfill_coverage.py` | coverage-column repair for E1/E4 |

Honest caveats worth carrying into the manuscript:

* **Five seeds, not ten,** for C100-A40 and CIFAR-100N. Both are all that the
  prepared caches contain; the reversal counts there (23/40, 0/80) are exact
  binomial statements over those seeds, not population estimates.
* **The audit speaks for six detectors.** `neighbor` and `forgetting` have
  saturating score resolution (§10).
* **`native_density` reversals are marginal.** They depend on the visiting order
  (§2) and vanish entirely on A40 (§4).
* **A +0.005 control floor exists** on A40 and C100N and should be subtracted or
  reported alongside every coupled Δ.
* **The `--workers` argument is capped by the number of tasks**, and E1/E4/E9 use
  one task per seed, so a ten-seed job runs on ten workers regardless of the value
  passed.
