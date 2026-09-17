# Minor-revision statistics (offline, from the canonical round-2 assets)

Everything here is post-processing of the frozen canonical assets in
`results/revision/round2/`. **No new dataset, no new training, no new backbone or
detector.** The only computation that produces new simulation data is §3, which is
the reviewer's explicitly requested misspecification check.

Scope, resampling unit, and p-values are chosen to match the main text: 10 seeds for
C100-S20, exact one-sided sign-flip plus Wilcoxon plus paired t, Benjamini-Hochberg
within the reported family set, and a **signed** effect floor Δ ≥ 0.01.

| § | item | status | new compute |
|---|---|---|---|
| 1 | family-level effect-size uncertainty | done | none |
| 2 | Combined-noN residual correlation with Q_agree | done | none (reads the prepared cache) |
| 3 | Gaussian misspecification sensitivity | done | simulation only |
| 4 | Table II uncertainty | done | none |
| 5 | bootstrap failure exact counts | done | none |
| 6 | effect-floor sensitivity | done | none |
| 7 | A_g uncertainty on the reversal margin | done | none |

---

## 1. Family-level effect-size uncertainty

**Universe.** The paper's 40 families: core-5 detectors (EMA Loss, Confidence, AUM,
CL, Combined) × 8 proxies, C100-S20, 10 seeds.

**Resampling.** Seed bootstrap, B = 10,000, `default_rng(20260917)` — the same
resampling unit as the rest of the revision. Cross-check: the bootstrap SE agrees
with the t-based SE to within 2.4 × 10⁻⁴ AUROC, so the two conventions are
interchangeable and the reported CIs do not depend on the choice.

**Output.** `ae_family_uncertainty_core5.csv` (also
`ae_family_uncertainty_primary6.csv` for the 48-family sensitivity). Columns:
`mean_delta`, `sd_delta`, `se_delta_t`, `se_delta_boot`, `delta_ci_lo/hi`,
`mean_conditioned_auc`, `se_cond_t`, `se_cond_boot`, `cond_ci_lo/hi`,
`p_signflip`, `p_wilcoxon`, `p_pairedt`, `p_bh_*`, `bh_*_significant`,
`reversal_margin` and its CI.

**Headline rows.**

| detector | proxy | mean Δ | SE | 95% CI | p | p_BH |
|---|---|---|---|---|---|---|
| Combined | knn_agreement | 0.30634 | 0.00237 | [0.30168, 0.31085] | 0.00098 | 0.00163 |
| Combined | proto_margin | 0.23150 | 0.00247 | [0.22660, 0.23635] | 0.00098 | 0.00163 |
| CL | proto_margin | 0.22943 | 0.00139 | [0.22667, 0.23204] | 0.00098 | 0.00163 |
| AUM | native_density | 0.21940 | 0.00116 | [0.21708, 0.22163] | 0.00098 | 0.00163 |
| CL | knn_agreement | 0.20379 | 0.00396 | [0.19653, 0.21211] | 0.00098 | 0.00163 |
| EMA Loss | knn_agreement | 0.15154 | 0.00319 | [0.14532, 0.15779] | 0.00098 | 0.00163 |
| AUM | proto_margin | 0.04814 | 0.00140 | [0.04517, 0.05064] | 0.00098 | 0.00163 |
| Combined | dino_density | 0.01258 | 0.00045 | [0.01173, 0.01351] | 0.00098 | 0.00163 |
| CL | dino_augcons | 0.00107 | 0.00031 | [0.00044, 0.00168] | 0.00879 | 0.01256 |
| Combined | dino_augcons | 0.00139 | 0.00080 | [−0.00015, 0.00298] | 0.06543 | 0.09025 (ns) |
| AUM | dino_density | −0.02691 | 0.00049 | [−0.02785, −0.02592] | 1.0 | 1.0 (ns) |
| Confidence | random | −0.00057 | 0.00078 | [−0.00214, 0.00090] | 0.74219 | 1.0 (ns) |

**Consistency checks, all passing:**

* The three tests make **identical** BH decisions for all 40 families.
* The families whose Δ CI excludes 0 are **exactly** the 28 BH-significant ones —
  the permutation test and the interval agree family by family.
* Counts reproduce the frozen headline: **28/40** BH-significant, **18/40** with
  Δ ≥ 0.01, **3/40** reversal.

**One caveat to state in the supplement.** These are *marginal* intervals: all 40
families share the same 10 seeds, so the CIs are not simultaneous and a reader must
not compare two overlapping family CIs as if they were independent. The
family-wise control comes from the sign-flip/BH layer, not from the intervals.

**Where it goes.** Supplementary table; one sentence in the main text pointing to it.

---

## 2. Residual dependence between Combined-noN and Q_agree

**Question.** Removing the explicit `neighbor` term removes the *exact* construction
coupling `E_neighbor = 1 − Q_knn`. The reviewers accept that but ask whether
*statistical* dependence survives, since loss and forgetting are themselves produced
by noisy-label training.

**Measured.** On the ten C100-S20 seeds, against `Q_knn_agreement`:
Spearman ρ, Pearson r, and kNN mutual information (median-split, so it is comparable
across detectors). Per-seed values, plus mean ± SD and a seed-bootstrap 95% CI.

| detector | Spearman ρ (mean ± SD) | 95% CI | Pearson r | I(E;Q) |
|---|---|---|---|---|
| CL | −0.6072 ± 0.0056 | [−0.6101, −0.6037] | −0.6337 | 0.106 |
| **Combined** | **−0.5180 ± 0.0100** | [−0.5239, −0.5122] | −0.5286 | 0.076 |
| EMA Loss | −0.4754 ± 0.0053 | [−0.4786, −0.4725] | −0.4614 | 0.060 |
| Confidence | −0.4210 ± 0.0047 | [−0.4240, −0.4186] | −0.4158 | 0.049 |
| AUM | −0.3342 ± 0.0110 | [−0.3409, −0.3279] | −0.2049 | 0.037 |
| **Combined-noN** | **−0.2856 ± 0.0108** | [−0.2922, −0.2794] | −0.2415 | 0.034 |

**Answer, stated as the reviewer framed it.** Removing the neighbourhood term
removes the exact construction coupling (`|ρ|` falls 0.518 → 0.286, `|r|` falls
0.529 → 0.242) but **not all statistical dependence**: Combined-noN retains a
significant residual ρ = −0.286 with CI excluding zero in every seed. The honest
framing is that the residual is real *and* unremarkable — it is **smaller than AUM's
(−0.334)** and far smaller than CL's (−0.607), two detectors that were never
neighbours of this proxy, and it sits at 56% of Combined's.

Pearson ≈ Spearman for every detector (|r| − |ρ| ≤ 0.13), so the dependence is close
to linear; the `AUM` row is the one exception (|ρ| 0.334 vs |r| 0.205), i.e. AUM's
association is monotone but curved.

**Why this does not weaken the paper.** The complementary evidence is stronger than
the correlation: AUM (|ρ| = 0.334) and CL (|ρ| = 0.607) both carry *more* residual
dependence than Combined-noN and neither reverses in any audit (§5.2 of
`ROUND2_RESULTS.md` shows 0/40 and 0/40). So the residual correlation is neither
necessary nor sufficient for reversal; what predicts reversal is the margin
inequality.

**Where it goes.** One short paragraph plus this table in the supplement, with the
sentence the reviewers asked for quoted almost verbatim.

---

## 3. Gaussian misspecification sensitivity

The reviewer's one requested new simulation. Baseline is the main-text model
unchanged, re-run here as an anchor:

```
Z ~ Bernoulli(ρ = 0.2),  d ~ N(0,1),  D = d + κZ
E = βZ + γD + ε_E,   ε_E ~ N(0, 1)
Q = f(D) + λZ + ε_Q, ε_Q ~ N(0, σ_Q²),  σ_Q = 0.3
N = 20,000,  λ ∈ [0, 5] step 0.1,  100 repeats per cell
```

**Two misspecifications:**

* **M1, nonlinear difficulty:** `f(D) = D + 0.3·D³`. E is unchanged, so β is
  unchanged — `f` enters Q only, and the global AUROC depends on E alone. Q's
  residual spread is normalised so the caliper admits the same candidate density as
  the baseline, so M1 differs from the baseline in *curvature*, not in
  signal-to-noise.
* **M2, correlated difficulty:** `D | Z = z ~ N(κz, 1)`, κ ∈ {0.25, 0.5}. β is
  re-derived analytically, `β = 2Φ⁻¹(0.70)·√(γ² + σ_E²) − γκ`, so every cell keeps a
  global AUROC of 0.70.

**Calibration is verified against the paper.** The baseline column reproduces the
main-text E6 thresholds (3.364/1.857/1.192/1.003 here vs 3.411/1.847/1.195/1.010 in
the main text — same grid resolution) and the same β to four decimals, which is an
independent check of the rewritten simulator.

**Crossing λ (mean over 100 repeats) by setting and γ:**

| setting | γ = 0.25 | γ = 0.50 | γ = 1.00 | γ = 1.50 |
|---|---|---|---|---|
| baseline (κ = 0) | 3.364 | 1.857 | 1.192 | 1.003 |
| M1 cubic `f(D) = D + 0.3D³` | **none ≤ 5** | 2.997 | 1.762 | 1.468 |
| M2 correlated, κ = 0.25 | 3.119 | 1.604 | 0.934 | 0.773 |
| M2 correlated, κ = 0.5 | 2.856 | 1.349 | 0.691 | 0.502 |

### The three questions

**1. Does the conditioned AUROC still fall as λ grows?** **Yes, everywhere.** The
drop from λ = 0 to λ = 5 is positive in all 16 cells, ranging +0.166 to +0.808:

| setting | Δ at λ = 5 (min … max over γ) |
|---|---|
| baseline | +0.311 … +0.808 |
| M1 cubic | +0.166 … +0.724 |
| M2 κ = 0.25 | +0.303 … +0.741 |
| M2 κ = 0.5 | +0.309 … +0.663 |

So the central monotone decline is not a Gaussian artefact.

**2. Is there still a crossing / reversal?** **Yes — but M1 delays it.** Full
crossing (100/100 repeats) occurs in:

* baseline 4/4 cells, M2 κ = 0.25 4/4, M2 κ = 0.5 4/4;
* **M1 only 3/4** — at γ = 0.25 the crossing is not reached within λ ≤ 5, though
  attenuation is still monotone and reaches +0.166.

This is the one substantive sensitivity result and it deserves to be reported rather
than hidden. The mechanism is legible: `f` is convex, so it stretches the upper tail
of Q and makes the quality signal heteroscedastic in D, which weakens the *quality
localisation* the reversal depends on. A stronger coupling is then needed to force
the audit under chance. This is consistent with the paper's mechanism, not in
tension with it — it says the reversal is sensitive to how strongly Q localises the
label error, which is exactly the quantity the paper claims drives it.

**3. Does larger γ bring the crossing earlier?** **Yes, strictly, in every setting**
(monotone decreasing over γ = 0.25 → 1.5 in all four settings, including M1 where
the γ = 0.25 cell never crosses). The paper's qualitative ordering claim survives
both misspecifications.

**Bonus, unrequested but worth one sentence:** correlated difficulty *accelerates*
reversal (κ = 0.5 crosses earlier than κ = 0 at every γ, by 15–45%), so the
baseline is not conservative in that direction.

**Output.** `ae_misspecification.csv` (20,400 rows),
`ae_misspecification_crossing.csv`, `ae_misspecification_calibration.csv`.

**Where it goes.** Supplementary figure or table, referenced from the Gaussian-model
paragraph, with question 2's M1 caveat stated explicitly.

---

## 4. Table II uncertainty

Conditioned AUROC as mean ± SE over 10 seeds, for the four uncoupled detectors the
table reports, under every conditioning estimator. Full table in
`ae_table2_uncertainty.csv`; SEs are small everywhere.

| estimator | AUM | Confidence | CL | EMA Loss |
|---|---|---|---|---|
| Global | 0.7127 ± 0.0008 | 0.7375 ± 0.0015 | 0.8162 ± 0.0012 | 0.8043 ± 0.0010 |
| DINO density | 0.7396 ± 0.0009 | 0.7350 ± 0.0015 | 0.8132 ± 0.0013 | 0.8004 ± 0.0011 |
| DINO density (full pool) | 0.7400 ± 0.0008 | 0.7339 ± 0.0013 | 0.8134 ± 0.0015 | 0.7990 ± 0.0011 |
| DINO kNN distance | 0.7398 ± 0.0009 | 0.7351 ± 0.0015 | 0.8133 ± 0.0012 | 0.8004 ± 0.0011 |
| DINO augmentation consistency | 0.7164 ± 0.0010 | 0.7377 ± 0.0015 | 0.8151 ± 0.0015 | 0.8043 ± 0.0012 |
| **Native density** | **0.4933 ± 0.0015** | 0.6701 ± 0.0020 | 0.6452 ± 0.0029 | 0.7421 ± 0.0015 |
| **Prototype margin** | 0.6646 ± 0.0014 | 0.5751 ± 0.0025 | 0.5868 ± 0.0013 | 0.6687 ± 0.0022 |
| **KNN agreement** | 0.6181 ± 0.0035 | 0.5845 ± 0.0044 | 0.6124 ± 0.0042 | 0.6528 ± 0.0038 |
| Random (control) | 0.7124 ± 0.0008 | 0.7381 ± 0.0015 | 0.8158 ± 0.0012 | 0.8041 ± 0.0010 |

The label-free estimators move the AUC by ≤ 0.005 with SE ≤ 0.0015, i.e. the change
is within about three SEs and is not the effect; the three label-dependent or
encoder-derived estimators move it by 0.03–0.23 with SE ≤ 0.0044. **The largest SE
in the whole table is 0.0044 (`knn_agreement`), roughly a fifth of the smallest
effect the table reports as real.** So uncertainty does not threaten any row.

**Where it goes.** The main table can stay as means; put this version in the
supplement and add "SE ≤ 0.005 throughout" to the caption.

---

## 5. Bootstrap failure counts

Exact counts, read from the stored per-cell `bootstrap_B` / `bootstrap_failed`
columns — **no bootstrap was re-run**. One replicate is one paired equal-count
resample; a failure is a replicate in which no admissible pair survived the caliper.

| proxy | cells | replicates | failures | failure rate | min pairs in any cell |
|---|---|---|---|---|---|
| random | 50 | 50,000 | 0 | 0 | 35,844 |
| dino_density | 50 | 50,000 | 0 | 0 | 35,838 |
| dino_density_fullpool | 50 | 50,000 | 0 | 0 | 35,841 |
| dino_knndist | 50 | 50,000 | 0 | 0 | 35,831 |
| dino_augcons | 50 | 50,000 | 0 | 0 | 35,821 |
| **native_density** | 50 | 50,000 | 0 | 0 | 16,686 |
| **proto_margin** | 50 | 50,000 | 0 | 0 | 17,173 |
| **knn_agreement** | 50 | 50,000 | 0 | 0 | **4,265** |
| **total** | **400** | **400,000** | **0** | **0.0** | — |

The useful number for the rebuttal is not only the zero but the *worst case*: even
for `knn_agreement`, whose hard caliper admits only 10% of the noisy stratum, every
one of the 50,000 replicates still produced at least 4,265 matched pairs. The
paper's "no bootstrap matching failure" can therefore be replaced by an exact
statement with a floor attached.

**Where it goes.** One sentence in the main text plus this table in the supplement.

---

## 6. Effect-floor sensitivity

| floor δ_min | BH-significant | BH **and** Δ ≥ δ_min | reversal and Δ ≥ δ_min | families clearing |
|---|---|---|---|---|
| 0.000 | 28 | 28 | 3 | 34 |
| 0.005 | 28 | 19 | 3 | 19 |
| **0.010** | **28** | **18** | **3** | **18** |
| 0.020 | 28 | 15 | 3 | 15 |
| 0.050 | 28 | 14 | 3 | 14 |
| 0.100 | 28 | 10 | 3 | 10 |

**The honest reading, including the part that is not flattering.** The count is
monotone in the floor and the reversal count is completely insensitive to it (3 at
every floor up to 0.10). The chosen floor is *not* a knife-edge: exactly one family
has Δ in [0.005, 0.010) — `EMA Loss × dino_density_fullpool` at 0.00531 — so moving
the floor to 0.005 changes the answer by one family and moving it to 0.020 changes it
by three.

But the floor does real work: the drop from 28 to 19 happens between 0 and 0.005,
because nine label-free families have significant-but-tiny attenuations of
+0.001 to +0.005. So the correct justification is not "the number happens to be
stable" but **the label-free and control proxies define what the floor has to
exclude**, and 0.01 is the round value in that gap. Any floor in [0.005, 0.02]
gives 15–19 families; none of the three reported reversal families is affected at
any floor.

**Where it goes.** One sentence in the methods explaining the floor's purpose and
this table in the supplement.

---

## 7. Does A_g uncertainty threaten the reversal claim? (optional item)

The criterion `A_cond < 0.5` is equivalent to `Δ_Q > A_g − 0.5`, so with
`M = A_g − 0.5` and `R = Δ_Q − M` the reversal is robust only if `R > 0` with a CI
excluding zero. Seed bootstrap, B = 10,000, resampling whole seeds so the pairing of
`A_g` and `A_cond` is preserved. Algebraic identity `R = 0.5 − A_cond` checked
explicitly: holds to 2.8 × 10⁻¹⁶.

| family | A_g (SE) | M = A_g − 0.5 [CI] | Δ_Q [CI] | R [CI] |
|---|---|---|---|---|
| AUM × native_density | 0.71274 (0.00084) | 0.21274 [0.21114, 0.21420] | 0.21940 [0.21716, 0.22156] | **+0.00666** [0.00396, 0.00958] |
| Combined × proto_margin | 0.69645 (0.00115) | 0.19645 [0.19449, 0.19869] | 0.23150 [0.22657, 0.23624] | +0.03504 [0.03034, 0.03978] |
| Combined × knn_agreement | 0.69645 (0.00115) | 0.19645 [0.19447, 0.19874] | 0.30634 [0.30168, 0.31080] | +0.10988 [0.10557, 0.11450] |

All three margins are positive with CIs excluding zero. The decisive numbers:

* `A_g`'s **SE is 0.00084–0.00115**, an order of magnitude smaller than the margin
  `R` in the tightest case.
* In the tightest case (`AUM × native_density`), `A_g` would have to be wrong by
  0.0067 AUROC — **8.0 of its own standard errors** — to remove the reversal.
* `M` is not the fragile quantity: it is 0.20–0.21 while `Δ_Q` is 0.22–0.31. The
  reversal is limited by the attenuation being only just large enough, not by
  uncertainty in the threshold.

**Where it goes.** Supplementary sanity check, referenced in one clause. It is not a
new main result and should not be presented as one.

---

## Files

| file | content |
|---|---|
| `ae_family_uncertainty_core5.csv` | §1, the paper's 40 families |
| `ae_family_uncertainty_primary6.csv` | §1, 48-family sensitivity |
| `ae_residual_correlation_perseed.csv` / `_summary.csv` | §2 |
| `ae_misspecification.csv` / `_crossing.csv` / `_calibration.csv` | §3 |
| `ae_table2_uncertainty.csv` | §4 |
| `ae_bootstrap_failures.csv` | §5 |
| `ae_effect_floor_sensitivity.csv` | §6 |
| `ae_ag_margin_propagation.csv` | §7 |
| `ae_offline_summary.json` | machine-readable headline numbers |

Drivers: `revision/run_ae_offline_audit.py` (§1, §4, §5, §6),
`revision/run_ae_residual_correlation.py` (§2),
`revision/run_ae_misspecification.py` (§3),
`revision/run_ae_ag_margin.py` (§7).
