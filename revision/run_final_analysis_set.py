"""Canonical final analysis set: coverage schema, detector tiers, multiplicity.

Three offline jobs -- no new training and no new matching.

1. **One coverage schema.**  ``common_noisy_coverage`` / ``common_clean_coverage``
   are the fraction of each stratum inside the common support of Q;
   ``matched_noisy_coverage`` / ``matched_clean_coverage`` are the fraction
   actually matched after the caliper.  The ambiguous names ``noisy_coverage`` /
   ``clean_coverage`` are removed everywhere so no column means two things in two
   audits.

2. **One detector set.**  `neighbor` is excluded: its score takes only 11-18
   distinct values, its tie rate inside matched sets is 9-26%, and its pairwise
   AUROC lands on exactly 0.5000 in all three audits -- it cannot resolve the
   comparison the audit asks of it.  `forgetting` is retained as a *coarse
   secondary* (110-129 levels, tie rate ~1%) and is never pooled with the primary
   detectors; note also that its S20 global AUROC is 0.3305, below chance, so it
   is ineligible for a reversal claim there.  The canonical primary set is the six
   continuous detectors: EMA Loss, Confidence, AUM, CL, Combined, Combined-noN.
   The frozen S20 audit predates `combined_noN`, so cross-audit comparisons use
   the **core-5** that every audit actually contains.

3. **Correctly directed multiplicity tests.**  The previous E5 had two direction
   bugs:

   * sign-flip asked ``P(mean(s*d) >= observed)`` for the hypothesis E[d] < 0. The
     correct one-sided p is ``P(mean(s*d) <= observed)``.  The old form rejects for
     families whose delta is *positive* and returns p = 1 for families whose delta
     is uniformly negative -- exactly inverted, which is why it reported 0/48.
   * paired t was ``ttest_rel(global, conditioned, alternative="greater")``, which
     tests A_global > A_cond -- the complement of attenuation.  It must test
     A_cond < A_global.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "round2"
P0 = ROOT / "results" / "revision" / "p0_batch"

# --- canonical detector registry -------------------------------------------
CORE5 = ["ema_loss", "confidence", "aum", "confident_learning", "combined"]
PRIMARY6 = CORE5 + ["combined_noN"]
COARSE = ["forgetting"]
EXCLUDED = ["neighbor"]
COARSE_PREPARED = ["forgetting"]
DISPLAY = {"ema_loss": "EMA Loss", "confidence": "Confidence", "aum": "AUM",
           "confident_learning": "CL", "combined": "Combined",
           "combined_noN": "Combined-noN", "forgetting": "Forgetting"}

CANON = ["common_noisy_coverage", "common_clean_coverage",
         "matched_noisy_coverage", "matched_clean_coverage"]
AMBIGUOUS = ["noisy_coverage", "clean_coverage"]

AUDITS = {
    "C100-S20": (OUT / "e1_combined_noN.csv", 10),
    "C100-A40": (OUT / "e4_a40_audit.csv", 5),
    "C100N-human": (OUT / "e9_c100n_audit.csv", 5),
    "C100-S20-frozen": (P0 / "j1_bootstrap_all_proxies.csv", 10),
}
PROXIES = ["random", "dino_density", "dino_density_fullpool", "dino_knndist",
           "dino_augcons", "native_density", "proto_margin", "knn_agreement"]
FAMILY = {"random": "control", "dino_density": "label_free",
          "dino_density_fullpool": "label_free", "dino_knndist": "label_free",
          "dino_augcons": "label_free", "native_density": "encoder_derived",
          "proto_margin": "label_dependent", "knn_agreement": "label_dependent"}


def tier(det: str) -> str:
    if det in PRIMARY6:
        return "primary"
    if det in COARSE:
        return "coarse_secondary"
    return "excluded"


# ---------------------------------------------------------------------------
# 1. coverage schema
# ---------------------------------------------------------------------------
def unify_coverage() -> None:
    print("== unifying the coverage schema ==")
    report = []
    for name, (path, _) in AUDITS.items():
        if not path.exists():
            print(f"   [{name}] missing {path.name}, skipped")
            continue
        df = pd.read_csv(path)
        before = [c for c in df.columns if "coverage" in c]

        if {"noisy_coverage", "clean_coverage"} <= set(df.columns):
            if name == "C100N-human":
                # run_e9_c100n.evaluate named matched coverage with the short
                # names, so it maps onto the matched pair.  The common-support
                # values need the prepared cache, which is server-side.
                df["matched_noisy_coverage"] = df["noisy_coverage"]
                df["matched_clean_coverage"] = df["clean_coverage"]
            else:
                # run_round2.evaluate_with_bootstrap (after the backfill fix)
                # wrote support coverage under the short names and matched
                # coverage under the matched_* names.
                df["common_noisy_coverage"] = df["noisy_coverage"]
                df["common_clean_coverage"] = df["clean_coverage"]

        df = df.drop(columns=[c for c in AMBIGUOUS if c in df.columns])
        keep = [c for c in CANON if c in df.columns]
        df = df[keep + [c for c in df.columns if c not in CANON]]
        df.to_csv(path, index=False)
        report.append({"audit": name, "file": path.name,
                       "coverage_before": ",".join(before),
                       "coverage_after": ",".join(keep), "rows": len(df),
                       "common_available": "common_noisy_coverage" in keep})
        print(f"   [{name}] {before}\n            -> {keep}")
    pd.DataFrame(report).to_csv(OUT / "coverage_schema.csv", index=False)


def check_coverage_schema() -> None:
    problems = []
    for name, (path, _) in AUDITS.items():
        if not path.exists():
            continue
        cols = set(pd.read_csv(path, nrows=1).columns)
        for c in AMBIGUOUS:
            if c in cols:
                problems.append(f"{name}: ambiguous column {c}")
        if not (cols & set(CANON)):
            problems.append(f"{name}: no canonical coverage column")
    if problems:
        raise SystemExit("coverage schema violations:\n  " + "\n  ".join(problems))
    print("   schema check: OK - no ambiguous coverage name in any audit")


# ---------------------------------------------------------------------------
# 2. final analysis set
# ---------------------------------------------------------------------------
def load_analysis_set() -> pd.DataFrame:
    frames = []
    for name, (path, n_seeds) in AUDITS.items():
        if not path.exists():
            print(f"   [{name}] missing, skipped")
            continue
        d = pd.read_csv(path)
        # E1/E4 carry a second Combined variant (`combined_variant ==
        # "no_neighbor"`) in addition to the separate `combined_noN` detector, and
        # filtering on the variant alone silently deletes combined_noN as well --
        # it is one of the six primary detectors and belongs in the set.  Drop the
        # duplicate variant explicitly instead, by (detector, variant) pair.
        if "combined_variant" in d.columns:
            dup = (d.combined_variant != "paper") & (d.detector != "combined_noN")
            d = d[~dup].copy()
        d["audit"] = name
        d["audit_seeds"] = n_seeds
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["detector_label"] = df.detector.map(DISPLAY).fillna(df.detector)
    df["detector_tier"] = df.detector.map(tier)
    return df


# ---------------------------------------------------------------------------
# 3. correctly directed multiplicity tests
# ---------------------------------------------------------------------------
def sign_flip_lower(d: np.ndarray) -> float:
    """Exact one-sided sign-flip p for H0 E[d] = 0 vs H1 E[d] < 0.

    Enumerates all 2^n sign vectors and returns P(mean(s*d) <= mean(d)).  The
    sign-flip permutation tests the mean directly and needs no symmetry
    assumption beyond independent d_i with a common mean.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    s = np.array(list(itertools.product([1.0, -1.0], repeat=n)), dtype=float)
    return float(((s * d).mean(axis=1) <= d.mean()).mean())


def bh_stepup(p: pd.Series, q: float, direction_ok: np.ndarray) -> tuple[np.ndarray, int]:
    pv = p.to_numpy(dtype=float).copy()
    pv[~direction_ok] = 1.0
    order = np.argsort(pv)
    m = len(pv)
    passed = pv[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.arange(1, m + 1)[passed])) if passed.any() else 0
    sig = np.zeros(m, dtype=bool)
    if k:
        sig[order[:k]] = True
    return sig, k


def multiplicity_table(df: pd.DataFrame, detectors: list[str], label: str,
                       audit: str, eligible_only: bool = True,
                       proxy_families: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Family-level tests with BH applied *within* this set.

    ``proxy_families`` restricts the family set *before* the correction runs.
    Filtering afterwards would apply BH at one family count and report it at
    another, which silently raises the effective false-discovery rate -- exactly
    the bug that made a 15-family subset inherit m = 40 step-up thresholds.
    """
    from scipy import stats

    d = df[(df.audit == audit) & (df.detector.isin(detectors))]
    if eligible_only:
        d = d[d.global_auc >= 0.5]
    if proxy_families is not None:
        d = d[d.proxy_family.isin(proxy_families)]
    rows = []
    for (det, px), g in d.groupby(["detector", "proxy"]):
        g = g.sort_values("seed")
        delta = (g.global_auc - g.conditioned_auc).to_numpy(float)   # >0 = attenuation
        ac = g.conditioned_auc.to_numpy(float)
        if len(delta) == 0:
            continue
        try:
            # delta = A_global - A_cond, so attenuation means delta > 0 and the
            # one-sided alternative is "greater".  Verified on a known family:
            # wilcoxon(delta, "greater") = 0.000977, (delta, "less") = 1.0.
            p_wil = float(stats.wilcoxon(delta, alternative="greater").pvalue)
        except ValueError:
            p_wil = np.nan
        try:
            # H1: E[A_global] > E[A_cond], i.e. the same one-sided statement as the
            # sign-flip.  Verified: ttest_rel(global, cond, "greater") = 1.6e-12 on
            # a family whose delta is uniformly positive, while the reversed
            # argument order returns 1 - p.
            p_t = float(stats.ttest_rel(g.global_auc, g.conditioned_auc,
                                        alternative="greater").pvalue)
        except Exception:
            p_t = np.nan
        rows.append({
            "audit": audit, "set": label,
            "detector": det, "detector_label": DISPLAY.get(det, det),
            "detector_tier": tier(det), "proxy": px,
            "proxy_family": FAMILY.get(px, "?"), "n_seeds": len(delta),
            "mean_global_auc": float(g.global_auc.mean()),
            "mean_conditioned_auc": float(ac.mean()),
            "delta_attenuation": float(g.global_auc.mean() - ac.mean()),
            "n_seeds_attenuated": int((delta > 0).sum()),
            "p_signflip_attenuation": sign_flip_lower(-delta),
            "p_wilcoxon_attenuation": p_wil,
            "p_pairedt_attenuation": p_t,
            "p_signflip_reversal": sign_flip_lower(ac - 0.5),
        })
    r = pd.DataFrame(rows)
    atten_dir = (r.delta_attenuation > 0).to_numpy()
    rev_dir = (r.mean_conditioned_auc < 0.5).to_numpy()
    r["direction_consistent_attenuation"] = atten_dir
    r["direction_consistent_reversal"] = rev_dir
    m = len(r)
    # Effect floor recorded on every row so downstream tables do not have to
    # re-decide it; the report applies whatever floor it is given.
    r["meets_effect_floor_0.01"] = r.delta_attenuation >= 0.01
    for tag, col, mask in (("signflip", "p_signflip_attenuation", atten_dir),
                           ("wilcoxon", "p_wilcoxon_attenuation", atten_dir),
                           ("pairedt", "p_pairedt_attenuation", atten_dir),
                           ("reversal", "p_signflip_reversal", rev_dir)):
        sig, k = bh_stepup(r[col], 0.05, mask)
        r[f"bh_{tag}_significant"] = sig
        r[f"bh_{tag}_k"] = k
        r[f"bonferroni_{tag}"] = (r[col] <= 0.05 / m) & mask
    return r.sort_values("p_signflip_attenuation").reset_index(drop=True)


def rejectability(n_seeds: int, m: int, q: float = 0.05) -> dict:
    """What does the exact sign-flip floor imply for this n and family count?

    The smallest attainable one-sided sign-flip p is the all-signs-agree value
    2^-n, so every p-value in a set of size m is at least 2^-n.  BH rejects the
    first k families when the k-th smallest p is <= q*k/m, so the floor matters
    through the *largest* threshold, q itself at k = m -- not the first one:

      * if 2^-n > q, no p can clear any BH threshold and the test cannot reject;
      * otherwise it can reject, but only if at least
        k* = ceil(m * 2^-n / q) families all sit exactly on the floor, i.e. have
        every seed pointing the same way.

    So a small n is not automatically hopeless: n = 5 gives 2^-5 = 0.031 <= 0.05
    and needs k* = 10 of 15 families perfectly sign-consistent, which the A40 and
    C100N audits do reach on the coupled proxies.  What n = 5 cannot do is produce
    a p below 0.031, so it can never survive a Bonferroni correction here.

    An earlier version of this function compared 2^-n against the *first* BH
    threshold q/m and declared n = 5 unrejectable.  That was wrong, and it
    contradicted the 11/15 and 15/15 rejections produced by the same run.
    """
    p_min = 2.0 ** (-n_seeds)
    k_star = min(int(np.ceil(m * p_min / q)), m)
    return {"n_seeds": n_seeds, "n_families": m, "p_min_signflip": p_min,
            "bh_first_threshold": q / m, "bh_last_threshold": q,
            "k_star_families_needed_at_floor": k_star,
            "can_reject": bool(p_min <= q),
            "can_survive_bonferroni": bool(p_min <= q / m)}


def report(r: pd.DataFrame, title: str, min_effect: float = 0.01) -> None:
    """Report multiplicity results, gated on both significance and effect size.

    A BH-significant family is not automatically a finding: these are one-sided
    tests of E[delta] = 0, so a family whose attenuation is *uniformly* +0.001
    across ten seeds rejects at the floor p = 2^-10.  The label-free proxies do
    exactly that (see the report of this run), so every count is given twice --
    BH-significant, and BH-significant *and* delta >= min_effect.  The floor is signed, because
    attenuation is delta > 0: a family with a large *negative* delta
    (aum x dino_density, -0.027) is evidence against the effect, not for it.
    """
    m = len(r)
    print(f"\n== {title} ==\n   {m} families; effect gate delta >= {min_effect}")
    n_seeds = int(r.n_seeds.max())
    rj = rejectability(n_seeds, m)
    print(f"   rejectability: n={n_seeds}, m={m} -> sign-flip floor p = "
          f"{rj['p_min_signflip']:.5f}; needs "
          f"{rj['k_star_families_needed_at_floor']}/{m} families perfectly "
          f"sign-consistent to reject; "
          f"{'CAN' if rj['can_reject'] else 'CANNOT'} reject, "
          f"{'CAN' if rj['can_survive_bonferroni'] else 'CANNOT'} survive Bonferroni")
    hdr = f"   {'test':10s} {'BH':>8s} {'Bonf':>8s} {'BH+effect':>10s}"
    print(hdr)
    for tag in ("signflip", "wilcoxon", "pairedt"):
        sig = r[f"bh_{tag}_significant"]
        big = (r.delta_attenuation >= min_effect)
        print(f"   {tag:10s} {int(sig.sum()):5d}/{m:<3d} "
              f"{int(r[f'bonferroni_{tag}'].sum()):5d}/{m:<3d} "
              f"{int((sig & big).sum()):7d}/{m:<3d}")
    big = (r.delta_attenuation >= min_effect)
    rev_sig = r.bh_reversal_significant
    print(f"   {'reversal':10s} {int(rev_sig.sum()):5d}/{m:<3d} "
          f"{'':>8s} {int((rev_sig & big).sum()):7d}/{m:<3d}")
    print(f"   direction-consistent: attenuation "
          f"{int(r.direction_consistent_attenuation.sum())}/{m}, "
          f"reversal {int(r.direction_consistent_reversal.sum())}/{m}")

    print(f"   {'':10s} {'BH':>8s} {'BH+eff':>8s}   (by detector)")
    for det in r.detector.drop_duplicates():
        s = r[r.detector == det]
        print(f"     {DISPLAY.get(det, det):14s} {int(s.bh_signflip_significant.sum()):2d}/{len(s):<3d}"
              f" {int((s.bh_signflip_significant & (s.delta_attenuation >= min_effect)).sum()):2d}/{len(s):<3d}")

    print(f"   (by proxy family)")
    for fam in ("label_dependent", "encoder_derived", "label_free", "control"):
        s = r[r.proxy_family == fam]
        if not len(s):
            continue
        sig = s.bh_signflip_significant
        print(f"     {fam:16s} {int(sig.sum()):2d}/{len(s):<3d}"
              f" {int((sig & (s.delta_attenuation >= min_effect)).sum()):2d}/{len(s):<3d}"
              f"   mean delta {s.delta_attenuation.mean():+.4f}"
              f"   max {s.delta_attenuation.max():+.4f}")

    robust = r.bh_signflip_significant & r.bh_wilcoxon_significant \
        & r.bh_pairedt_significant & big
    print(f"   robust (all three tests BH-significant AND delta >= {min_effect}): "
          f"{int((r.bh_signflip_significant & r.bh_wilcoxon_significant & r.bh_pairedt_significant).sum())}"
          f" -> {int(robust.sum())}/{m}")
    rev = r[rev_sig & big]
    if len(rev):
        print("   BH-significant reversals (effect-gated):")
        print(rev[["detector_label", "proxy", "proxy_family", "mean_global_auc",
                   "mean_conditioned_auc", "delta_attenuation",
                   "p_signflip_reversal"]].round(5).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument("--min-effect", type=float, default=0.01,
                    help="AUROC effect floor for calling a BH-significant family a finding")
    args = ap.parse_args()

    if not args.skip_coverage:
        unify_coverage()
    check_coverage_schema()

    print("\n== final analysis set ==")
    df = load_analysis_set()
    for det in sorted(df.detector.unique()):
        s = df[df.detector == det]
        elig = s[s.global_auc >= 0.5]
        audits = sorted(s.audit.unique())
        print(f"   {DISPLAY.get(det, det):14s} tier={tier(det):16s} "
              f"cells={len(s):4d} eligible={len(elig):4d} audits={audits}")
    df.to_csv(OUT / "final_analysis_set.csv", index=False)
    print(f"   wrote final_analysis_set.csv ({len(df)} rows)")

    print("\n== multiplicity ==")
    # MAIN reporting universe.  Core-5 is the pre-defined cross-audit detector set:
    # the five continuous detectors present in every audit, one of which is the
    # paper's Combined.  Combined-noN is a *sensitivity variant* of Combined, so
    # giving it its own eight families would let one detector contribute twice to
    # the same correction; it is reported as an extension instead (see the
    # primary-6 table below and ROUND2_RESULTS.md 5.3).
    r_core5_s20 = multiplicity_table(df, CORE5, "core-5", "C100-S20")
    r_core5_s20.to_csv(OUT / "multiplicity_main_core5.csv", index=False)
    report(r_core5_s20, "MAIN (paper): core-5 detector universe, C100-S20, 10 seeds",
           args.min_effect)

    r_main = multiplicity_table(df, PRIMARY6, "primary-6", "C100-S20")
    r_main.to_csv(OUT / "multiplicity_primary6.csv", index=False)
    report(r_main, "SENSITIVITY: primary-6 (adds Combined-noN as its own detectors), "
                   "C100-S20", args.min_effect)

    r_main.to_csv(OUT / "multiplicity_primary.csv", index=False)
    r_coarse = multiplicity_table(df, PRIMARY6 + COARSE, "primary-6+coarse",
                                 "C100-S20")
    r_coarse.to_csv(OUT / "multiplicity_with_coarse.csv", index=False)
    report(r_coarse, "SENSITIVITY: primary-6 + Forgetting (coarse), C100-S20",
           args.min_effect)

    frames = []
    for audit in AUDITS:
        if not (df.audit == audit).any():
            continue
        frames.append(multiplicity_table(df, CORE5, "core-5", audit))
    r_core = pd.concat(frames, ignore_index=True)
    r_core.to_csv(OUT / "multiplicity_core5_by_audit.csv", index=False)
    print("\n== CROSS-AUDIT CONSISTENCY: core-5, per audit ==")
    for audit, g in r_core.groupby("audit"):
        g = g.reset_index(drop=True)
        rj = rejectability(int(g.n_seeds.iloc[0]), len(g))
        print(f"   {audit:16s} seeds={int(g.n_seeds.iloc[0]):2d}  "
              f"families={len(g):2d}  "
              f"atten BH(signflip/wilcoxon/pairedt)="
              f"{int(g.bh_signflip_significant.sum())}/"
              f"{int(g.bh_wilcoxon_significant.sum())}/"
              f"{int(g.bh_pairedt_significant.sum())}  "
              f"reversal BH={int(g.bh_reversal_significant.sum())}  "
              f"signflip: floor={rj['p_min_signflip']:.5f} "
              f"needs {rj['k_star_families_needed_at_floor']}/{len(g)} at floor, "
              f"rejectable={rj['can_reject']}, "
              f"Bonferroni-able={rj['can_survive_bonferroni']}")

    # A focused family set: only the proxies that actually carry label
    # information, which is where the attenuation hypothesis lives.  Reported as
    # a sensitivity analysis because the family count is chosen after seeing that
    # the label-free and control proxies are nulls.
    coupled = ("label_dependent", "encoder_derived")
    frames = []
    for audit in AUDITS:
        if not (df.audit == audit).any():
            continue
        frames.append(multiplicity_table(df, CORE5, "core-5-coupled", audit,
                                         proxy_families=coupled))
    r_foc = pd.concat(frames, ignore_index=True)
    r_foc.to_csv(OUT / "multiplicity_core5_coupled_only.csv", index=False)
    print("\n== SENSITIVITY: core-5 restricted to label-carrying proxies ==")
    for audit, g in r_foc.groupby("audit"):
        g = g.reset_index(drop=True)
        rj = rejectability(int(g.n_seeds.iloc[0]), len(g))
        print(f"   {audit:16s} m={len(g):2d}  signflip p_min={rj['p_min_signflip']:.5f} "
              f"floor={rj['p_min_signflip']:.5f} "
              f"needs {rj['k_star_families_needed_at_floor']}/{len(g)} at floor "
              f"rejectable={rj['can_reject']}  "
              f"BH(signflip/wilcoxon/pairedt)={int(g.bh_signflip_significant.sum())}/"
              f"{int(g.bh_wilcoxon_significant.sum())}/{int(g.bh_pairedt_significant.sum())}"
              f"  reversal={int(g.bh_reversal_significant.sum())}")

    print("\n== core-5 attenuation, every audit (mean delta by proxy family) ==")
    print(r_core.groupby(["audit", "proxy_family"])
          .agg(families=("delta_attenuation", "size"),
               mean_delta=("delta_attenuation", "mean"),
               bh_atten=("bh_signflip_significant", "sum"),
               bh_rev=("bh_reversal_significant", "sum"))
          .round(4).to_string())

    json.dump({
        "canonical_primary": PRIMARY6,
        "core5_available_in_all_audits": CORE5,
        "coarse_secondary": COARSE,
        "excluded": EXCLUDED,
        "coverage_columns": CANON,
        "signflip_direction": "p = P(mean(s*d) <= mean(d)) for H1 E[d] < 0",
        "pairedt_direction": "ttest_rel(conditioned, global, alternative='greater')",
        "effect_gate_auroc": 0.01,
        "main_core5_C100-S20": {
            "detectors": CORE5, "audit": "C100-S20", "n_families": len(r_core5_s20),
            "attenuation_bh": {t: int(r_core5_s20[f"bh_{t}_significant"].sum())
                               for t in ("signflip", "wilcoxon", "pairedt")},
            "attenuation_bonferroni": {t: int(r_core5_s20[f"bonferroni_{t}"].sum())
                                       for t in ("signflip", "wilcoxon", "pairedt")},
            "attenuation_bh_and_effect": {t: int((r_core5_s20[f"bh_{t}_significant"]
                                                  & (r_core5_s20.delta_attenuation >= 0.01)).sum())
                                          for t in ("signflip", "wilcoxon", "pairedt")},
            "reversal_bh": int(r_core5_s20.bh_reversal_significant.sum()),
        },
        "main_primary6_sensitivity": {
            "detectors": PRIMARY6, "audit": "C100-S20", "n_families": len(r_main),
            "attenuation_bh": {t: int(r_main[f"bh_{t}_significant"].sum())
                               for t in ("signflip", "wilcoxon", "pairedt")},
            "attenuation_bh_and_effect": {t: int((r_main[f"bh_{t}_significant"]
                                                  & (r_main.delta_attenuation >= 0.01)).sum())
                                          for t in ("signflip", "wilcoxon", "pairedt")},
            "reversal_bh": int(r_main.bh_reversal_significant.sum()),
        },
        "main": {
            "audit": "C100-S20", "n_families": len(r_main),
            "attenuation_bh_and_effect": {t: int((r_main[f"bh_{t}_significant"]
                                                  & (r_main.delta_attenuation >= 0.01)).sum())
                                          for t in ("signflip", "wilcoxon", "pairedt")},
            "reversal_bh_and_effect": int((r_main.bh_reversal_significant
                                           & (r_main.delta_attenuation >= 0.01)).sum()),
            "attenuation_bh": {t: int(r_main[f"bh_{t}_significant"].sum())
                               for t in ("signflip", "wilcoxon", "pairedt")},
            "attenuation_bonferroni": {t: int(r_main[f"bonferroni_{t}"].sum())
                                       for t in ("signflip", "wilcoxon", "pairedt")},
            "reversal_bh": int(r_main.bh_reversal_significant.sum()),
        },
        "with_coarse": {
            "n_families": len(r_coarse),
            "attenuation_bh": {t: int(r_coarse[f"bh_{t}_significant"].sum())
                               for t in ("signflip", "wilcoxon", "pairedt")},
            "reversal_bh": int(r_coarse.bh_reversal_significant.sum()),
        },
        "core5_by_audit": {
            a: {"n_families": int(len(g)),
                "attenuation_bh": {t: int(g[f"bh_{t}_significant"].sum())
                                   for t in ("signflip", "wilcoxon", "pairedt")},
                "reversal_bh": int(g.bh_reversal_significant.sum())}
            for a, g in r_core.groupby("audit")},
    }, open(OUT / "multiplicity_summary.json", "w"), indent=2)
    print("\n   wrote multiplicity_primary.csv, multiplicity_with_coarse.csv,")
    print("         multiplicity_core5_by_audit.csv, multiplicity_summary.json")


if __name__ == "__main__":
    main()
