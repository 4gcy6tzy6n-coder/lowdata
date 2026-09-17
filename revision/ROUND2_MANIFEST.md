# Round-2 deliverable manifest

Generated 2026-09-17T01:39:38Z from results/revision/round2/.
Schema/version: canonical-round2-v1  -  see canonical_archive/ARCHIVE_PROVENANCE.md

These are the authoritative copies.

Coverage schema (ROUND2_RESULTS.md 5.1): the ambiguous
noisy_coverage/clean_coverage names are gone from every audit.  All four audits
carry all four canonical columns: common_noisy_coverage, common_clean_coverage,
matched_noisy_coverage, matched_clean_coverage.  C100N-human's common pair was
backfilled on the server by revision/run_backfill_c100n_common.py, which needs
the prepared cache.

Analysis set (ROUND2_RESULTS.md 5.2): primary = EMA Loss, Confidence, AUM, CL,
Combined, Combined-noN; forgetting = coarse_secondary; neighbor = excluded
(11-18 distinct score values, pairwise AUROC exactly 0.5000 in all audits).

Multiplicity (ROUND2_RESULTS.md 5.3): exact one-sided sign-flip, Wilcoxon and
paired t; BH applied within the reported family set; signed effect floor
delta >= 0.01 AUROC.  Headline: 28/40 BH-significant attenuation families,
18/40 after the effect floor, 3/40 reversal.

```
725eb584138bfc6c89a8acb37880b3f8ec9f00faaa132fcaeb7de03ac946e2a8  _j1_for_analysis.csv
1a343e60686dffd2e20d42f3c4c88fd221b38adda3e82cc1041a02c34a9c7823  coverage_schema.csv
8ade91772967218c3ff7549f5c5a515e38a6765084bfabf1f6234288e1fb1441  e10_weak_detector_account.csv
8576870a1857712191c937ad786553a5f838ead4a43986d49986327039ef6f0e  e10_weak_detector_account_by_strength.csv
34f5980134bed7ecd4d6bfc99453d4672bca5fc8010f71802845d8bf29ae4702  e11_variance_decomposition.csv
32d2028003e70719cbe6dc098cb15f5ac8870efe57dce0f55b441de32ddb6f44  e11_variance_decomposition_by_cell.csv
36cee7cb752fc2117094676e58385075c30b81631459eca31255a18ec3d38f6c  e11_variance_decomposition_by_strength.csv
a4065660f95c9c44d236ddd19df5fcc2edf087583a98e7403f66b07f83a0215a  e11_variance_decomposition_null.csv
daf9cb3439f73ba12d19d1828640e08e2aa25baec80cb3f720354ae269d4aa2d  e1_combined_noN.csv
99469ead3d75128a5408fe073cf1fd30e5bbe26f709f674afe0b837297b3e8b2  e1_summary.csv
be8ecae41c1aebacfd79de794189640659ba2dc1e8a20593c6b4156668a32e2c  e2e3_order_replacement.csv
5133cb3f4fc3e4cf686415fc031c19acb025d200af83362fc00f9c100157ead0  e2e3_summary.csv
4a38e87915f0c8988a936fff9e49e0903cbc704691e618c64eec49f141950fbf  e4_a40_audit.csv
1c12b5f751a6b52ac0e955a27219aaf15ab2f698181f50ba3d7eefc39ff802b0  e4_summary.csv
9abfc7d7275c58cb702dd6ee4edcd6688588203a3ffe962bae79bb212c55719c  e6_crossing.csv
ba925319043c052d2fc510e0a0bccda458715ca934ac58a575ba0bbda4a89716  e6_crossing_lam5_n51_g0.25-0.5-1.0-1.5.csv
e1790971b63fa08d3b8fca5c8d8298212ac57b28c85bc95c27eaec626c188369  e6_simulation.csv
f67c4e86d4843201392f0db1e1d76b0bdd4e4d4edac4b951e092b744dcd024b7  e6_simulation_lam5_n51_g0.25-0.5-1.0-1.5.csv
35bf23afe1fb25c68597f903ef61e6e32d95bb433f5e7a762b88fb3ee91db45b  e7_detector_strength.csv
83d5bb2059a77d3c665b93be0113ef0d464b88b3c69baf11ce1c4938176d3611  e8_mutual_information_a40.csv
4670ad55b9ec6f954e83e35ed66232e0e17693a0666b03410df8b328f408b31e  e8_mutual_information_a40_summary.csv
0a13702a50eea5727a69902899a2088cfb812d6169d5b1ae7ed51e5d1f5b9f6d  e8_mutual_information_s20.csv
1163ab9393e9cc12a47c4c4bf2ace850e49766e3d1d18940ef3da3228f8b5b6b  e8_mutual_information_s20_summary.csv
482cdfd321b74fd7af1bd6bb340f20dfd97d2f10aa6047dbabdea3a8748991aa  e9_c100n_audit.csv
e23dfffe4377abe992b4139660170e59af0c978516b8cbf28456a89086ac7841  e9_c100n_summary.csv
bd6b1afd308d967a4d44eeaad9abefec21fb876a1da25945113ffc471ca7b10a  final_analysis_set.csv
094b70c929efeb0e2abec29fc4bda4fe231e1564e452d05be5dcabf5ed00fa1f  granularity_audit.csv
27d97216809ec5f87b07f35681da6400f02c007c7317537fb22e3e2c8d5ab0a5  granularity_audit_summary.csv
9eecefb9f7ac9fd9e508792272123195425b1d74fc3e202dcdc3ffd07c35d8f1  multiplicity_core5_by_audit.csv
1e570c98972d28c9f16834553fc9b0c9c5cf7af65b6b58ce6ff8de9fdd10ede7  multiplicity_core5_coupled_only.csv
aea19f9f4a92244a082b3f3c799c4f3427cc0043c37cb22c82cb2998a82a3922  multiplicity_primary.csv
e9a740da398e822229f5cbbeebbb20e22da77a8308aef195f7333887c0dcbd21  multiplicity_summary.json
29bba276578bf14211bdc8dc12e5a876065046135b26745a1e9f35c70527d58d  multiplicity_with_coarse.csv
```

## Driver scripts

ba38e657df9b4fec3dd473dc51d390240d6c9dd94cc4fd0100b417f99b487a2f  revision/run_backfill_c100n_common.py
4668d5427056f4ef9685ede7f3a132c098121ac936348b23841f0e1122b75d3d  revision/run_backfill_coverage.py
3fe0176c77e70b728a23d70c856abf867cf4be14f8f079216d9d3a27aab22eab  revision/run_e10_weak_detector.py
7814f45f0c3b952406613bf3eb9cd416e61821babe6237d341e2c23bf382a158  revision/run_e11_variance.py
b3c32713cfd7121bbcfad9cd55a6f3d6aba4db85cfa50efd87bfe3ad4e41443c  revision/run_e8_parallel.py
e02909d73a443dbf910b9ce9948bd40961baf156afd82741dac0a132456a6b0d  revision/run_e9_c100n.py
2ee3c3999928891cbc891b7ff13f3be393c1fb44a6839c3f9ba3cb363e16427a  revision/run_final_analysis_set.py
ee729c9812c217f641b38e97f4e675a67e489ec9d98ffa2e5d6bb0afaa4d1f75  revision/run_granularity_audit.py
f10997da25359ea93300472004b70ce68ab34b64718e5cd1980616d639282bc1  revision/run_round2.py

## Archived on the AutoDL persistent disk

```
/root/autodl-tmp/lowdata_backup/canonical-round2-v1.tar.gz   <- AUTHORITATIVE
    950,948 bytes
    sha256 034c784973612bf9557fdfee0b625543716ea0fbd60c37da9e65fc9fdbff4468
    git commit 1acaa3c   schema canonical-round2-v1
    contains ARCHIVE_PROVENANCE.md, README.md, MANIFEST.sha256.txt,
             results/revision/round2/ (unified audit CSVs + all derived tables),
             results/revision/p0_batch/j1_bootstrap_all_proxies.csv,
             and the ten driver scripts
    self-checked by `python revision/verify_canonical.py --root .` from an unpack

/root/autodl-tmp/lowdata_backup/round2_results.tar.gz        <- SUPERSEDED
    241,699 bytes
    sha256 8898e913f4a73b5fee9380c9dda1a90a33f0dd51dc8182d8a5d6d47610d4e497
    PRE-CANONICALIZATION snapshot; its limits are listed in the sidecar file
    round2_results.tar.gz.SUPERSEDED.  Do not use it for reproduction.

/root/autodl-tmp/lowdata_backup/c100s20_traces.tar.gz
    3,869,261,703 bytes (C100-S20 training traces, 10 runs x 200 epochs)
```
