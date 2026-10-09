# Pre-registration: confirmatory study (frozen 8 Oct 2026, before seeds 0-29 of E2, E4b, E6 were run)

I wrote this after the development pilot (seeds 100-105) and before running any confirmatory seed of
experiments E2, E4b and E6. E1 (audit ranking) was run on seeds 0-29 before this file was written; it
replicates an exploratory finding from the first pilot and is reported as such, not as a confirmatory test.

## Frozen design
- Data: 8 target dataset-attribute pairs (Law-race, Adult-sex, Credit-age, OULAD-disability, COMPAS-race,
  COMPAS-sex, Dutch-sex, German-age); 9 further pairs for E4b. At most 12,000 rows per dataset (stratified).
- Splits per seed: train 50% / labelling pool 20% / test 30%, stratified by label x attribute. Seeds 0-29.
- Protected attribute never a model input. Revealed only for selected pool rows (proxy arms) or all training
  rows (oracle arm, a reference, not a deployable method).
- Learner: logistic regression with a soft equalized-odds penalty, l2 = 1e-3, penalty strengths
  {0.3, 1, 3, 10, 30}. Neural check (E6): one-hidden-layer MLP (32 tanh units), strengths {1, 3, 10}.
- Proxy: logistic regression P(A|X) fitted on the labelled rows; soft probabilities enter the penalty.
- Label budgets n in {25, 50, 100, 200, 400} (capped at half the pool).
- Selection strategies: random (primary), outcome-stratified (secondary). Uncertainty, minority-seeking and
  diverse selection were dropped after the pilot (none beat random); their pilot results are reported.
- Disclosure bias (MNAR): random selection where the protected group discloses with 1/3 the probability.
- Comparison methods: ARL, CVaR-DRO, JTT, cluster reweighting (demographic-free, strength sweeps);
  FairDSR-style confident hard proxy and FairRF-style related-feature penalty (both from the same n labels).
- Evaluation: test split, threshold set so the selection rate equals the training base rate.
  Primary metric: equalized-odds gap (max of |TPR gap|, |FPR gap|). Secondary: TPR gap, DP gap, AUC,
  accuracy, worst-group TPR.
- Tuning: T1 (primary, demographic-free): strongest setting whose pool AUC is within 0.01 of the
  unconstrained model; otherwise fall back to the unconstrained model. T2 (secondary): within the same AUC
  budget, minimise the EO gap measured on the labelled pool rows. Sensitivity: budgets 0.005 and 0.02.

## Hypotheses and decision rules
- H1 (main). Random proxy with n = 100 labels, T1: mean EO-gap reduction over the 8 targets exceeds that
  of every demographic-free baseline. Per-target one-sided Wilcoxon signed-rank vs the unconstrained model
  across 30 seeds, Holm-corrected over the 8 targets; report the number of significant targets.
- H2. Random proxy with n = 400, T1, reaches at least 75% of the oracle's mean EO-gap reduction.
- H3. Outcome-stratified selection does not beat random by more than 3 points of mean reduction at any n.
- H4. Disclosure bias (MNAR) lowers the mean reduction relative to random selection at n = 100 and 400.
- H5. T2 does not beat T1 at n <= 100.
- H6 (E4). The agreement cosine between true-group gaps (labelled rows) and proxy-group gaps (pool),
  computable before deployment, predicts harm (EO gap increased by > 0.01 on test): AUROC of
  (-cosine) for harm > 0.70 at n = 100, pooled over 17 pairs x 30 seeds. Guarded rule: apply the proxy
  correction only if cosine > 0 and the labelled-row gap estimate >= 0.05; it must cut the harm rate
  while keeping >= 80% of the unguarded mean reduction.
- H7 (E6). With the MLP, the random proxy at n = 400 has a larger mean reduction than ARL and JTT.
- E5 (descriptive). Error of the naive (labelled rows only) and proxy-weighted (whole pool) estimates of the
  unconstrained model's EO gap, against the test-set truth.

Anything not listed here is exploratory and will be labelled as such.

## Deviations (recorded during the confirmatory runs)
- E4b (9 additional pairs for the backfire diagnostic) was cut from 30 to 10 seeds and its oracle arm was
  dropped, because Adult-race and Diabetes made the 30-seed run take more than 7 hours on the 2-CPU
  machine. The diagnostic itself does not use the oracle. The 8 main pairs contribute all 30 seeds from E2.
- Adult seeds 21-29 of E2 were run in a second process to save time; results are identical in design and
  duplicates (none expected) are dropped on load.
