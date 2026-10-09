# fairlab: fairness with few demographic labels

Pre-registered study of bias auditing and mitigation when the protected attribute is unavailable or
available for only a few rows. Real data, 8 main dataset-attribute pairs (17 for audit/backfire analyses).

## Layout
- `fairlab/data.py` loaders (9 datasets, 12k-row cap), `load_acs` for Folktables (run locally)
- `fairlab/learn.py` penalised logistic regression and MLP (soft equalized-odds penalty), ARL with MLP learner
- `fairlab/proxy.py` labelling strategies (random, outcome-stratified, diverse, uncertainty, minority) and proxy fit
- `fairlab/baselines.py` JTT, CVaR-DRO, linear ARL, cluster reweighting
- `fairlab/analysis.py` tuning rules (T1 demographic-free, T2 label-informed), summaries, Holm
- `scripts/exp_audit.py` E1 audit ranking and identity check
- `scripts/exp_main.py` E2/E3/E5 label budget x strategy x method (+ demographic-free baselines)
- `scripts/exp_diag.py` E4 backfire diagnostic on extra pairs
- `scripts/exp_mlp.py` E6 neural check
- `scripts/analyze_final.py`, `analyze_e4.py`, `analyze_e4_explore.py`, `curve_data.py` regenerate every table
- `PREREGISTRATION.md` frozen design, hypotheses, logged deviations
- `results/` raw per-run CSVs (pilot_*: dev seeds 100-105; e2_*, e4b, e6_*: confirmatory)

## Reproduce
    pip install numpy pandas scipy scikit-learn autograd
    # data/ must hold the CSVs (see fairlab/data.py for file names and sources)
    python scripts/exp_audit.py 0 30 results/e1_audit.csv
    python scripts/exp_main.py 0 30 results/e2_x.csv law:race,adult:sex,credit:age,oulad:disability,compas:race,compas:sex,dutch:sex,german:age
    python scripts/exp_diag.py 0 10 results/e4b.csv adult:race,german:sex,law:sex,credit:sex,oulad:sex,bank:age,bank:marital,diabetes:race,diabetes:sex
    python scripts/exp_mlp.py 0 3 results/e6_x.csv <targets>
    python scripts/analyze_final.py && python scripts/analyze_e4.py

Runtime on 2 CPUs: about 6 hours for E2 (Adult dominates).
