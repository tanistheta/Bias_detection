# Running the ACS (Folktables) experiments on your own computer

## 1. One-time setup (10 minutes)
Install Python 3.10 or newer, unzip fairlab.zip, and open a terminal in the `fairlab` folder.

Windows (PowerShell):
    py -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

macOS / Linux:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## 2. Check the download works (first run downloads ~100-500 MB per state into data/acs/)
    python -c "from fairlab.data import load_capped; X,y,A=load_capped('acs_CA_2018_ACSIncome'); print(X.shape, y.mean(), A.mean())"
You should see something like (12000, ~70) and the share of people earning over $50k.

## 3. Run the main experiment
Dataset names follow `acs_<STATE>_<YEAR>_<TASK>`; attribute is `race` or `sex`.
Start with a quick 3-seed smoke run, then the real 30-seed run:

    python scripts/exp_main.py 0 3  results/acs_smoke.csv acs_CA_2018_ACSIncome:race
    python scripts/exp_main.py 0 30 results/acs_a.csv acs_CA_2018_ACSIncome:race,acs_TX_2018_ACSIncome:race,acs_NY_2018_ACSIncome:sex
    python scripts/exp_main.py 0 30 results/acs_b.csv acs_FL_2018_ACSIncome:race,acs_PA_2018_ACSEmployment:sex

Running two terminals at once (acs_a and acs_b) uses two CPU cores. Expect roughly 4 to 8 minutes per
seed per target on a laptop, so a 30-seed target takes a few hours: start it before bed.
Results are saved after every seed, so stopping early loses nothing.

## 4. Analyse
Delete results/acs_smoke.csv first, then:
    python scripts/analyze_final.py "acs_*.csv"
This prints the same tables as the paper draft (gap reductions, significance, hypotheses, gap-estimation
error) for the ACS targets, and writes results/final_summary_acs_*.csv.

## 5. Pre-register before the real run
Before step 3's 30-seed run, add a short dated section to PREREGISTRATION.md listing the states, tasks and
attributes, and the fixed gap-guard threshold (0.08) you will test. That is what makes the ACS run
confirmatory rather than exploratory.

## Notes
- Occupation codes are coarsened to major groups and place of birth to US-born vs abroad region
  (see `load_acs`), to keep the number of columns manageable.
- If a download fails, delete the partial folder under data/acs/ and rerun.
