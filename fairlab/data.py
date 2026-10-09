"""Load real fairness datasets into (X, y, A) where A holds hidden protected attributes.

Protected attributes are NEVER placed in X (demographic-free setting). They are used
only for evaluation (and by oracle baselines, which are clearly labelled).
"""
import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _onehot(df):
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return pd.get_dummies(df, columns=cat, drop_first=False, dtype=float).astype(float)


def adult():
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital", "occupation",
            "relationship", "race", "sex", "cap_gain", "cap_loss", "hours", "country", "income"]
    d = pd.read_csv(f"{D}/adult-all.csv", header=None, names=cols, na_values="?").dropna()
    y = (d.income.str.strip().str.startswith(">50K")).astype(int).values
    A = pd.DataFrame({"sex": (d.sex == "Female").astype(int).values,
                      "race": (d.race != "White").astype(int).values})
    # relationship encodes Husband/Wife -> direct sex proxy; kept, as in practice
    X = d.drop(columns=["income", "sex", "race", "fnlwgt", "education"])
    return _onehot(X), y, A


def compas():
    d = pd.read_csv(f"{D}/compas-scores-two-years_clean.csv")
    d = d[d.race.isin(["African-American", "Caucasian"])]
    # positive outcome = no recidivism within two years (favourable)
    y = (1 - d.two_year_recid).astype(int).values
    A = pd.DataFrame({"race": (d.race == "African-American").astype(int).values,
                      "sex": (d.sex == "Female").astype(int).values})
    X = d[["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "c_charge_degree"]].copy()
    return _onehot(X), y, A


def german():
    d = pd.read_csv(f"{D}/german_data_credit.csv")
    y = d["class-label"].astype(int).values
    A = pd.DataFrame({"sex": (d.sex == "female").astype(int).values,
                      "age": (d.age < 25).astype(int).values})
    X = d.drop(columns=["class-label", "sex", "age", "marital-status"])
    return _onehot(X), y, A


def law():
    d = pd.read_csv(f"{D}/law_school_clean.csv").dropna()
    y = d.pass_bar.astype(int).values
    A = pd.DataFrame({"race": (d.race == "Non-White").astype(int).values,
                      "sex": (d.male == 0).astype(int).values})
    X = d.drop(columns=["pass_bar", "race", "male"])
    return _onehot(X), y, A


def bank():
    d = pd.read_csv(f"{D}/bank-full.csv")
    y = (d.y == "yes").astype(int).values
    A = pd.DataFrame({"age": (d.age < 30).astype(int).values,
                      "marital": (d.marital == "single").astype(int).values})
    X = d.drop(columns=["y", "age", "marital", "duration"])  # duration leaks the label
    return _onehot(X), y, A


def credit():
    d = pd.read_csv(f"{D}/credit-card-clients.csv")
    y = (1 - d["default payment"]).astype(int).values  # favourable = no default
    A = pd.DataFrame({"sex": (d.SEX == 2).astype(int).values,
                      "age": (d.AGE < 25).astype(int).values})
    X = d.drop(columns=["default payment", "SEX", "AGE", "MARRIAGE"]).copy()
    X["EDUCATION"] = X.EDUCATION.astype(str)
    return _onehot(X), y, A


def dutch():
    d = pd.read_csv(f"{D}/dutch.csv")
    y = d.occupation.astype(int).values
    A = pd.DataFrame({"sex": (d.sex == "female").astype(int).values})
    X = d.drop(columns=["occupation", "sex"]).astype(str)
    X["age"] = d.age.astype(float)
    X["edu_level"] = d.edu_level.astype(float)
    return _onehot(X), y, A


def oulad():
    d = pd.read_csv(f"{D}/oulad_clean.csv")
    y = (d.final_result == "Pass").astype(int).values
    A = pd.DataFrame({"sex": (d.gender == "F").astype(int).values,
                      "disability": (d.disability == "Y").astype(int).values})
    X = d.drop(columns=["final_result", "gender", "disability", "id_student"])
    return _onehot(X), y, A


def diabetes():
    d = pd.read_csv(f"{D}/diabetes-clean.csv")
    d = d[d.race.isin(["Caucasian", "AfricanAmerican"])]
    y = (d.readmitted == ">30").astype(int).values  # favourable = not readmitted within 30 days
    A = pd.DataFrame({"race": (d.race == "AfricanAmerican").astype(int).values,
                      "sex": (d.gender == "Female").astype(int).values})
    drop = ["readmitted", "race", "gender", "encounter_id", "patient_nbr", "diag_1", "diag_2", "diag_3"]
    X = d.drop(columns=[c for c in drop if c in d.columns])
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if c in X:
            X[c] = X[c].astype(str)
    X = X.loc[:, X.nunique() > 1]
    return _onehot(X), y, A


DATASETS = dict(adult=adult, compas=compas, german=german, law=law, bank=bank,
                credit=credit, dutch=dutch, oulad=oulad, diabetes=diabetes)


def load(name):
    X, y, A = DATASETS[name]()
    X = X.reset_index(drop=True)
    A = A.reset_index(drop=True)
    assert len(X) == len(y) == len(A)
    return X, np.asarray(y), A


if __name__ == "__main__":
    for n in DATASETS:
        X, y, A = load(n)
        print(f"{n:9s} n={len(y):6d} d={X.shape[1]:4d} P(y=1)={y.mean():.3f} " +
              " ".join(f"{c}:{A[c].mean():.2f}" for c in A))


# ---------------------------------------------------------------- v2 additions
CAP = 12000  # rows per dataset (stratified by label x first attribute) to keep 30-seed runs tractable

TARGETS = [("law", "race"), ("adult", "sex"), ("credit", "age"), ("oulad", "disability"),
           ("compas", "race"), ("compas", "sex"), ("dutch", "sex"), ("german", "age")]
ALL_PAIRS = TARGETS + [("adult", "race"), ("german", "sex"), ("law", "sex"), ("credit", "sex"),
                       ("oulad", "sex"), ("bank", "age"), ("bank", "marital"),
                       ("diabetes", "race"), ("diabetes", "sex")]


def load_capped(name, cap=CAP, seed=12345):
    X, y, A = _load_any(name)
    if len(y) > cap:
        rng = np.random.default_rng(seed)
        strata = y * 2 + A.iloc[:, 0].values
        keep = []
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            k = int(round(cap * len(ix) / len(y)))
            keep.append(rng.choice(ix, k, replace=False))
        keep = np.sort(np.concatenate(keep))
        X, y, A = X.iloc[keep].reset_index(drop=True), y[keep], A.iloc[keep].reset_index(drop=True)
    X = X.loc[:, X.std() > 0]
    return X, y, A


def load_acs(state="CA", year=2018, task="ACSIncome", root=None):
    """ACS PUMS via folktables (pip install folktables). Downloads once into data/acs/ on first use.
    Returns X, y, A with A columns 'race' (1 = not white alone) and 'sex' (1 = female).
    High-cardinality codes are coarsened: occupation to its major group (code // 1000),
    place of birth to US-born vs abroad region (code // 100)."""
    import folktables as ft
    root = root or os.path.join(D, "acs")
    ds = ft.ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=root)
    df = ds.get_data(states=[state], download=True)
    T = getattr(ft, task)
    Xa, y, _ = T.df_to_pandas(df)
    A = pd.DataFrame({"race": (Xa["RAC1P"] != 1).astype(int).values, "sex": (Xa["SEX"] == 2).astype(int).values})
    X = Xa.drop(columns=[c for c in ("RAC1P", "SEX") if c in Xa]).copy()
    if "OCCP" in X:
        X["OCCP"] = (X["OCCP"] // 1000).astype(int)
    if "POBP" in X:
        X["POBP"] = np.where(X["POBP"] < 100, 0, X["POBP"] // 100).astype(int)
    cat = [c for c in ("COW", "MAR", "OCCP", "POBP", "RELP", "SCHL", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC",
                       "NATIVITY", "DEAR", "DEYE", "DREM") if c in X]
    num = [c for c in X if c not in cat]
    X = pd.get_dummies(X.astype({c: str for c in cat}), columns=cat, dtype=float).astype(float)
    for c in num:
        X[c] = X[c].astype(float)
    return X.reset_index(drop=True), np.asarray(y).ravel().astype(int), A


def _load_any(name):
    """'acs_CA_2018_ACSIncome' style names go to folktables; everything else to the bundled CSVs."""
    if name.startswith("acs_"):
        _, state, year, task = name.split("_")
        return load_acs(state, int(year), task)
    return load(name)
