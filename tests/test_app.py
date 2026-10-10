"""Tests for the web app (app/): it must reproduce the stored 30-seed study and render every page.

Run from the repository root:  pytest -q
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import build_summaries  # noqa: E402
import engine as E  # noqa: E402

RES = os.path.join(ROOT, "results")


@pytest.fixture(scope="module")
def e2():
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "e2_*.csv"))])


@pytest.mark.parametrize("ds,attr,seed,n", [("law", "race", 0, 100), ("compas", "race", 3, 50), ("german", "age", 5, 25)])
def test_fix_reproduces_the_stored_rows(e2, ds, attr, seed, n):
    f = E.fix(E.Run(ds, attr, seed), n)
    paper = e2.query("dataset == @ds and attr == @attr and seed == @seed")
    paper = paper[paper.method.isin(["erm", "oracle", "cluster_rw"]) |
                  ((paper.method == "proxy_soft") & (paper.strategy == "random") & (paper.nlab == n))]
    m = f["raw"].merge(paper.drop_duplicates(["method", "mu"]), on=["method", "mu"], suffixes=("", "_paper"))
    assert len(m) == len(f["raw"]) == 1 + 5 + 5 + 3
    for c in ("eo_gap", "auc", "pool_auc", "acc"):
        assert np.allclose(m[c], m[c + "_paper"], atol=1e-12), c
    p = paper[paper.method == "proxy_soft"].iloc[0]
    for k in ("true_eo", "est_naive", "est_proxy"):
        assert f["measure"][k] == pytest.approx(p[k], abs=1e-12), k


@pytest.mark.parametrize("ds,attr,seed", [("law", "race", 0), ("adult", "sex", 7), ("compas", "sex", 12)])
def test_audit_reproduces_e1(ds, attr, seed):
    a = E.audit(E.Run(ds, attr, seed))
    e1 = pd.read_csv(os.path.join(RES, "e1_audit.csv")).query("dataset == @ds and attr == @attr and seed == @seed")
    m = a.merge(e1, on="group", suffixes=("", "_paper"))
    assert len(m) == len(a) == len(e1)
    assert np.allclose(m.eo_gap, m.eo_gap_paper, atol=1e-12)
    # the score-gap identity behind Finding 1 holds exactly
    assert np.nanmax(np.abs(a.gap_y0 - a.ident_y0)) < 1e-10


def test_label_draw_zero_is_the_papers_and_others_differ():
    run = E.Run("compas", "race", 1)
    d = E.measure_draws(run, 50, k=5)
    assert d.true_eo.nunique() == 1
    assert d.labelled_only.nunique() > 1


def test_saved_tables_are_current():
    """app/data/*.csv must equal what build_summaries.py computes from results/ now."""
    classic = build_summaries.load(build_summaries.CLASSIC)
    c1, c2 = build_summaries.load(build_summaries.CENSUS1), build_summaries.load(build_summaries.CENSUS2)
    fresh = {"audit": build_summaries.audit(), "guard": pd.concat([build_summaries.guard(c1, "US census (round 1)"),
                                                                     build_summaries.guard(c2, "US census (round 2)")]),
             "e5": pd.concat([build_summaries.e5(classic, "Classic datasets"), build_summaries.e5(c1, "US census (round 1)"),
                              build_summaries.e5(c2, "US census (round 2)")])}
    for name, t in fresh.items():
        stored = pd.read_csv(os.path.join(ROOT, "app", "data", name + ".csv"))
        num = stored.select_dtypes("number").columns
        assert np.allclose(stored[num].to_numpy(float), t.reset_index(drop=True)[num].to_numpy(float), rtol=1e-5), name


def test_headline_numbers():
    a = pd.read_csv(os.path.join(ROOT, "app", "data", "audit.csv"))
    assert (a.runs.sum(), a.hidden_first.sum()) == (660, 0)
    g = pd.read_csv(os.path.join(ROOT, "app", "data", "guard.csv")).set_index(["family", "nlab"])
    assert g.loc[("US census (round 1)", 100), "harm"] == pytest.approx(0.26, abs=0.005)
    assert g.loc[("US census (round 1)", 100), "harm_guarded"] == pytest.approx(0.073, abs=0.005)
    h = pd.read_csv(os.path.join(ROOT, "app", "data", "hypotheses.csv")).set_index("id")
    assert h.loc["H1", "result"].startswith("14.5%")


@pytest.fixture(scope="module")
def AppTest():
    return pytest.importorskip("streamlit.testing.v1").AppTest


APP = os.path.join(ROOT, "app", "streamlit_app.py")
PAGES = ["🏠 Overview", "🔍 Find", "📏 Measure", "🔧 Fix", "📊 30-seed results", "📖 Method & limits"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(AppTest, page):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]


def test_small_dataset_caps_the_label_budget(AppTest):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.selectbox(key="pair").set_value(("german", "age"))
    at.sidebar.select_slider(key="n").set_value(400)
    at.sidebar.radio(key="page").set_value("🔧 Fix").run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("at most 100" in c.value for c in at.caption)
