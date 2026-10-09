"""Hidden-Group Bias Lab: interactive working model of "Measuring Hidden-Group Bias with Few Demographic Labels".

Run from the repository root:  streamlit run app/streamlit_app.py
Every live page reruns the paper's own code (fairlab + scripts/exp_*.py) for one seed; seeds 0-29 reproduce
the stored 30-seed study exactly. The saved-results page reads app/data/*.csv (built by app/build_summaries.py).
"""
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

st.set_page_config(page_title="Hidden-Group Bias Lab", page_icon="⚖️", layout="wide")
DATA = os.path.join(HERE, "data")
REPO_URL = "https://github.com/tanistheta/Bias_detection"


# ===================================================================== look
def colors():
    """Validated categorical slots (light / dark steps of the same hues) plus a recessive gray."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:  # older Streamlit or no browser context (tests)
        dark = False
    if dark:
        return dict(proxy="#3987e5", naive="#d95926", oracle="#199e70", base="#8f8e88", ink="#c3c2b7")
    return dict(proxy="#2a78d6", naive="#eb6834", oracle="#1baf7a", base="#9a9893", ink="#52514e")


def bars(df, x, y, color_field, domain, rng, x_title, height, fmt=".3f", sort=None, rule=None):
    """Horizontal bars with rounded data ends, a value label at each end and a hover tooltip."""
    lo, hi = min(0.0, float(df[x].min())), max(0.0, float(df[x].max()))
    pad = 0.12 * (hi - lo or 1)
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", title=None, sort=sort or "-x", axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[lo - (pad if lo < 0 else 0), hi + pad])),
        tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)],
    )
    marks = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        color=alt.Color(f"{color_field}:N", scale=alt.Scale(domain=domain, range=rng),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=320)))
    ink = colors()["ink"]
    pos = base.transform_filter(f"datum['{x}'] >= 0").mark_text(align="left", dx=4, fontSize=12, color=ink)
    neg = base.transform_filter(f"datum['{x}'] < 0").mark_text(align="right", dx=-4, fontSize=12, color=ink)
    layers = [marks, pos.encode(text=alt.Text(f"{x}:Q", format=fmt)), neg.encode(text=alt.Text(f"{x}:Q", format=fmt))]
    if rule is not None:
        layers.append(alt.Chart(pd.DataFrame({"v": [rule]})).mark_rule(color=ink).encode(x="v:Q"))
    return alt.layer(*layers).properties(height=height)


def pct(v):
    return "–" if pd.isna(v) else f"{v:+.1f}%"


def tuned(row, label="μ = {:g}"):
    """T1 falls back to the original model (mu 0) when every setting costs more than the AUC budget."""
    return "fell back to the original model" if row.mu == 0 else label.format(row.mu)


# ===================================================================== cached computation
@st.cache_resource(max_entries=6, show_spinner=False)
def get_run(ds, attr, seed):
    return E.Run(ds, attr, seed)


@st.cache_data(max_entries=32, show_spinner=False)
def get_audit(ds, attr, seed):
    return E.audit(get_run(ds, attr, seed))


@st.cache_data(max_entries=32, show_spinner=False)
def get_draws(ds, attr, seed, n, k=30):
    return E.measure_draws(get_run(ds, attr, seed), n, k)


@st.cache_data(max_entries=32, show_spinner=False)
def get_fix(ds, attr, seed, n):
    f = E.fix(get_run(ds, attr, seed), n)
    f["measure"] = {k: v for k, v in f["measure"].items() if k not in ("lab", "q")}
    return f


@st.cache_data(show_spinner=False)
def saved(name):
    return pd.read_csv(os.path.join(DATA, name + ".csv"))


# ===================================================================== sidebar
PAIRS = E.TARGETS + [p for p in E.ALL_PAIRS if p not in E.TARGETS]
PAGES = ["🏠 Overview", "🔍 Find", "📏 Measure", "🔧 Fix", "📊 30-seed results", "📖 Method & limits"]

with st.sidebar:
    st.markdown("### ⚖️ Hidden-Group Bias Lab")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="page")
    st.divider()
    st.markdown("**Live experiment**")
    pair = st.selectbox("Dataset and hidden group", PAIRS, index=PAIRS.index(("adult", "sex")),
                        format_func=lambda p: f"{E.LABELS[p[0]]} · {p[1]}", key="pair")
    seed = st.number_input("Seed (random split)", 0, 999, 0, 1, key="seed",
                           help="Seeds 0-29 are the paper's confirmatory runs; the app reproduces them exactly.")
    n_req = st.select_slider("People who reveal their group", E.NS, 100, key="n",
                             help="How many people in the labelling pool reveal their protected group.")
    st.divider()
    st.caption("Chirag Joshi · Soumya Shradha · Tanishk Gangwar · Vivansh Garg · Yagya Salwan  \n"
               "Department of Computer Science and Engineering, Manipal University Jaipur")

DS, ATTR = pair
GROUP = E.GROUPS[pair]
C = colors()


def run_and_budget():
    with st.spinner("Loading the data and training the original model…"):
        run = get_run(DS, ATTR, int(seed))
    budgets = run.budgets()
    n = n_req if n_req in budgets else max(budgets)
    if n != n_req:
        st.caption(f"This dataset's labelling pool allows at most {max(budgets)} revealed labels, so {n} is used.")
    return run, n


def setting_line(run, n=None):
    s = (f"**{E.LABELS[DS]}** · {run.n_rows:,} people · {run.n_features} model inputs · hidden group: **{GROUP}** "
         f"({run.group_share:.0%}) · seed {int(seed)}")
    st.caption(s + (f" · {n} people reveal their group" if n else ""))


# ===================================================================== pages
def page_overview():
    st.title("Measuring hidden-group bias with few demographic labels")
    st.markdown("**Why demographic-free audits fail, and when proxy correction helps.**")
    st.markdown("AI models help decide who gets a loan, a job interview or bail. To check whether a model treats "
                "a group such as women or a racial minority worse, you normally need to know who belongs to which "
                "group, and that information is often never collected. This app reruns the study's experiments "
                "live, with the group column hidden from every model like an answer key.")
    a = saved("audit")
    fx = saved("fix")
    mean = fx[fx.family == "Classic datasets"].groupby("arm").eo_red.mean()
    df_best = mean[["arl", "cvar", "jtt", "cluster_rw"]].max()
    c = st.columns(3)
    with c[0]:
        st.subheader("🔍 Find it?")
        st.markdown(f"**No.** Without the group column, an audit never ranked the hidden group first: "
                    f"**{a.hidden_first.sum()} of {a.runs.sum()}** audits.")
    with c[1]:
        st.subheader("📏 Measure it?")
        st.markdown("**Yes, with about 100 people.** A small *guesser* trained on 100 revealed labels measures the "
                    "gap far more accurately than looking at those 100 people alone.")
    with c[2]:
        st.subheader("🔧 Fix it?")
        st.markdown(f"**Only when the gap is clearly big.** With 100 labels the correction removed "
                    f"**{mean['proxy[random] n=100']:.1f}%** of the gap on average, against at most "
                    f"{df_best:.1f}% for methods that use no demographic data. A safety rule (gap ≥ 0.08) "
                    "prevents most harmful fixes.")
    st.divider()
    st.markdown("#### How to use this app")
    st.markdown(
        "1. Pick a **dataset and hidden group** in the sidebar (for example Adult income, with sex hidden).\n"
        "2. **🔍 Find** ranks every group an auditor could form from the model's inputs and shows where the hidden "
        "group lands.\n"
        "3. **📏 Measure** lets a few people reveal their group and compares two ways of estimating the bias with "
        "the true value.\n"
        "4. **🔧 Fix** corrects the model with those few labels, applies the safety rule and grades the result "
        "with the answer key.\n"
        "5. **📊 30-seed results** shows the full study: 9 classic datasets and 10 US states, 30 repeats each.")
    st.info("Seeds 0–29 are the study's confirmatory runs: for those seeds the live pages reproduce the stored "
            "results exactly. The answer key (the real group) is used only to grade, never to train or tune.")


def page_find():
    st.header("🔍 Find: can an audit spot the hidden group?")
    st.markdown("An auditor without demographic data can only split people by what the model sees, for example "
                "*income above the median* or *has a past conviction*, and look for the split the model treats "
                "most unequally. Here every such split competes with the real hidden group.")
    run, _ = run_and_budget()
    setting_line(run)
    with st.spinner("Auditing every candidate group…"):
        a = get_audit(DS, ATTR, int(seed))
    hid = a[a.is_A].iloc[0]
    top = a.iloc[0]
    c = st.columns(3)
    c[0].metric("Hidden group's rank", f"{int(hid['rank'])} of {len(a)}")
    c[1].metric(f"Gap for {GROUP}", f"{hid.eo_gap:.3f}", help="Equalized-odds gap of the original model.")
    c[2].metric("Gap of the top-ranked group", f"{top.eo_gap:.3f}",
                help="The group an auditor without demographic data would investigate first.")
    if hid["rank"] == 1:
        st.success("In this run the hidden group ranked first. Across the 30-seed study this never happened.")
    else:
        st.warning(f"The audit would point at **{top.label}** first; the hidden group comes "
                   f"{int(hid['rank'])}th. The model's own inputs always win.")
    d = a.assign(label=np.where(a.is_A, f"Hidden group: {GROUP}", a.label),
                 kind=np.where(a.is_A, "Hidden group", "Group formed from a model input"))
    d = d.head(15) if hid["rank"] <= 15 else pd.concat([d.head(14), d[d.is_A]])
    st.altair_chart(bars(d, "eo_gap", "label", "kind", ["Hidden group", "Group formed from a model input"],
                         [C["proxy"], C["base"]], "Equalized-odds gap of the original model (higher = treated more "
                         "unequally)", 34 * len(d) + 70), use_container_width=True)
    worst = np.nanmax(np.abs(np.r_[a.gap_y0 - a.ident_y0, a.gap_y1 - a.ident_y1]))
    st.markdown(f"**Why:** for any group, the score gap equals σ·ρ/√(π(1−π)), where ρ is how closely the model's "
                f"score follows that group. A model that never saw **{ATTR}** follows it only indirectly, so its own "
                f"inputs always correlate more. In this run the identity holds to {worst:.0e}.")
    with st.expander("Table view"):
        t = a.assign(group=np.where(a.is_A, f"HIDDEN: {GROUP}", a.label))[
            ["rank", "group", "eo_gap", "rho_y0", "rho_y1", "reliance", "frac"]]
        t.columns = ["Rank", "Group", "EO gap", "Score correlation (y=0)", "Score correlation (y=1)",
                     "Model reliance", "Share of people"]
        st.dataframe(t.round(3), hide_index=True, width="stretch")
    s = saved("audit")
    row = s[s.target == f"{DS}-{ATTR}"]
    if len(row):
        r = row.iloc[0]
        st.caption(f"30-seed study for this pair: hidden group ranked first in {r.hidden_first} of {r.runs} runs "
                   f"(median rank {r.median_rank:.0f} of {r.candidates}).")


def page_measure():
    st.header("📏 Measure: how much bias, with only a few labels?")
    st.markdown("A few people reveal their group. You can measure the model's gap **only among them**, or train a "
                "small *guesser* on them that estimates everyone's group and measure the gap on the **whole pool**. "
                "The true gap (answer key) is measured on the test split.")
    run, n = run_and_budget()
    setting_line(run, n)
    with st.spinner("Drawing 30 different sets of people…"):
        d = get_draws(DS, ATTR, int(seed), n)
    r0 = d.iloc[0]
    c = st.columns(3)
    c[0].metric("True gap (answer key)", f"{r0.true_eo:.3f}")
    c[1].metric(f"Only the {n} labelled people", f"{r0.labelled_only:.3f}",
                f"error {abs(r0.labelled_only - r0.true_eo):.3f}", delta_color="off")
    c[2].metric("Guesser on the whole pool", f"{r0.proxy:.3f}", f"error {abs(r0.proxy - r0.true_eo):.3f}",
                delta_color="off")
    long = d.melt(id_vars=["draw", "true_eo"], value_vars=["labelled_only", "proxy"], var_name="m", value_name="est")
    dom = [f"Only the {n} labelled people", "Guesser on the whole pool"]
    long["Method"] = long.m.map({"labelled_only": dom[0], "proxy": dom[1]})
    # one row per method (1 = labelled only, 0 = guesser), dots spread vertically so overlapping draws stay visible
    long["row"] = np.where(long.m == "labelled_only", 1.0, 0.0) + ((long.draw % 6) - 2.5) * 0.07
    pts = alt.Chart(long).mark_circle(size=90, opacity=0.85, stroke="white", strokeWidth=1).encode(
        x=alt.X("est:Q", title="Estimated gap of the original model (each dot = a different set of people)"),
        y=alt.Y("row:Q", title=None, scale=alt.Scale(domain=[-0.5, 1.5]),
                axis=alt.Axis(values=[0, 1], grid=False, ticks=False, domain=False, labelLimit=260,
                              labelExpr=f"datum.value == 1 ? '{dom[0]}' : '{dom[1]}'")),
        color=alt.Color("Method:N", scale=alt.Scale(domain=dom, range=[C["naive"], C["proxy"]]),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=320)),
        tooltip=[alt.Tooltip("Method:N"), alt.Tooltip("draw:Q", title="draw"), alt.Tooltip("est:Q", format=".3f")],
    )
    truth = pd.DataFrame({"v": [r0.true_eo], "l": [f"true gap {r0.true_eo:.3f}"]})
    rule = alt.Chart(truth).mark_rule(strokeDash=[4, 3], strokeWidth=2, color=C["ink"]).encode(x="v:Q")
    lab = alt.Chart(truth).mark_text(align="left", dx=5, dy=8, color=C["ink"], fontSize=12).encode(
        x="v:Q", y=alt.value(0), text="l:N")
    chart = alt.layer(rule, pts, lab).properties(height=230)
    st.altair_chart(chart, use_container_width=True)
    e_n = (d.labelled_only - d.true_eo).abs().mean()
    e_p = (d.proxy - d.true_eo).abs().mean()
    st.markdown(f"Over 30 different sets of {n} people, the average error is **{e_n:.3f}** using only the labelled "
                f"people and **{e_p:.3f}** with the guesser ({e_n / e_p:.1f}× smaller). Small samples are noisy, and "
                "noise always looks like a gap, so the labelled-only estimate overstates the bias.")
    st.caption("Draw 0 is the set of people the study used for this seed; draws 1–29 are new random sets.")
    e5 = saved("e5")
    long5 = e5.melt(id_vars=["family", "nlab"], value_vars=["err_labelled", "err_proxy"], var_name="m", value_name="err")
    long5["Method"] = long5.m.map({"err_labelled": "Only the labelled people", "err_proxy": "Guesser on the whole pool"})
    dom5 = ["Only the labelled people", "Guesser on the whole pool"]
    st.markdown("#### In the 30-seed study")
    cols = st.columns(long5.family.nunique())
    for col, (fam, g) in zip(cols, long5.groupby("family", sort=False)):
        line = alt.Chart(g).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2).encode(
            x=alt.X("nlab:Q", title="People who revealed their group", scale=alt.Scale(type="log"),
                    axis=alt.Axis(values=E.NS)),
            y=alt.Y("err:Q", title="Mean absolute error of the estimate", scale=alt.Scale(domain=[0, 0.3])),
            color=alt.Color("Method:N", scale=alt.Scale(domain=dom5, range=[C["naive"], C["proxy"]]),
                            legend=alt.Legend(orient="bottom", title=None, labelLimit=320)),
            tooltip=["Method:N", "nlab:Q", alt.Tooltip("err:Q", format=".3f")],
        ).properties(height=240, title=fam)
        col.altair_chart(line, use_container_width=True)
    with st.expander("Table view"):
        t = e5.copy()
        t.columns = ["Data", "People labelled", "Error: labelled only", "Error: guesser", "Bias: labelled only",
                     "Bias: guesser"]
        st.dataframe(t.round(3), hide_index=True, width="stretch")


def page_fix():
    st.header("🔧 Fix: correct the model with those few labels")
    st.markdown(f"The guesser's estimates of who belongs to the group feed a fairness penalty while the model is "
                f"retrained. The penalty strength is tuned **without** demographic data (strongest setting that "
                f"costs at most {E.BUDGET} AUC on the pool). The **safety rule** applies the correction only when "
                f"the measured gap is at least **{E.GUARD}**.")
    run, n = run_and_budget()
    setting_line(run, n)
    with st.spinner("Training the original, corrected, comparison and oracle models (about 15 fits)…"):
        f = get_fix(DS, ATTR, int(seed), n)
    m, s, erm = f["measure"], f["sel"], f["erm"]
    if f["applied"]:
        st.success(f"Safety rule: the guesser measures a gap of **{m['est_proxy']:.3f} ≥ {E.GUARD}**, so the "
                   "correction is applied.")
    else:
        st.info(f"Safety rule: the guesser measures a gap of **{m['est_proxy']:.3f} < {E.GUARD}**, so the original "
                "model is kept and the measurement is reported instead.")
    px = s.loc["proxy_soft"]
    if f["harm"]:
        st.warning("Graded with the answer key, this correction would have **increased** the gap. " +
                   ("The safety rule did not stop it." if f["applied"] else "The safety rule stopped it."))
    deployed_gap = px.eo_gap if f["applied"] else erm.eo_gap
    deployed_auc = px.auc if f["applied"] else erm.auc
    rows = [
        ("Original model", "no", erm.eo_gap, erm.auc, erm.acc, "–"),
        ("No demographics: cluster reweighting", "no", s.loc["cluster_rw"].eo_gap, s.loc["cluster_rw"].auc,
         s.loc["cluster_rw"].acc, tuned(s.loc["cluster_rw"], "strength level {:g}")),
        (f"Correction with {n} labels", f"{n} people", px.eo_gap, px.auc, px.acc, tuned(px)),
        ("Deployed after the safety rule", f"{n} people", deployed_gap, deployed_auc,
         px.acc if f["applied"] else erm.acc, "applied" if f["applied"] else "kept original"),
        ("Oracle (sees everyone's group, reference only)", "everyone", s.loc["oracle"].eo_gap, s.loc["oracle"].auc,
         s.loc["oracle"].acc, tuned(s.loc["oracle"])),
    ]
    t = pd.DataFrame(rows, columns=["Model", "Group labels used", "EO gap (test)", "AUC", "Accuracy", "Setting"])
    t["Gap change"] = [pct(100 * (g / erm.eo_gap - 1)) for g in t["EO gap (test)"]]
    c = st.columns(3)
    c[0].metric("Original gap", f"{erm.eo_gap:.3f}")
    c[1].metric("Deployed gap", f"{deployed_gap:.3f}", pct(100 * (deployed_gap / erm.eo_gap - 1)),
                delta_color="inverse")
    c[2].metric("AUC change", f"{100 * (deployed_auc - erm.auc):+.2f} points", delta_color="off")
    chart_df = t.iloc[[0, 1, 2, 4]].assign(kind=["Original", "No demographics", f"{n} labels", "Oracle"])
    st.altair_chart(bars(chart_df, "EO gap (test)", "Model", "kind",
                         ["Original", "No demographics", f"{n} labels", "Oracle"],
                         [C["base"], C["naive"], C["proxy"], C["oracle"]],
                         "Equalized-odds gap on the test split, graded with the answer key (lower is fairer)",
                         4 * 46 + 70, sort=list(chart_df.Model)), use_container_width=True)
    st.dataframe(t[["Model", "Group labels used", "EO gap (test)", "Gap change", "AUC", "Accuracy", "Setting"]]
                 .round(4), hide_index=True, width="stretch")
    fx = saved("fix")
    sub = fx[(fx.t == f"{DS}-{ATTR}") & fx.arm.isin([f"proxy[random] n={n}", "cluster_rw", "oracle"])]
    if len(sub):
        st.markdown("#### This dataset in the 30-seed study")
        names = {f"proxy[random] n={n}": f"Correction with {n} labels", "cluster_rw": "No demographics: cluster "
                 "reweighting", "oracle": "Oracle (reference)"}
        u = sub.assign(Model=sub.arm.map(names))[["Model", "eo_red", "ci_lo", "ci_hi", "harm_rate", "n_seeds"]]
        u.columns = ["Model", "Gap removed (%)", "95% CI low", "95% CI high", "Runs made worse", "Seeds"]
        st.dataframe(u.round(2), hide_index=True, width="stretch")
    else:
        st.caption("This pair was used only in the safety-rule study, so the main 30-seed table has no row for it.")


def page_results():
    st.header("📊 The 30-seed study")
    st.caption("9 classic datasets and 10 US states (census), every test rerun with 30 random splits. Tables are "
               "computed from the raw results in results/ by app/build_summaries.py.")
    tabs = st.tabs(["🔍 Find", "📏 Measure", "🔧 Fix", "🛡️ Safety rule", "📝 Predictions"])
    with tabs[0]:
        a = saved("audit")
        st.markdown(f"**The hidden group ranked first in {a.hidden_first.sum()} of {a.runs.sum()} audits.**")
        t = a[["family", "target", "runs", "hidden_first", "median_rank", "candidates", "hidden_gap"]].copy()
        t.columns = ["Data", "Target", "Audits", "Hidden group ranked first", "Median rank", "Candidate groups",
                     "Hidden group's gap"]
        st.dataframe(t.round(3), hide_index=True, width="stretch")
    with tabs[1]:
        e5 = saved("e5")
        p = e5.pivot_table(index="nlab", columns="family", values=["err_labelled", "err_proxy"])
        out = pd.DataFrame(index=p.index)
        for fam in e5.family.unique():
            out[f"{fam}: labelled only"] = p[("err_labelled", fam)]
            out[f"{fam}: guesser"] = p[("err_proxy", fam)]
            out[f"{fam}: × better"] = p[("err_labelled", fam)] / p[("err_proxy", fam)]
        st.markdown("Mean absolute error of the estimated gap, by number of people who revealed their group.")
        st.dataframe(out.round(3).reset_index().rename(columns={"nlab": "People labelled"}), hide_index=True,
                     width="stretch")
    with tabs[2]:
        fx = saved("fix")
        cl = fx[fx.family == "Classic datasets"]
        names = {"arl": "ARL", "cvar": "CVaR-DRO", "jtt": "JTT", "cluster_rw": "Cluster reweighting",
                 "oracle": "Oracle (sees everyone's group)"}
        names.update({f"proxy[random] n={k}": f"Correction, {k} labels" for k in E.NS})
        kinds = {"arl": "No demographics", "cvar": "No demographics", "jtt": "No demographics",
                 "cluster_rw": "No demographics", "oracle": "Oracle"}
        mean = cl[cl.arm.isin(names)].groupby("arm").eo_red.mean().reset_index()
        mean["Method"] = mean.arm.map(names)
        mean["kind"] = mean.arm.map(kinds).fillna("Few labels")
        order = [names[k] for k in ["arl", "cvar", "jtt", "cluster_rw"] + [f"proxy[random] n={k}" for k in E.NS]
                 + ["oracle"]]
        st.markdown("Mean share of the equalized-odds gap removed over the 8 main classic targets (T1 tuning).")
        st.altair_chart(bars(mean, "eo_red", "Method", "kind", ["No demographics", "Few labels", "Oracle"],
                             [C["naive"], C["proxy"], C["oracle"]], "Gap removed (%)", 10 * 34 + 70, fmt=".1f",
                             sort=order, rule=0), use_container_width=True)
        fam = st.radio("Data", ["Classic datasets", "US census (round 1)"], horizontal=True, key="fix_family")
        arm = st.selectbox("Method", list(names), format_func=names.get,
                           index=list(names).index("proxy[random] n=100"), key="fix_arm")
        u = fx[(fx.family == fam) & (fx.arm == arm)][["target", "eo_red", "ci_lo", "ci_hi", "p", "eo0", "harm_rate"]]
        u.columns = ["Target", "Gap removed (%)", "95% CI low", "95% CI high", "p (Wilcoxon, one-sided)",
                     "Original gap", "Runs made worse"]
        st.dataframe(u.round(3), hide_index=True, width="stretch")
    with tabs[3]:
        g = saved("guard")
        st.markdown(f"**Safety rule:** apply the correction only if the guesser measures a gap of at least "
                    f"{E.GUARD}. A run counts as *made worse* if the gap rises by more than {E.HARM}.")
        long = g.melt(id_vars=["family", "nlab"], value_vars=["harm", "harm_guarded"], var_name="w", value_name="v")
        long["Rule"] = long.w.map({"harm": "Always correct", "harm_guarded": "With the safety rule"})
        long["Setting"] = long.family.str.replace("US census ", "Census ") + ", " + long.nlab.astype(int).astype(str) + \
            " labels"
        ch = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Rule:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            x=alt.X("v:Q", title="Share of runs made worse", axis=alt.Axis(format="%")),
            color=alt.Color("Rule:N", scale=alt.Scale(domain=["Always correct", "With the safety rule"],
                                                      range=[C["naive"], C["proxy"]]),
                            legend=alt.Legend(orient="bottom", title=None)),
            row=alt.Row("Setting:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=["Setting:N", "Rule:N", alt.Tooltip("v:Q", format=".1%")],
        ).properties(height=46, width=420)
        st.altair_chart(ch, use_container_width=False)
        t = g.copy()
        t.columns = ["Data", "People labelled", "Made worse: always", "Made worse: with rule", "Rule applied",
                     "Gap removed: always (%)", "Gap removed: with rule (%)"]
        st.dataframe(t.round(3), hide_index=True, width="stretch")
        st.caption("The rule is cautious, not free: where correcting already worked well it sometimes holds back.")
    with tabs[4]:
        h = saved("hypotheses")
        h["Outcome"] = h.supported.map({True: "✅ supported", False: "❌ not supported"}).fillna("➖ inconclusive")
        st.dataframe(h[["id", "prediction", "result", "Outcome"]].rename(
            columns={"id": "", "prediction": "Prediction written before the runs", "result": "Result"}),
            hide_index=True, width="stretch")
        st.caption("H1–H7 are from PREREGISTRATION.md. Census rounds: the guesser measured better on 5 of 5 new "
                   "states, and the safety rule cut harmful fixes (Safety rule tab).")


def page_method():
    st.header("📖 Method and limits")
    st.markdown(
        "**Setup (from PREREGISTRATION.md)**\n"
        "- Each seed splits a dataset into **training 50% / labelling pool 20% / test 30%**, stratified by label "
        "and group. The protected attribute is **never a model input**.\n"
        "- Learner: logistic regression with a soft equalized-odds penalty, strengths μ ∈ {0.3, 1, 3, 10, 30}.\n"
        "- Guesser (proxy): logistic regression P(group | inputs) trained on the pool rows that revealed their "
        "group; its soft probabilities enter the penalty.\n"
        "- Tuning without demographics (T1): the strongest μ whose pool AUC stays within 0.01 of the original "
        "model, otherwise the original model.\n"
        "- Grading: on the test split, with the threshold set so the selection rate equals the training base rate. "
        "Main metric: equalized-odds gap, max(|TPR gap|, |FPR gap|).\n"
        "- Oracle: the same penalty with everyone's true group. A reference, not a usable method.")
    st.markdown(
        "**What did not work**\n"
        "- Popular methods that need no demographics (ARL, DRO, JTT, clustering) barely reduced the gap.\n"
        "- Choosing smarter people to ask was no better than choosing at random.\n"
        "- When one group reveals its membership less often, results got 3–5 points worse.\n"
        "- Two of the seven predictions failed (H2, H6) and the neural-network check (H7) was inconclusive.\n"
        "- The safety rule is cautious: it sometimes holds back a correction that would have helped.")
    st.markdown(
        "**About this app**\n"
        "- Live pages call the study's code (fairlab, scripts/exp_main.py, scripts/exp_audit.py). For seeds 0–29 "
        "they reproduce the stored rows exactly; automated tests check this.\n"
        "- The audit model uses l2 = 1e-4, the setting that produced results/e1_audit.csv (the other experiments "
        "use 1e-3).\n"
        "- The US census data needs a large download, so it appears only in the 30-seed results.")
    st.markdown(f"Code and raw results: [{REPO_URL}]({REPO_URL})")


{"🏠 Overview": page_overview, "🔍 Find": page_find, "📏 Measure": page_measure, "🔧 Fix": page_fix,
 "📊 30-seed results": page_results, "📖 Method & limits": page_method}[page]()
