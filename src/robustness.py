"""
Robustness of the lgoistic regression results (How robust peptide yield vs transmembrane result).

"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import RELIABILITY_ORDER, LEVEL_ORD, DATA_DIR, TAB_DIR
from utils_features import _tryptic_count

BIOPHYS_BASE = ["gravy", "log_mw", "n_tm", "has_signal",
                "frac_pos_celltypes", "reliability_ord"]


def _fit_chem(df, tryptic_col="log_tryptic"):
    """Chemistry stratum logistic; return (n, b_tryptic, p_tryptic, b_ntm, p_ntm)."""
    sub = df[df["type"].isin(["measured", "not_measured"])].copy()
    need = ["log_abundance"] + [c for c in BIOPHYS_BASE if c in sub.columns] + [tryptic_col]
    sub = sub.dropna(subset=need)
    need = [c for c in need if sub[c].nunique() > 1]   # drop constants (e.g. reliability at a floor)
    if sub["gtex_detected"].nunique() < 2 or len(sub) < 100:
        return len(sub), np.nan, np.nan, np.nan, np.nan
    z = sub.copy()
    for c in need:
        sd = z[c].astype(float).std(ddof=0)
        z[c] = (z[c].astype(float) - z[c].astype(float).mean()) / sd if sd else 0.0
    tis = pd.get_dummies(z["tissue"], prefix="t", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([z[need], tis], axis=1))
    f = sm.Logit(sub["gtex_detected"].values, X).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": sub["ensg"].values})
    return (len(sub), f.params[tryptic_col], f.pvalues[tryptic_col],
            f.params["n_tm"], f.pvalues["n_tm"])


def _row(label, res):
    n, bt, pt, bn, pn = res
    if np.isnan(bt):
        return f"{label:24s}  n={n:>6,}   [skipped]"
    return (f"{label:24s}  n={n:>6,}   "
            f"log_tryptic {bt:+.3f} (p={pt:.1e})   n_tm {bn:+.3f} (p={pn:.1e})")


def tryptic_window_counts(lo, hi):
    """In-silico tryptic peptide counts in [lo, hi] aa on the mature chain, same digest as stage 3.3."""
    up = pd.read_csv(DATA_DIR / "uniprot_human.tsv", sep="\t", dtype=str, keep_default_na=False,
                     na_values=[], usecols=["Entry", "Sequence"]).rename(columns={"Entry": "uniprot", "Sequence": "seq"})
    fp = pd.read_csv(TAB_DIR / "features_protein.tsv", sep="\t",
                     usecols=["ensg", "uniprot", "mature_start", "mature_end"]).drop_duplicates("ensg")
    fp = fp.merge(up, on="uniprot", how="inner")
    out = {r.ensg: _tryptic_count(r.seq.strip().upper(), lo, hi, int(r.mature_start), int(r.mature_end))
           for r in fp.itertuples(index=False)}
    return pd.Series(out, name="ntw")


def ihc_maxlevel():
    """Per (ensg,tissue) max IHC level among present cell types, for threshold sweep."""
    ic = pd.read_csv(TAB_DIR / "ihc_celltype_long.tsv", sep="\t")
    ic = ic[ic["ihc_call"] == "present"].copy()
    ic["lev"] = ic["level"].map(LEVEL_ORD)
    return ic.groupby(["ensg", "tissue"])["lev"].max()


tbl = pd.read_csv(TAB_DIR / "model_table.tsv", sep="\t")
L = ["How robust the LR chemistry result", "robust iff log_tryptic stays + / significant and n_tm stays null.", "",
     f"baseline:", _row("Low+ / 7-30 / all", _fit_chem(tbl)), "", "(A) IHC present threshold (Low+ default):",
     _row("Low+  (default)", _fit_chem(tbl))]
# (A) IHC present threshold
t2 = tbl.merge(ihc_maxlevel().rename("maxlev"), on=["ensg", "tissue"], how="left")
L.append(_row("Medium+", _fit_chem(t2[t2["maxlev"] >= 2])))
L.append(_row("High only", _fit_chem(t2[t2["maxlev"] >= 3])))
L.append("")

# (B) tryptic length window
L.append("(B) tryptic length window (7-30 default, mature chain):")
L.append(_row("7-30  (default)", _fit_chem(tbl)))
for lo, hi in [(6, 40), (8, 25), (9, 35)]:
    cnt = tryptic_window_counts(lo, hi)
    t3 = tbl.merge(cnt, left_on="ensg", right_index=True, how="left")
    t3["lt"] = np.log10(t3["ntw"].clip(lower=1))
    L.append(_row(f"{lo}-{hi}", _fit_chem(t3, tryptic_col="lt")))
L.append("")

# (C) reliability floor
L.append("(C) reliability floor:")
L.append(_row("all (default)", _fit_chem(tbl)))
floors = {"Approved+": {"Enhanced", "Supported", "Approved"},
          "Supported+ (Enh+Sup)": {"Enhanced", "Supported"}}
for lab, keep in floors.items():
    L.append(_row(lab, _fit_chem(tbl[tbl["best_reliability"].isin(keep)])))
L.append("")

# (D) reliability gradient vs abundance gradient
L.append("(D) reliability gradient vs abundance gradient:")
L.append("raw MS-dark fraction by antibody class, then within whole-body abundance quartile.")
t = tbl[tbl["best_reliability"].isin(RELIABILITY_ORDER)].copy()
t["q"] = pd.qcut(t["log_abundance"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
raw = t.groupby("best_reliability").agg(
    n=("gtex_detected", "size"),
    dark=("gtex_detected", lambda s: 1 - s.mean()),
    med_log_ab=("log_abundance", "median")).reindex(RELIABILITY_ORDER)
xt = pd.crosstab(t["best_reliability"], t["q"],
                 values=1 - t["gtex_detected"], aggfunc="mean").reindex(RELIABILITY_ORDER)
L.append(f"  {'grade':10s} {'n':>7s}  {'dark':>5s}  {'med_log_ab':>10s}   "
         + "  ".join(f"{c:>7s}" for c in xt.columns))
for g in RELIABILITY_ORDER:
    L.append(f"  {g:10s} {raw.loc[g, 'n']:>7,}  {raw.loc[g, 'dark']:.3f}  "
             f"{raw.loc[g, 'med_log_ab']:>10.3f}   "
             + "  ".join(f"{xt.loc[g, c]:>7.3f}" for c in xt.columns))
L.append("")

# (E) transmembrane check
L.append("(E) transmembrane check, chemistry model:")
chem = tbl[tbl["type"].isin(["measured", "not_measured"])].dropna(
    subset=["log_abundance", "n_tm", "log_tryptic"]).copy()
for c in ["log_abundance", "n_tm", "log_tryptic"]:
    s = chem[c].astype(float)
    chem[c] = (s - s.mean()) / s.std(ddof=0)
tis = pd.get_dummies(chem["tissue"], prefix="t", drop_first=True).astype(float)
grp = chem["ensg"].values
Xb = sm.add_constant(pd.concat([chem[["log_abundance", "n_tm"]], tis], axis=1))
fb = sm.Logit(chem["gtex_detected"].values, Xb).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": grp})
Xa = sm.add_constant(pd.concat([chem[["n_tm", "log_abundance"]], tis], axis=1))
fa = sm.OLS(chem["log_tryptic"].values, Xa).fit(
    cov_type="cluster", cov_kwds={"groups": grp})
L.append(f"n_tm total effect on detection (abundance + tissue controlled): "
         f"{fb.params['n_tm']:+.3f} (p={fb.pvalues['n_tm']:.1e})")
L.append(f"a-path, log_tryptic ~ n_tm (abundance + tissue controlled):    "
         f"{fa.params['n_tm']:+.3f} (p={fa.pvalues['n_tm']:.1e})")

text = "\n".join(L)
(TAB_DIR / "robustness_summary.txt").write_text(text)
print(text)
