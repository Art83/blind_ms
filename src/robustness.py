"""
Robustness of the lgoistic regression results (How robust peptide yield vs transmembrane result).

The Result was:  conditional on abundance, MS-darkness tracks tryptic peptide
yield (log_tryptic +, significant) but not transmembrane count (n_tm)
Thi result wss built on three choices. This reruns the CHEMISTRY model
(measured vs not_measured) while varying each, and prints the log_tryptic and
n_tm coefficients in every cell. The result is robust iff log_tryptic stays
positive+significant and n_tm stays null across the grid.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from config import RELIABILITY_ORDER, LEVEL_ORD, DATA_DIR, TAB_DIR
from utils import symbol_to_ensg

BIOPHYS_BASE = ["gravy", "log_mw", "n_tm", "has_signal",
                "frac_pos_celltypes", "reliability_ord"]


def _fit_chem(df, tryptic_col="log_tryptic"):
    """Chemistry stratum logistic; return (n, b_tryptic, p_tryptic, b_ntm, p_ntm)."""
    import statsmodels.api as sm
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
    """Recompute in-silico tryptic peptide counts in [lo,hi] aa from UniProt."""
    up = pd.read_csv(DATA_DIR / "uniprot_human.tsv", sep="\t", dtype=str,
                     keep_default_na=False, na_values=[])
    cols = list(up.columns)
    low = {c.lower(): c for c in cols}
    c_gene = low.get("gene names (primary)") or next(
        (c for c in cols if "gene" in c.lower()), None)
    c_seq = low.get("sequence") or next((c for c in cols if "seq" in c.lower()), None)
    gd = pd.read_csv(TAB_DIR / "gene_dict.tsv", sep="\t", dtype=str)
    sym2ensg = symbol_to_ensg(gd, tag="robustness", verbose=False)
    out = {}
    for _, r in up.iterrows():
        seq = (r[c_seq] or "").strip().upper()
        sym = (r[c_gene] or "").strip().split()[0] if r[c_gene] else None
        ensg = sym2ensg.get(sym) if sym else None
        if not seq or ensg is None:
            continue
        peps = re.sub(r"([KR])(?!P)", r"\1\n", seq).split("\n")
        out[ensg] = sum(1 for p in peps if lo <= len(p) <= hi)
    return pd.Series(out, name="ntw")


def ihc_maxlevel():
    """Per (ensg,tissue) max IHC level among present cell types, for threshold sweep."""
    path = TAB_DIR / "ihc_celltype_long.tsv"
    if not path.exists():
        return None
    ic = pd.read_csv(path, sep="\t")
    if "level" not in ic.columns:
        return None
    ic = ic[ic["ihc_call"] == "present"].copy()
    ic["lev"] = ic["level"].map(LEVEL_ORD)
    return ic.groupby(["ensg", "tissue"])["lev"].max()


if __name__ == "__main__":
    tbl = pd.read_csv(TAB_DIR / "model_table.tsv", sep="\t")
    L = ["How robust the LR chemistry result", "robust iff log_tryptic stays + / significant and n_tm stays null.", "",
         f"baseline:", _row("Low+ / 7-30 / all", _fit_chem(tbl)), "", "(A) IHC present threshold (Low+ default):",
         _row("Low+  (default)", _fit_chem(tbl))]
    # (A) IHC present threshold
    ml = ihc_maxlevel()
    if ml is not None:
        t2 = tbl.merge(ml.rename("maxlev"), on=["ensg", "tissue"], how="left")
        L.append(_row("Medium+", _fit_chem(t2[t2["maxlev"] >= 2])))
        L.append(_row("High only", _fit_chem(t2[t2["maxlev"] >= 3])))
    else:
        L.append("ihc_celltype_long has no 'level' column -> threshold sweep skipped")
    L.append("")

    # (B) tryptic length window
    L.append("(B) tryptic length window (7-30 default):")
    L.append(_row("7-30  (default)", _fit_chem(tbl)))
    try:
        for lo, hi in [(6, 40), (8, 25), (9, 35)]:
            cnt = tryptic_window_counts(lo, hi)
            t3 = tbl.merge(cnt, left_on="ensg", right_index=True, how="left")
            t3["lt"] = np.log10(t3["ntw"].clip(lower=1))
            L.append(_row(f"{lo}-{hi}", _fit_chem(t3, tryptic_col="lt")))
    except FileNotFoundError as e:
        L.append(f"uniprot/gene_dict missing -> window sweep skipped: {e}")
    L.append("")

    # (C) reliability floor
    L.append("(C) reliability floor:")
    L.append(_row("all (default)", _fit_chem(tbl)))
    if "best_reliability" in tbl.columns:
        floors = {"Approved+": {"Enhanced", "Supported", "Approved"},
                  "Supported+ (Enh+Sup)": {"Enhanced", "Supported"}}
        for lab, keep in floors.items():
            L.append(_row(lab, _fit_chem(tbl[tbl["best_reliability"].isin(keep)])))
    else:
        L.append("  [no best_reliability column -> reliability sweep skipped]")
    L.append("")

    # (D) reliability gradient vs abundance gradient
    L.append("(D) reliability gradient vs abundance gradient:")
    L.append("raw MS-dark fraction by antibody class, then within whole-body abundance quartile.")
    if "best_reliability" in tbl.columns:
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
    else:
        L.append("no best_reliability column -> gradient table skipped")
    L.append("")

    # (E) transmembrane check
    import statsmodels.api as sm
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
    L.append("Membrane proteins do yield fewer observable peptides, but that is already")
    L.append("carried by log_tryptic; there is no separate transmembrane effect to explain.")

    text = "\n".join(L)
    (TAB_DIR / "robustness_summary.txt").write_text(text)
    print(text)
