"""
Do the proteins that the literature has actually nominated as biomarkers
from affinity data fall in the classes MS cannot confirm?
Nomination groups per fluid:
  affinity_only   nominated by SomaScan or Olink, never by MS
  both            nominated by an affinity platform AND by MS
  ms_only         nominated by MS, by no affinity platform
  other_only      nominated only by Luminex, NULISAseq or ELISA arrays

"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from utils_validation import MIN_CLEAN, ORDER
from config import DATA_DIR, TAB_DIR

BIOMARKERS = DATA_DIR / "biomarkers.xlsx"
DGIDB = DATA_DIR / "interactions.tsv"
NOT_CONF = ["chemistry_limited", "inference_limited", "inference_locked"]
AFF = ["SomaScan", "Olink PEA"]
OTHER = ["Luminex xMAP", "NULISAseq", "ELISA-based protein array"]


def load_nominated():
    x = pd.ExcelFile(BIOMARKERS)
    rows = []
    for fluid in x.sheet_names:
        d = x.parse(fluid)
        d.columns = [c.strip() for c in d.columns]
        for _, r in d.iterrows():
            rec = dict(fluid=fluid, gene=r["HGNC"], uniprot=str(r["Uniprot"]).split("-")[0].strip())
            for m in ["Mass spectrometry"] + AFF + OTHER:
                rec[m] = int(m in d.columns and pd.notna(r[m]))
            rows.append(rec)
    n = pd.DataFrame(rows)
    ms = n["Mass spectrometry"] == 1
    aff = n[AFF].sum(axis=1) > 0
    oth = n[OTHER].sum(axis=1) > 0
    n["group"] = np.select([aff & ~ms, aff & ms, ms & ~aff, oth & ~aff & ~ms],
                           ["affinity_only", "both", "ms_only", "other_only"], default="unclassified")
    return n


def class_all(n, out):
    pa = pd.read_csv(out / "panel_audit.tsv", sep="\t")
    cen = pd.read_csv(out / "proteome_census.tsv", sep="\t", usecols=["uniprot", "pe", "p_seq"]).drop_duplicates("uniprot")
    pep = pd.read_csv(out / "features_peptides.tsv", sep="\t",
                      usecols=["uniprot", "pep_n_unique", "pep_n_clean", "pep_frac_shared"])
    fp = pd.read_csv(out / "features_protein.tsv", sep="\t", low_memory=False).drop_duplicates("uniprot")
    keep = [c for c in ("ensg", "is_secreted", "has_signal") if c in fp.columns]
    t = n.merge(cen, on="uniprot", how="left").merge(pep, on="uniprot", how="left").merge(fp[["uniprot"] + keep], on="uniprot", how="left")

    hard = float(pa.loc[pa["pe"] == 1, "p_seq"].quantile(0.25))
    cls = np.where(t["pep_n_unique"] == 0, "inference_locked",
          np.where(t["pep_n_clean"] < MIN_CLEAN, "inference_limited",
          np.where(t["p_seq"] < hard, "chemistry_limited", "confirmable")))
    t["ms_class"] = np.where(t["p_seq"].isna(), "unscored", cls)
    t["on_somascan"] = t["uniprot"].isin(pa.loc[pa["panel"] == "SomaScan", "uniprot"]).astype(int)
    t["on_olink"] = t["uniprot"].isin(pa.loc[pa["panel"] == "Olink", "uniprot"]).astype(int)
    og = out / "organoid_gene_table.tsv"
    if og.exists() and "ensg" in t.columns:
        g = pd.read_csv(og, sep="\t")
        for arm, tag in (("Organoid", "lysate"), ("Media", "media")):
            a = g[g["arm"] == arm][["ensg", "ms_present", "aff_present"]].rename(
                columns={"ms_present": f"org_{tag}_ms", "aff_present": f"org_{tag}_aff"})
            t = t.merge(a, on="ensg", how="left")
    if DGIDB.exists():
        dg = pd.read_csv(DGIDB, sep="\t", usecols=["gene_name", "approved"])
        t["druggable"] = t["gene"].isin(set(dg["gene_name"].dropna())).astype(int)
        t["approved_drug"] = t["gene"].isin(set(dg.loc[dg["approved"].astype(str).str.upper() == "TRUE", "gene_name"])).astype(int)
    return t, pa, hard


def fisher(a_nc, a_n, b_nc, b_n):
    tab = [[a_nc, a_n - a_nc], [b_nc, b_n - b_nc]]
    odds, p = fisher_exact(tab)
    return odds, p


def summarise(t, pa, hard):
    L = ["Nominated Biomarkers: are the proteins the literature nominated from affinity data MS-confirmable?", "",
         f"classes as in panel_audit.py (chemistry-limited threshold {hard:.3f}). Not-confirmable = chemistry + inference classes.",
         ""]
    bg = {p: pa[pa["panel"] == p] for p in ("SomaScan", "Olink")}
    bg_nc = {p: (int(d["ms_class"].isin(NOT_CONF).sum()), int(d["ms_class"].ne("unscored").sum())) for p, d in bg.items()}
    L.append("panel background, not-confirmable share: " + "  ".join(f"{p} {a / b:.1%} ({a:,}/{b:,})" for p, (a, b) in bg_nc.items()))
    L.append("")
    rows = []
    for fluid in ("plasma", "csf", "cortex"):
        d = t[(t["fluid"] == fluid) & (t["ms_class"] != "unscored")]
        if not len(d):
            continue
        L.append(f"{fluid} nominated proteins {len(d):,}")
        for grp in ("affinity_only", "both", "ms_only", "other_only"):
            s = d[d["group"] == grp]
            if not len(s):
                continue
            vc = s["ms_class"].value_counts()
            nc = int(s["ms_class"].isin(NOT_CONF).sum())
            L.append(f"   {grp:14s} n={len(s):4d}  not confirmable {nc / len(s):5.1%}   " +
                     "  ".join(f"{c.replace('_', ' ')} {int(vc.get(c, 0))}" for c in ORDER if vc.get(c, 0)))
            rows.append(dict(fluid=fluid, group=grp, n=len(s), not_confirmable=nc))
        a = d[d["group"] == "affinity_only"]
        m = d[d["group"] == "ms_only"]
        if len(a) >= 10:
            a_nc = int(a["ms_class"].isin(NOT_CONF).sum())

            for plat, col, key in (("SomaScan", "SomaScan", "SomaScan"), ("Olink", "Olink PEA", "Olink")):
                s = a[a[col] == 1]
                if len(s) >= 10:
                    s_nc = int(s["ms_class"].isin(NOT_CONF).sum())
                    b_nc, b_n = bg_nc[key]
                    odds, p = fisher(s_nc, len(s), b_nc, b_n)
                    L.append(f"   {plat}-nominated, affinity-only (n={len(s)}): not confirmable {s_nc / len(s):.1%} vs "
                             f"{plat} panel {b_nc / b_n:.1%}   OR {odds:.2f}  p={p:.2e}")
            if len(m) >= 10:
                m_nc = int(m["ms_class"].isin(NOT_CONF).sum())
                odds, p = fisher(a_nc, len(a), m_nc, len(m))
                L.append(f"   affinity-only vs MS-only nominees: {a_nc / len(a):.1%} vs {m_nc / len(m):.1%}   OR {odds:.2f}  p={p:.2e}")
        if "org_lysate_ms" in d.columns and len(a):
            e = a[a["org_lysate_aff"] == 1]
            if len(e) >= 10:
                L.append(f"   affinity-only nominees measured by affinity in organoid lysate: {len(e)}, MS-confirmed there "
                         f"{e['org_lysate_ms'].mean():.1%}  (confirmable class {e.loc[e['ms_class'] == 'confirmable', 'org_lysate_ms'].mean():.1%}, "
                         f"not-confirmable classes {e.loc[e['ms_class'].isin(NOT_CONF), 'org_lysate_ms'].mean():.1%})")
        L.append("")
    # named list
    named = t[(t["group"] == "affinity_only") & t["ms_class"].isin(NOT_CONF)].copy()
    if len(named):
        L.append(f"affinity-only nominees in the not-confirmable classes ({len(named)}), by fluid:")
        for fluid, s in named.groupby("fluid"):
            s = s.sort_values(["ms_class", "gene"])
            items = []
            for r in s.itertuples(index=False):
                tag = r.ms_class.replace("_limited", "").replace("_locked", "-locked")
                extra = []
                if getattr(r, "approved_drug", 0) == 1:
                    extra.append("approved drug")
                elif getattr(r, "druggable", 0) == 1:
                    extra.append("druggable")
                if getattr(r, "org_lysate_aff", np.nan) == 1:
                    extra.append("organoid MS " + ("yes" if r.org_lysate_ms == 1 else "no"))
                items.append(f"{r.gene} ({tag}{', ' + ', '.join(extra) if extra else ''})")
            L.append(f"   {fluid}: " + "; ".join(items))
        L.append("")
    return "\n".join(L), pd.DataFrame(rows)


n = load_nominated()
print(f"  nominated: {len(n):,} rows, groups {n['group'].value_counts().to_dict()}")
t, pa, hard = class_all(n, TAB_DIR)
t.to_csv(TAB_DIR / "nominated_audit.tsv", sep="\t", index=False)
text, _ = summarise(t, pa, hard)
(TAB_DIR / "nominated_audit.txt").write_text(text)
print(text)
