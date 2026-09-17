"""
Affinity-panel audit: for every target on the SomaScan 11k and Olink Explore
HT panels, is there a protein-level reason MS could never confirm it?

Affinity platforms report a value for every target. Whether that target could
be validated by mass spectrometry is a different question, and it has a
protein level answer independent of any sample: a protein with no unique
tryptic peptide cannot be confirmed by shotgun MS by any protease.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import DATA_DIR, TAB_DIR
from utils_validation import read_adat, MIN_CLEAN, ORDER

ORG = DATA_DIR / "organoids"
SOMA_ADAT = sorted((ORG / "somalogic").glob("*medNormSMP*_organoids.adat"))
OLINK = ORG / "olink" / "olink_slim.csv"
DGIDB = DATA_DIR / "interactions.tsv"


def panel_targets():
    rows = []
    if SOMA_ADAT:
        _, _, cm = read_adat(SOMA_ADAT[-1])
        cm = cm[(cm["Type"] == "Protein") & (cm["Organism"] == "Human")]
        for r in cm.itertuples(index=False):
            accs = [a.strip() for a in str(r.UniProt).replace("|", ",").replace(";", ",").split(",") if a.strip()]
            rows.append(dict(panel="SomaScan", feature_id=r.SeqId, uniprot=accs[0] if accs else None,
                             gene=r.EntrezGeneSymbol, multi_target=int(len(accs) > 1)))
    if OLINK.exists():
        o = pd.read_csv(OLINK, usecols=["OlinkID", "UniProt", "Assay", "AssayType"])
        o = o[o["AssayType"] == "assay"].drop_duplicates("OlinkID")
        for r in o.itertuples(index=False):
            accs = [a.strip() for a in str(r.UniProt).replace("|", ",").replace(";", ",").split(",") if a.strip()]
            rows.append(dict(panel="Olink", feature_id=r.OlinkID, uniprot=accs[0] if accs else None,
                             gene=r.Assay, multi_target=int(len(accs) > 1)))
    t = pd.DataFrame(rows)
    t["uniprot"] = t["uniprot"].str.split("-").str[0]
    return t


def annotate(t, out):
    cen = pd.read_csv(out / "proteome_census.tsv", sep="\t")
    cen_cols = ["uniprot", "pe", "p_seq"] + [c for c in ("ensg",) if c in cen.columns]
    t = t.merge(cen[cen_cols].drop_duplicates("uniprot"), on="uniprot", how="left")
    pep = pd.read_csv(out / "features_peptides.tsv", sep="\t",
                      usecols=["uniprot", "pep_n_unique", "pep_n_clean", "pep_frac_shared"])
    t = t.merge(pep, on="uniprot", how="left")
    fp = pd.read_csv(out / "features_protein.tsv", sep="\t", low_memory=False).drop_duplicates("uniprot")
    keep = [c for c in ("ensg", "is_secreted", "has_signal", "is_membrane", "is_ecm", "length", "mature_length")
            if c in fp.columns]
    fp = fp[["uniprot"] + keep].rename(columns={"ensg": "ensg_fp"})
    t = t.merge(fp, on="uniprot", how="left")
    if "ensg" in t.columns:
        t["ensg"] = t["ensg"].where(t["ensg"].notna(), t["ensg_fp"])
    else:
        t["ensg"] = t["ensg_fp"]
    t = t.drop(columns=["ensg_fp"])
    ml = pd.read_csv(out / "ml_per_gene_predictions.tsv", sep="\t").drop_duplicates("ensg")
    t = t.merge(ml[["ensg", "y", "p_detect_intrinsic", "top_dark_reason"]], on="ensg", how="left")
    wb = pd.read_csv(out / "paxdb_wholebody.tsv", sep="\t").dropna().drop_duplicates("ensg")
    t = t.merge(wb.rename(columns={"paxdb_ppm_global": "paxdb_ppm"}), on="ensg", how="left")
    if "paxdb_ppm" in t.columns:
        t["log_abundance"] = np.log10(t["paxdb_ppm"].where(t["paxdb_ppm"] > 0))
    og = out / "organoid_gene_table.tsv"
    if og.exists():
        g = pd.read_csv(og, sep="\t")
        for arm, tag in (("Organoid", "lysate"), ("Media", "media")):
            a = g[g["arm"] == arm][["ensg", "ms_present", "soma_present", "olink_present", "aff_abund_z"]]
            a = a.rename(columns={c: f"org_{tag}_{c}" for c in a.columns if c != "ensg"})
            t = t.merge(a, on="ensg", how="left")
    bf = out / "biofluid_gene_table.tsv"
    if bf.exists():
        b = pd.read_csv(bf, sep="\t", usecols=["ensg", "set", "ms_present", "aff_present"])
        for st in b["set"].unique():
            tag = st.lower().replace(" ", "_")
            a = b[b["set"] == st][["ensg", "ms_present", "aff_present"]]
            a = a.rename(columns={"ms_present": f"bf_{tag}_ms_present", "aff_present": f"bf_{tag}_aff_present"})
            t = t.merge(a, on="ensg", how="left")
    if DGIDB.exists():
        dg = pd.read_csv(DGIDB, sep="\t", usecols=["gene_name", "approved"])
        druggable = set(dg["gene_name"].dropna())
        approved = set(dg.loc[dg["approved"].astype(str).str.upper() == "TRUE", "gene_name"].dropna())
        t["druggable"] = t["gene"].isin(druggable).astype(int)
        t["approved_drug"] = t["gene"].isin(approved).astype(int)
    return t


def classify(t):
    pe1 = t.loc[t["pe"] == 1, "p_seq"]
    hard = float(pe1.quantile(0.25)) if len(pe1) else np.nan
    cls = np.where(t["pep_n_unique"] == 0, "inference_locked",
          np.where(t["pep_n_clean"] < MIN_CLEAN, "inference_limited",
          np.where(t["p_seq"] < hard, "chemistry_limited", "confirmable")))
    cls = np.where(t["p_seq"].isna(), "unscored", cls)
    t["ms_class"] = cls
    return t, hard


def summarise(t, hard):
    L = ["affinity audit: could MS confirm what the panel reports?", "",
         f"classes: inference_locked = no unique tryptic peptide; inference_limited = < {MIN_CLEAN} unique clean",
         f"peptides; chemistry_limited = sequence-only P(detect) < {hard:.3f} (PE1 25th percentile); else confirmable.",
         ""]
    for panel, d in t.groupby("panel"):
        L.append(f"[{panel}] targets {len(d):,}, distinct UniProt {d['uniprot'].nunique():,}, "
                 f"multi-target features {int(d['multi_target'].sum()):,}, "
                 f"in the GTEx-labelled universe {int(d['y'].notna().sum()):,}")
        vc = d["ms_class"].value_counts()
        for c in ORDER:
            n = int(vc.get(c, 0))
            if n:
                sub = d[d["ms_class"] == c]
                line = f"   {c:18s} {n:6,} ({n / len(d):5.1%})"
                if "druggable" in d.columns:
                    line += f"   druggable {int(sub['druggable'].sum()):,}, approved-drug {int(sub['approved_drug'].sum()):,}"
                if "org_lysate_ms_present" in d.columns:
                    m = sub[sub["org_lysate_soma_present"].eq(1) | sub["org_lysate_olink_present"].eq(1)]
                    if len(m) >= 20:
                        line += f"   organoid lysate MS-confirmed {m['org_lysate_ms_present'].mean():5.1%} (n={len(m):,})"
                    mm = sub[sub["org_media_soma_present"].eq(1) | sub["org_media_olink_present"].eq(1)]
                    if len(mm) >= 20:
                        line += f"   media {mm['org_media_ms_present'].mean():5.1%} (n={len(mm):,})"
                L.append(line)
                bf_cols = [c for c in d.columns if c.startswith("bf_") and c.endswith("_ms_present")]
                if bf_cols:
                    parts = []
                    for c in bf_cols:
                        tag = c[3:-len("_ms_present")]
                        m = sub[sub[f"bf_{tag}_aff_present"].eq(1)]
                        if len(m) >= 20:
                            parts.append(f"{tag.replace('_', ' ')} {m[c].mean():5.1%} (n={len(m):,})")
                    if parts:
                        L.append(f"   {'':18s}        biofluid MS-confirmed (all batches): " + "   ".join(parts))
        # abundance strata
        if "log_abundance" in d.columns:
            q = pd.qcut(d["log_abundance"], 3, labels=["low", "mid", "high"])
            d = d.assign(abund_tier=q.astype(object).fillna("no PaxDb value"))
            L.append("   by whole-body abundance tier: not-confirmable share (chemistry + inference)")
            for tier in ("high", "mid", "low", "no PaxDb value"):
                s = d[d["abund_tier"] == tier]
                if len(s):
                    nc = s["ms_class"].isin(["chemistry_limited", "inference_limited", "inference_locked"]).mean()
                    L.append(f"     {tier:15s} n={len(s):6,}  not confirmable {nc:5.1%}")
        # secreted vs not
        if "is_secreted" in d.columns:
            for lab, m in (("secreted / signal peptide", (d["is_secreted"] == 1) | (d["has_signal"] == 1)),
                           ("intracellular", (d["is_secreted"] != 1) & (d["has_signal"] != 1))):
                s = d[m]
                if len(s):
                    nc = s["ms_class"].isin(["chemistry_limited", "inference_limited", "inference_locked"]).mean()
                    L.append(f"   {lab:26s} n={len(s):6,}  not confirmable {nc:5.1%}")
        L.append("")
    # overlap between panels
    both = t.groupby("uniprot")["panel"].nunique()
    shared = set(both[both > 1].index)
    if shared:
        s = t[t["uniprot"].isin(shared)].drop_duplicates(["panel", "uniprot"])
        piv = s.pivot_table(index="uniprot", columns="panel", values="ms_class", aggfunc="first")
        agree = (piv["SomaScan"] == piv["Olink"]).mean() if {"SomaScan", "Olink"} <= set(piv.columns) else np.nan
        L.append(f"targets on both panels: {len(shared):,}   class agreement {agree:.1%} (same protein, same class by construction "
                 f"unless the two panels map different accessions)")
        L.append("")
    return "\n".join(L)


t = panel_targets()
print(f"panel targets: {len(t):,} ({t['panel'].value_counts().to_dict()})")
t = annotate(t, TAB_DIR)
t, hard = classify(t)
t.to_csv(TAB_DIR / "panel_audit.tsv", sep="\t", index=False)
text = summarise(t, hard)
(TAB_DIR / "panel_audit.txt").write_text(text)
print(text)
