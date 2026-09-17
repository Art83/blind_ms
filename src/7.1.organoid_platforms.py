"""
Validation of the MS-detectability model in iPSC cerebral organoids
measured on three platforms (DIA-NN MS, SomaScan 11k, Olink Explore HT) in two
compartments (organoid lysate, conditioned media), two cell lines (TDR1
control, TDR4 AD), five organoids each.

preparation script

Detection rules:
  SomaScan  per aptamer, LoD = median(log2 RFU of buffer wells) + 3 * MAD

  Olink     per assay, LoD = median(NPX of negative-control samples) + 3 * SD,
            and Count >= OLINK_MIN_COUNT.

  MS   DIA-NN reports 0 for not detected. Above LoD if intensity > 0.

"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config import DATA_DIR, TAB_DIR
from utils_validation import build_resolver, read_adat

ORG = DATA_DIR / "organoids"
SOMA = {
    "Organoid": ORG / "somalogic" / "SS-25153694_v5.0_other.hybNorm.medNormInt.plateScale.medNormSMP.20260211_organoids.adat",
    "Media": ORG / "somalogic" / "SS-25153695_v5.0_other.hybNorm.medNormInt.plateScale.medNormSMP.20260126_media.adat"}
OLINK = ORG / "olink" / "olink_slim.csv"
MS_PG = ORG / "MS" / "Human_report.pg_matrix.tsv"
GEN_META = ORG / "gen_meta.csv"
MS_META = ORG / "MS_meta.csv"

PRESENT_FRAC = 0.70
SOMA_MAD_K = 3.0
OLINK_SD_K = 3.0
OLINK_MIN_COUNT = 150
MS_ANNOT = ["Protein.Group", "Protein.Names", "Genes", "First.Protein.Description",
            "N.Sequences", "N.Proteotypic.Sequences"]
ARMS = ("Organoid", "Media")


# --- Helpers---
def meta_affinity(company):
    gm = pd.read_csv(GEN_META, dtype=str)
    gm = gm[gm["Company"].str.lower() == company.lower()]
    return gm.rename(columns={"ID": "sample", "Cell line": "line", "Sample type": "arm",
                              "Status": "genotype"})[["sample", "line", "arm", "genotype"]]


def load_soma(arm, notes):
    vals, rm, cm = read_adat(SOMA[arm])
    keep_cols = cm[(cm["Type"] == "Protein") & (cm["Organism"] == "Human")]["SeqId"].tolist()
    log2 = np.log2(vals[keep_cols].where(vals[keep_cols] > 0))
    buffers = rm["SampleType"] == "Buffer"
    samples = rm["SampleType"] == "Sample"
    b = log2[buffers.values]
    med = b.median(axis=0)
    mad = (b - med).abs().median(axis=0) * 1.4826
    lod = med + SOMA_MAD_K * mad
    gm = meta_affinity("SomaScan")
    gm = gm[gm["arm"] == arm]
    sid = rm.loc[samples, "SampleId"].astype(str)
    matched = sid.isin(gm["sample"])
    unmatched = sid[~matched].tolist()
    notes.append(f"SomaScan {arm} samples {int(samples.sum())}, buffers {int(buffers.sum())}, "
                 f"human protein aptamers {len(keep_cols):,}, matched to gen_meta {int(matched.sum())}"
                 + (f", dropped unmatched {unmatched}" if unmatched else ""))
    x = log2[samples.values].copy()
    x.index = sid.values
    x = x.loc[matched.values]
    long = (x.reset_index().melt(id_vars="index", var_name="feature_id", value_name="value")
            .rename(columns={"index": "sample"}))
    long["above_lod"] = (long["value"].values > lod.reindex(long["feature_id"]).values).astype(int)
    # LoD sensitivity (item 24): presence under k = 3, 5, 10 MAD, same 70% rule
    for k in (3, 5, 10):
        lod_k = med + k * mad
        pres_k = ((x > lod_k).mean(axis=0) >= PRESENT_FRAC).sum()
        notes.append(
            f"[SomaScan {arm}] LoD sensitivity k={k}: aptamers present {int(pres_k):,} of {len(keep_cols):,} ({pres_k / len(keep_cols):.1%})")
    long = long.merge(gm, on="sample", how="left")
    ann = cm.set_index("SeqId").loc[keep_cols, ["UniProt", "EntrezGeneSymbol"]].reset_index()
    ann.columns = ["feature_id", "uniprot", "gene"]
    long = long.merge(ann, on="feature_id", how="left")
    long["platform"] = "SomaScan"
    lod_tab = pd.DataFrame({"feature_id": lod.index, "lod": lod.values, "platform": "SomaScan", "arm": arm})
    return long, lod_tab


def load_olink(notes):
    if not OLINK.exists():
        notes.append(f"[Olink] {OLINK.name} not found, Olink skipped")
        return pd.DataFrame(), pd.DataFrame()
    o = pd.read_csv(OLINK, dtype={"SampleName": str})
    o = o[(o["AssayType"] == "assay")]
    nc = o[o["SampleType"].str.upper().str.contains("NEGATIVE", na=False)]
    g = nc.groupby("OlinkID")["NPX"]
    lod = (g.median() + OLINK_SD_K * g.std(ddof=1).fillna(0)).rename("lod")
    notes.append(f"Olink negative-control samples {nc['SampleName'].nunique()}, "
                     f"LoD = median + {OLINK_SD_K:.0f} SD per assay, plus Count >= {OLINK_MIN_COUNT}")
    s = o[(o["SampleType"].str.upper() == "SAMPLE") & (o["SampleQC"] == "PASS") & (o["AssayQC"] == "PASS")].copy()
    gm = meta_affinity("OLink")
    matched = s["SampleName"].isin(gm["sample"])
    unmatched = sorted(s.loc[~matched, "SampleName"].unique().tolist())
    s = s[matched]
    notes.append(f"Olink assays {s['OlinkID'].nunique():,}, samples matched {s['SampleName'].nunique()}"
                 + (f", dropped unmatched {unmatched}" if unmatched else ""))
    s = s.rename(columns={"SampleName": "sample", "OlinkID": "feature_id", "UniProt": "uniprot",
                          "Assay": "gene", "NPX": "value"})
    s = s.merge(gm, on="sample", how="left")
    above = s["Count"] >= OLINK_MIN_COUNT
    if lod is not None:
        above &= s["value"].values > lod.reindex(s["feature_id"]).fillna(-np.inf).values
    s["above_lod"] = above.astype(int)
    s["platform"] = "Olink"
    long = s[["platform", "arm", "sample", "line", "genotype", "feature_id", "uniprot", "gene", "value", "above_lod"]]
    lod_tab = (pd.DataFrame({"feature_id": lod.index, "lod": lod.values, "platform": "Olink", "arm": "both"})
               if lod is not None else pd.DataFrame())
    return long, lod_tab


def load_ms(notes):
    pg = pd.read_csv(MS_PG, sep="\t")
    mm = pd.read_csv(MS_META, header=None, dtype=str)
    mm = pd.DataFrame({"raw": mm[0], "line": mm[3], "arm": mm[4], "genotype": mm[5]})
    mm["sample"] = mm["raw"].str.replace("\\", "/", regex=False).str.split("/").str[-1].str.replace(".raw", "",
                                                                                                    regex=False)
    mm = mm[mm["genotype"].isin(["Control", "AD"])]
    sc = [c for c in pg.columns if c not in MS_ANNOT]
    short = {c: c.replace("\\", "/").split("/")[-1].replace(".raw", "") for c in sc}
    keep = [c for c in sc if short[c] in set(mm["sample"])]
    dropped = [short[c] for c in sc if c not in keep]
    notes.append(f"[MS] protein groups {len(pg):,}, sample columns {len(sc)}, kept {len(keep)}"
                 + (f", dropped {dropped}" if dropped else ""))
    v = pg[keep].apply(pd.to_numeric, errors="coerce")
    v.columns = [short[c] for c in keep]
    v = v.where(v > 0)
    long = pd.concat([pd.DataFrame({"sample": col, "feature_id": pg["Protein.Group"], "value": np.log2(v[col]),
                                    "above_lod": v[col].notna().astype(int)}) for col in v.columns],
                     ignore_index=True)
    long = long.merge(mm[["sample", "line", "arm", "genotype"]], on="sample", how="left")
    ann = pd.DataFrame({"feature_id": pg["Protein.Group"],
                        "uniprot": pg["Protein.Group"].str.split(";").str[0],
                        "gene": pg["Genes"].astype(str).str.split(";").str[0],
                        "n_sequences": pg["N.Sequences"], "n_proteotypic": pg["N.Proteotypic.Sequences"]})
    long = long.merge(ann, on="feature_id", how="left")
    long["platform"] = "MS"
    return long


def summarise(long):
    def agg(g):
        return pd.Series({
            "n_samples": len(g),
            "det_rate": g["above_lod"].mean(),
            "det_rate_TDR1": g.loc[g["line"] == "TDR1", "above_lod"].mean(),
            "det_rate_TDR4": g.loc[g["line"] == "TDR4", "above_lod"].mean(),
            "mean_value": g["value"].mean(),
            "mean_value_detected": g.loc[g["above_lod"] == 1, "value"].mean(),
        })

    keys = ["platform", "arm", "feature_id", "uniprot", "gene", "ensg"]
    extra = [c for c in ("n_sequences", "n_proteotypic") if c in long.columns]
    f = long.groupby(keys, dropna=False)[["above_lod", "value", "line"]].apply(agg).reset_index()
    if extra:
        f = f.merge(long[keys + extra].drop_duplicates(keys), on=keys, how="left")
    f["present"] = (f["det_rate"] >= PRESENT_FRAC).astype(int)
    f["present_TDR1"] = (f["det_rate_TDR1"] >= PRESENT_FRAC).astype(int)
    f["present_TDR4"] = (f["det_rate_TDR4"] >= PRESENT_FRAC).astype(int)
    return f


def reconcile(feat, notes):
    for plat, tag in (("SomaScan", "soma"), ("Olink", "olink")):
        for arm in ARMS:
            p = ORG / f"{tag}_{arm.lower()}_detection.csv"
            if not p.exists():
                continue
            r = pd.read_csv(p)
            f = feat[(feat["platform"] == plat) & (feat["arm"] == arm)]
            if plat == "SomaScan":
                m = f.merge(r, on=["uniprot", "gene"], how="inner", suffixes=("", "_r"))
            else:
                m = f.groupby("uniprot", as_index=False).agg(mean_value=("mean_value", "mean"),
                                                             det_rate=("det_rate", "max"))
                m = m.merge(r, on="uniprot", how="inner")
            if len(m) < 10:
                notes.append(f"[reconcile {plat} {arm}] too few matched rows ({len(m)})")
                continue
            d = (m["mean_value"] - m["mean_abundance"]).abs()
            notes.append(f"[reconcile {plat} {arm}] n={len(m):,}  mean abundance |diff| median {d.median():.4f} "
                         f"max {d.max():.3f}   R detection_rate mean {m['detection_rate'].mean():.3f}  "
                         f"LoD-based det_rate mean {m['det_rate'].mean():.3f}  present {int((m['det_rate'] >= PRESENT_FRAC).sum()):,}")


notes = []
resolve = build_resolver()
parts, lods = [], []
for arm in ARMS:
    if SOMA[arm].exists():
        l, ld = load_soma(arm, notes)
        parts.append(l)
        lods.append(ld)
    else:
        notes.append(f"[SomaScan {arm}] file missing, skipped")
ol, old = load_olink(notes)
if len(ol):
    parts.append(ol)
    lods.append(old)
parts.append(load_ms(notes))
long = pd.concat(parts, ignore_index=True)
long["ensg"] = [resolve(u, g) for u, g in zip(long["uniprot"], long["gene"])]
cols = ["platform", "arm", "sample", "line", "genotype", "feature_id", "uniprot", "gene", "ensg",
        "value", "above_lod"] + [c for c in ("n_sequences", "n_proteotypic") if c in long.columns]
long = long[cols]
long.to_csv(TAB_DIR / "organoid_long.tsv", sep="\t", index=False)

feat = summarise(long)
feat.to_csv(TAB_DIR / "organoid_features.tsv", sep="\t", index=False)
reconcile(feat, notes)

L = ["Organoids: raw-file rebuild", ""]
L += notes
L.append("")
L.append("sample table (platform x arm x line):")
st = long.groupby(["platform", "arm", "line", "genotype"])["sample"].nunique().reset_index()
L.append("  " + st.to_string(index=False).replace("\n", "\n  "))
L.append("")
L.append(f"per-platform features and presence (above LoD in >= {PRESENT_FRAC:.0%} of samples):")
for (plat, arm), g in feat.groupby(["platform", "arm"]):
    ens = g[g["present"] == 1]["ensg"].dropna().nunique()
    L.append(f"{plat:9s} {arm:9s} features {len(g):6,}  present {int(g['present'].sum()):6,}  "
             f"ENSG-mapped present {ens:6,}  ENSG map rate {g['ensg'].notna().mean():.1%}  "
             f"present in both lines {int((g['present_TDR1'] & g['present_TDR4']).sum()):6,}  "
             f"one line only {int((g['present_TDR1'] ^ g['present_TDR4']).sum()):5,}")
L.append("")
if lods:
    lt = pd.concat(lods, ignore_index=True)
    for (plat, arm), g in lt.groupby(["platform", "arm"]):
        L.append(f"  LoD {plat} {arm}: median {g['lod'].median():.2f} (log2 units), IQR "
                 f"{g['lod'].quantile(0.25):.2f}-{g['lod'].quantile(0.75):.2f}")
text = "\n".join(L)
(TAB_DIR / "organoid_platforms.txt").write_text(text)
print(text)
