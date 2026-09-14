"""
PeptideAtlas validation.

Independent empirical check on the chemistry mechanism. PeptideAtlas (human
build 502, 2021; 2,322 samples, multi-tissue) is an MS source independent of
GTEx. For each protein it records the distinct peptides actually observed across
all those experiments. This script counts observed distinct peptides per gene and
tests two things:
 - cross-source concordance: are proteins not observed in GTEx also
      under-observed in PeptideAtlas?

 - mechanism: does the in-silico tryptic peptide count predict the number of
      peptides actually observed, conditional on abundance and protein length?
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from config import DATA_DIR, TAB_DIR

PEP2PROT = DATA_DIR / "APD_ensembl_hits.tsv"
ENSP2ENSG = DATA_DIR / "ensp2ensg.txt"

ENSP_RE = re.compile(r"ENSP\d+")
ENSG_RE = re.compile(r"ENSG\d+")


def load_ensp2ensg():
    m = pd.read_csv(ENSP2ENSG, sep=None, engine="python", dtype=str)
    ensp_col = "Protein stable ID"
    ensg_col = "Gene stable ID"
    out = m[[ensp_col, ensg_col]].copy()
    out.columns = ["ensp", "ensg"]
    out["ensp"] = out["ensp"].str.extract(r"(ENSP\d+)")
    out["ensg"] = out["ensg"].str.extract(r"(ENSG\d+)")
    return out.dropna().drop_duplicates()


def observed_peptides():
    """Distinct observed peptides per ENSG from the PeptideAtlas pep2prot map."""
    # headerless; we only need peptide id (col 0) and protein (col 2)
    p2p = pd.read_csv(PEP2PROT, sep="\t", header=None, usecols=[0, 2],
                      names=["pep_id", "prot"], dtype=str)
    n_raw = len(p2p)
    p2p["ensp"] = p2p["prot"].str.extract(r"(ENSP\d+)")
    ensp_hit = p2p["ensp"].notna().mean()
    p2p = p2p.dropna(subset=["ensp"])

    mp = load_ensp2ensg()
    j = p2p.merge(mp, on="ensp", how="left")
    map_rate = j["ensg"].notna().mean()
    j = j.dropna(subset=["ensg"])

    # peptide -> how many distinct genes it maps to (for proteotypic flag)
    pep_ngene = j.groupby("pep_id")["ensg"].nunique()
    j["proteotypic"] = j["pep_id"].map(pep_ngene).eq(1)

    n_pep = j.groupby("ensg")["pep_id"].nunique().rename("pa_n_peptides")
    n_prot = (j[j["proteotypic"]].groupby("ensg")["pep_id"].nunique()
              .rename("pa_n_proteotypic"))
    per_gene = (n_pep.to_frame().join(n_prot)
                .fillna({"pa_n_proteotypic": 0}).reset_index())
    return per_gene, n_raw, ensp_hit, map_rate


def gene_level_gtex(out):
    """Per-gene GTEx detectability + abundance from model_table."""
    mt = pd.read_csv(out / "model_table.tsv", sep="\t")
    g = mt.groupby("ensg").agg(
        frac_measured=("gtex_detected", "mean"),
        n_obs=("gtex_detected", "size"),
        median_log_abundance=("log_abundance", "median"),
    ).reset_index()
    return g


def _spear(a, b):
    from scipy.stats import spearmanr
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return r, p, int(m.sum())


def _partial_spear(y, x, controls):
    """Spearman of y vs x after linear residualisation on controls (rank space)."""
    from scipy.stats import spearmanr
    d = pd.concat([y, x] + controls, axis=1).dropna()
    if len(d) < 50:
        return np.nan, np.nan, len(d)
    R = d.rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c].values for c in d.columns[2:]])
    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    ry = resid(R[d.columns[0]].values)
    rx = resid(R[d.columns[1]].values)
    r, p = spearmanr(ry, rx)
    return r, p, len(d)


if __name__ == "__main__":
    feats = pd.read_csv(TAB_DIR / "features_protein.tsv", sep="\t")
    per_gene, n_raw, ensp_hit, map_rate = observed_peptides()
    gtex = gene_level_gtex(TAB_DIR)

    df = feats.merge(per_gene, on="ensg", how="left")
    df["pa_observed"] = df["pa_n_peptides"].notna().astype(int)
    df["pa_n_peptides"] = df["pa_n_peptides"].fillna(0)
    df["pa_n_proteotypic"] = df["pa_n_proteotypic"].fillna(0)
    df = df.merge(gtex, on="ensg", how="left")
    df["log_len"] = np.log10(df["length"].clip(lower=1))
    df["log_pa"] = np.log10(df["pa_n_peptides"].clip(lower=1) + 1)
    df["log_tryptic"] = np.log10(df["n_tryptic_7_30"].clip(lower=1))
    df.to_csv(TAB_DIR / "peptideatlas_per_gene.tsv", sep="\t", index=False)

    L = ["PEPTIDEATLAS VALIDATION (human build 502, independent of GTEx)", ""]
    L += ["coverage:",
          f"pep2prot rows: {n_raw:,}   rows with ENSP: {ensp_hit:.1%}",
          f"ENSP->ENSG mapping rate: {map_rate:.1%}",
          f"genes with >=1 observed peptide: {int(df['pa_observed'].sum()):,} "
          f"/ {len(df):,} ({df['pa_observed'].mean():.1%})",
          f"median observed peptides (observed genes): "
          f"{df.loc[df['pa_observed']==1,'pa_n_peptides'].median():.0f}", ""]

    # cross source concordance
    has_gtex = df["frac_measured"].notna()
    L.append("1. cross-source: GTEx vs PeptideAtlas")
    r, p, n = _spear(df.loc[has_gtex, "frac_measured"], df.loc[has_gtex, "log_pa"])
    L.append(f"Spearman(frac_measured GTEx, log PeptideAtlas peptides): r={r:+.3f} p={p:.1e} n={n:,}")
    # genes IHC-present but GTEx-dark everywhere vs detectable
    dark = df[has_gtex & (df["frac_measured"] == 0)]
    bright = df[has_gtex & (df["frac_measured"] > 0)]
    L.append(f"median PeptideAtlas peptides:  GTEx-dark genes {dark['pa_n_peptides'].median():.0f}"
             f"  vs GTEx-detectable {bright['pa_n_peptides'].median():.0f}")
    L.append(f"fraction never observed in PeptideAtlas:  GTEx-dark {1-dark['pa_observed'].mean():.1%}"
             f"  vs GTEx-detectable {1-bright['pa_observed'].mean():.1%}")
    L.append("")

    # mechanism: in-silico tryptic predicts observed peptides
    obs = df[df["pa_observed"] == 1]
    L.append("2. mechanism: in-silico tryptic yield vs observed peptide count (observed genes)")
    r, p, n = _spear(obs["log_tryptic"], obs["log_pa"])
    L.append(f"Spearman(log in-silico tryptic, log observed): r={r:+.3f} p={p:.1e} n={n:,}  [partly length-driven]")
    r, p, n = _partial_spear(obs["log_pa"], obs["log_tryptic"],
                             [obs["median_log_abundance"], obs["log_len"]])
    L.append(f"partial (control abundance + length): r={r:+.3f} p={p:.1e} n={n:,}")
    if "tryptic_per_kda" in obs.columns:
        r, p, n = _partial_spear(obs["log_pa"], obs["tryptic_per_kda"],
                                 [obs["median_log_abundance"], obs["log_len"]])
        L.append(f"size-normalised tryptic_per_kda (control abundance+length): r={r:+.3f} p={p:.1e} n={n:,}")

    text = "\n".join(L)
    (TAB_DIR / "peptideatlas_validation.txt").write_text(text)
    print(text)
