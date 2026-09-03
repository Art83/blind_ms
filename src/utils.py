"""
Functions for project
"""
from typing import Tuple, Any

import pandas as pd
from numpy import ndarray, dtype
from pandas import Series, DataFrame

from config import TISSUES, SUBSITE_AGG, IHC_ABSENT, IHC_PRESENT, RELIABILITY_ORDER


# General handling
def symbol_to_ensg(cross, tag="", verbose=True):
    """symbol->ENSG map that reports ambiguity """
    c = (cross.dropna(subset=["symbol", "ensg"])[["symbol", "ensg"]]
              .drop_duplicates())
    dup = c["symbol"].duplicated(keep=False)
    n_ambig = c.loc[dup, "symbol"].nunique()
    n_alt_dropped = int(dup.sum()) - n_ambig
    if verbose:
        print(f"map {tag} {len(c):,} symbol-ENSG pairs; {n_ambig:,} symbols "
              f"are ambiguous (>1 ENSG); {n_alt_dropped:,} alt rows dropped (kept first)")
    return c.drop_duplicates("symbol").set_index("symbol")["ensg"]


def _agg_subsites(df: pd.DataFrame, cols: list[str], how: str) -> pd.Series:
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    return sub.mean(axis=1) if how == "mean" else sub[cols[0]]


# Gtex processing
def load_gtex(med: pd.DataFrame, ts:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    med = med.rename(columns={med.columns[0]: "ensg"})
    med["ensg"] = med["ensg"].str.strip()

    cross = ts.rename(columns={"ensembl_id": "ensg", "entrez_id": "entrez",
                               "hgnc_symbol": "symbol", "hgnc_name": "hgnc_name"})
    cross["ensg"] = cross["ensg"].str.strip()

    rows = []
    for tname, spec in TISSUES.items():
        gcols = spec["gtex"]
        if not gcols:
            continue
        missing = [c for c in gcols if c not in med.columns]
        if missing:
            raise KeyError(f"{tname}: GTEx columns not found: {missing}")
        level = _agg_subsites(med, gcols, SUBSITE_AGG)
        block = pd.DataFrame({"ensg": med["ensg"], "tissue": tname, "gtex_level": level})
        block["gtex_measured"] = block["gtex_level"].notna()
        rows.append(block)
    long = pd.concat(rows, ignore_index=True)
    return long, cross


# HPA processing
def _call_from_level(level: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=level.index, dtype="object")
    out[level.isin(IHC_PRESENT)] = "present"
    out[level.isin(IHC_ABSENT)] = "absent"
    return out


def load_ihc(df: pd.DataFrame) -> tuple[Any, Any, Any, Any]:
    df = df.rename(columns={"Gene": "ensg", "Gene name": "symbol",
                            "Tissue": "hpa_tissue", "Cell type": "cell_type",
                            "Level": "level", "Reliability": "reliability"})
    for c in df.columns:
        df[c] = df[c].str.strip()

    level_audit = df["level"].value_counts(dropna=False).rename_axis("level").reset_index(name="rows")

    gene_dict = (df[["ensg", "symbol"]].dropna().drop_duplicates().drop_duplicates("ensg"))

    hpa2canon = {spec["hpa"]: t for t, spec in TISSUES.items()}
    df = df[df["hpa_tissue"].isin(hpa2canon)].copy()
    df["tissue"] = df["hpa_tissue"].map(hpa2canon)
    df["ihc_call"] = _call_from_level(df["level"])

    celltype = df[["ensg", "tissue", "cell_type", "ihc_call", "level", "reliability"]].copy()

    # collapse cell types -> tissue level call
    rel_rank = {r: i for i, r in enumerate(RELIABILITY_ORDER)}
    scored = df[df["ihc_call"].notna()].copy()
    scored["is_pos"] = (scored["ihc_call"] == "present").astype(int)
    scored["rel_rank"] = scored["reliability"].map(rel_rank)

    g = scored.groupby(["ensg", "tissue"])
    tissue_tbl = g.agg(
        n_celltypes=("ihc_call", "size"),
        n_pos_celltypes=("is_pos", "sum"),
        best_rel_rank=("rel_rank", "min"),
    ).reset_index()
    tissue_tbl["ihc_present"] = tissue_tbl["n_pos_celltypes"] > 0
    tissue_tbl["frac_pos_celltypes"] = (tissue_tbl["n_pos_celltypes"] / tissue_tbl["n_celltypes"])
    inv_rank = {i: r for r, i in rel_rank.items()}
    tissue_tbl["best_reliability"] = tissue_tbl["best_rel_rank"].map(inv_rank)
    tissue_tbl = tissue_tbl.drop(columns=["best_rel_rank"])

    return gene_dict, celltype, tissue_tbl, level_audit