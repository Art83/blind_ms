"""
Functions for project
"""
from typing import Tuple, Any

import pandas as pd
from numpy import ndarray, dtype
from pandas import Series, DataFrame

from config import TISSUES, SUBSITE_AGG, IHC_ABSENT, IHC_PRESENT, RELIABILITY_ORDER, PAXDB_DIR


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



def _attach_ensg(df: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    sym2ensg = symbol_to_ensg(cross, tag="paxdb")
    df = df.copy()
    df["ensg"] = df["symbol"].map(sym2ensg)
    return df


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


# --- Paxdb processing
def _read_paxdb_file(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#",
                     names=["symbol", "string_id", "paxdb_ppm"], dtype=str)
    df["paxdb_ppm"] = pd.to_numeric(df["paxdb_ppm"], errors="coerce")
    df["ensp"] = df["string_id"].str.replace(r"^\d+\.", "", regex=True)  # drop 9606.
    df = df.drop(columns=["string_id"]).dropna(subset=["paxdb_ppm"])
    return df


def load_paxdb(cross: pd.DataFrame):
    rows = []
    for tname, spec in TISSUES.items():
        f = PAXDB_DIR / f"{spec['paxdb']}.txt"
        if not f.exists():
            print(f"  [skip] {tname}: {f.name} not present")
            continue
        d = _attach_ensg(_read_paxdb_file(f), cross)
        d["tissue"] = tname
        rows.append(d[["ensg", "tissue", "paxdb_ppm", "ensp", "symbol"]])
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    wb_path = PAXDB_DIR / "hs_whole_body.txt"
    wb = _attach_ensg(_read_paxdb_file(wb_path), cross)
    wb = wb.rename(columns={"paxdb_ppm": "paxdb_ppm_global"})[
            ["ensg", "paxdb_ppm_global", "ensp", "symbol"]]
    return long, wb






# QC utils
def audit(out):
    import numpy as np
    gd = pd.read_csv(out / "gene_dict.tsv", sep="\t", dtype=str)
    sym2ensg = (gd.dropna(subset=["symbol", "ensg"]).drop_duplicates("symbol")
                  .set_index("symbol")["ensg"])

    wb = _read_paxdb_file(PAXDB_DIR / "hs_whole_body.txt")
    wb = wb[wb["paxdb_ppm"] > 0].copy()
    wb["ensg"] = wb["symbol"].map(sym2ensg)
    wb["mapped"] = wb["ensg"].notna()
    wb["log_ppm"] = np.log10(wb["paxdb_ppm"])
    return wb


def compare(wb) -> str:
    m = wb.loc[wb["mapped"], "log_ppm"]
    u = wb.loc[~wb["mapped"], "log_ppm"]
    L = ["Mapping check: PaxDb whole body abundance, mapped vs unmapped", "",
         f"total proteins (ppm>0): {len(wb):,}",
         f"mapped to ENSG:   {len(m):,} ({len(m) / len(wb):.1%})",
         f"unmapped (dropped): {len(u):,} ({len(u) / len(wb):.1%})", ""]
    if len(u) < 20 or len(m) < 20:
        L.append("too few in one group to test")
        return "\n".join(L)

    def q(s):
        return (f"median {s.median():+.2f}  IQR [{s.quantile(.25):+.2f}, "
                f"{s.quantile(.75):+.2f}]  (log10 ppm)")
    L.append(f"mapped:   {q(m)}")
    L.append(f"unmapped: {q(u)}")
    L.append("")

    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(m, u, alternative="two-sided")
    rbc = 1 - 2 * U / (len(m) * len(u))
    direction = ("unmapped lower abundance" if m.median() > u.median()
                 else "unmapped higher abundance" if m.median() < u.median()
                 else "no median difference")
    L.append(f"Mann-Whitney U: p={p:.2e} rank-biserial={rbc:+.3f}   ({direction})")
    L.append("")
    mag = abs(rbc)
    band = ("negligible" if mag < 0.1 else "small" if mag < 0.3
            else "moderate" if mag < 0.5 else "large")
    L.append(f"effect size is {band}.")
    if mag < 0.1:
        L.append("=> unmapped proteins are not materially different in abundance;")
        L.append("the 16% loss is ignorable w.r.t. the dominant detectability axis.")
    else:
        lo = "lower" if m.median() > u.median() else "higher"
        L.append(f"=> unmapped proteins skew {lo}-abundance dropping them makes the")
        L.append(f"blind-spot estimate {'conservative' if lo=='lower' else 'anti-conservative'}.")
    return "\n".join(L)