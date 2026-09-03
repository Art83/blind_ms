"""
Functions for project
"""
import pandas as pd
from config import TISSUES, SUBSITE_AGG

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