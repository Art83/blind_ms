"""
Building a per-protein biophysics feature table keyed to MS-detectability model.

Features and why (subject to change):
  hydrop  Kyte-Doolittle hydropathy  -> hydrophobic = MS-hostile
  length
"""

from __future__ import annotations
import pandas as pd
from utils_features import _hydropathy, _ensg_from_xref
from config import DATA_DIR, TAB_DIR, ISOFORM_PICK
from utils import symbol_to_ensg


UNIPROT_TSV = DATA_DIR / "uniprot_human.tsv"
GENE_DICT = TAB_DIR / "gene_dict.tsv"


def build_features(path_uniprot, path_gene_dict) -> pd.DataFrame:
    df = pd.read_csv(path_uniprot, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    gene_dict = pd.read_csv(path_gene_dict, sep="\t", dtype=str)
    # Columns for the job
    # Annotation , ensg is a nightmare, need to parse
    c_acc = "Entry"
    c_gene = "Gene Names (primary)"
    c_ens = "Ensembl"

    # features
    c_seq = "Sequence"
    c_len = "Length"
    c_mass = "Mass"

    sym2ensg = symbol_to_ensg(gene_dict, tag="uniprot")

    rows = []
    n_mapped = 0
    for _, r in df.iterrows():
        seq = (r[c_seq] or "").strip().upper()
        if not seq:
            continue
        sym = (r[c_gene] or "").strip().split()[0] if r[c_gene] else None
        ensg = sym2ensg.get(sym) if sym else None
        if ensg is None and c_ens:
            xr = _ensg_from_xref(r[c_ens])
            ensg = xr[0] if xr else None
        if ensg is None:
            continue
        n_mapped += 1
        mw = pd.to_numeric(str(r[c_mass]).replace(",", ""), errors="coerce") if c_mass else float("nan")
        length = int(pd.to_numeric(r[c_len], errors="coerce")) if c_len and r[c_len] else len(seq)
        feat = dict(
            ensg=ensg,
            uniprot=r[c_acc] if c_acc else "",
            length=length,
            mw_da=mw,
            hydrop=round(_hydropathy(seq), 4)
        )
        rows.append(feat)

    feats = pd.DataFrame(rows)
    return feats


print(build_features(UNIPROT_TSV, GENE_DICT))


