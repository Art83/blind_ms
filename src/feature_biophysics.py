"""
Building a per-protein biophysics feature table keyed to MS-detectability model.

Features and why (subject to change):
  gravy            Kyte-Doolittle hydropathy  -> hydrophobic = MS-hostile
  length, mw_da    Length and mol weight (probably ratio) -> small proteins evade MS
  pI               isoelectric point -> chromatography effect, binding to resins
  n_tm            transmembrane region count  -> MS-hostile (aggregation, poor digest)
"""

from __future__ import annotations
import pandas as pd
from utils_features import _gravy, _ensg_from_xref, _pI
from config import DATA_DIR, TAB_DIR, ISOFORM_PICK
from utils import symbol_to_ensg
import uniprot_annot as UA
import re


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
    c_tm = "Transmembrane"

    sym2ensg = symbol_to_ensg(gene_dict, tag="uniprot")

    have_chain = all(c in list(df.columns) for c in ("Chain", "Propeptide"))

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
        rd = r.to_dict()
        m_start, m_end = UA.mature_range(rd, len(seq)) if have_chain else (1, len(seq))
        mature = seq[m_start - 1:m_end]
        feat = dict(
            ensg=ensg,
            uniprot=r[c_acc] if c_acc else "",
            length=length,
            mw_da=mw,
            gravy=round(_gravy(seq), 4),
            pI=_pI(mature),
            n_tm=len(re.findall(r"TRANSMEM", r[c_tm])) if c_tm else 0,
        )
        rows.append(feat)

    feats = pd.DataFrame(rows)
    return feats


print(build_features(UNIPROT_TSV, GENE_DICT))


