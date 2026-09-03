"""
Load HPA IHC into cell type resolved tables and make the
master ENSG<->symbol dictionary (HPA has the broadest gene coverage, so it's better if PaxDb
maps through it rather than through gtex).
"""
from __future__ import annotations
import pandas as pd
from config import IHC_FILE, TAB_DIR
from utils import load_ihc


NEED = ["Gene", "Gene name", "Tissue", "Cell type", "Level", "Reliability"]


# --- data ---
enc = "utf-8"
if not IHC_FILE.exists():
    FileNotFoundError("HPA file not found")

df = pd.read_csv(IHC_FILE, sep="\t", dtype=str, encoding=enc, keep_default_na=False, na_values=[])
missing = [c for c in NEED if c not in df.columns]
if missing:
    raise KeyError(f"IHC file is missing expected columns {missing}; "
                   f"found {list(df.columns)}")
df = df[NEED].copy()
gene_dict, celltype, tissue_tbl, level_audit = load_ihc(df)
gene_dict.to_csv( TAB_DIR / "gene_dict.tsv", sep="\t", index=False)
celltype.to_csv(TAB_DIR / "ihc_celltype_long.tsv", sep="\t", index=False)
tissue_tbl.to_csv(TAB_DIR / "ihc_tissue.tsv", sep="\t", index=False)
level_audit.to_csv(TAB_DIR / "ihc_level_audit.tsv", sep="\t", index=False)

# --- summary ---
print(f"gene_dict: {len(gene_dict):,} ENSG<->symbol pairs")
print(f"ihc_celltype_long: {len(celltype):,} rows, {celltype['tissue'].nunique()} tissues")
print(f"ihc_tissue: {len(tissue_tbl):,} gene x tissue calls")
print(f"present: {int(tissue_tbl['ihc_present'].sum()):,}")
print("level audit:")
print(level_audit.to_string(index=False))
