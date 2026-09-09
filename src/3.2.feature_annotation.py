"""
Gene-level features from four resources computed from the
raw downloads.
  CORUM        complex membership
  TargetScan   conserved miRNA sites
  sORFs.org    uORF count, lengths, PhastCons
  ENCODE eCLIP narrowPeak BED files + gene coordinates
"""
from __future__ import annotations
import sys
import pandas as pd
from config import DATA_DIR, TAB_DIR
from utils import symbol_to_ensg
from utils_features import corum, targetscan, uorfs, eclip

CORUM = DATA_DIR / "corum_human_complexes.txt"
TS_SUMMARY = DATA_DIR / "Summary_Counts.default_predictions.txt"
TS_GENEINFO = DATA_DIR / "Gene_info.txt"
SORFS = DATA_DIR / "sorfs_human_5utr.csv"
GENE_COORDS = DATA_DIR / "gene_coordinates_grch38.tsv"
ECLIP_DIR = DATA_DIR / "eclip_beds"
HUMAN_TAXID = 9606


# this is for cache option in eclip func. It takes a while to pull 750 files from the internet.
refresh = "--refresh-eclip" in sys.argv

gd = pd.read_csv(TAB_DIR / "gene_dict.tsv", sep="\t", dtype=str)
sym2ensg = symbol_to_ensg(gd, tag="annotation")

tables = [t for t in (corum(sym2ensg, CORUM), targetscan(sym2ensg, TS_SUMMARY, TS_GENEINFO, HUMAN_TAXID), uorfs(SORFS), eclip(ECLIP_DIR, GENE_COORDS, refresh)) if t is not None]
if not tables:
    raise SystemExit("no annotation inputs found under data/")
feats = tables[0]
for t in tables[1:]:
    feats = feats.merge(t, on="ensg", how="outer")
    # absence matters here: a gene without a CORUM entry is not in a
    # complex, a gene without a peak has no measured binding
fill0 = ["is_in_complex", "n_complexes", "in_large_complex", "n_conserved_mirna_sites", "n_mirna_families",
             "uorf_count", "has_uorf", "rbp_n_binding_rbps", "rbp_hur_binding", "rbp_hnrnp_binding",
             "rbp_pumilio_binding", "rbp_destab_binding"]
for c in fill0:
    if c in feats.columns:
        feats[c] = feats[c].fillna(0)
feats.to_csv(TAB_DIR / "features_annotation.tsv", sep="\t", index=False)
print(f"features_annotation: {len(feats):,} genes, {feats.shape[1] - 1} columns")

