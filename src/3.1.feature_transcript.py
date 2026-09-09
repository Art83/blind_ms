"""
Script produces transcript level features per gene. The input is Biomart export.


Inputs:
mart_export_features.txt  BioMart TSV: Gene stable ID, Transcript count,
                             Transcript length (including UTRs and CDS),
                             Ensembl Canonical, Exon stable ID

mart_export_fasta.txt   BioMart FASTA, cDNA of the Ensembl canonical
                             transcript, header = Gene stable ID

Filter: protein_coding
"""
from __future__ import annotations
import pandas as pd
from config import DATA_DIR, TAB_DIR
from utils_features import parse_cdna

ENSEMBL_DIR = DATA_DIR / "ensembl"
BIOMART_FEATURES = ENSEMBL_DIR / "mart_export_features.txt"
CDNA_FASTA = ENSEMBL_DIR / "mart_export_fasta.txt"


bm = pd.read_csv(BIOMART_FEATURES, sep="\t")
bm_feat = (bm.groupby("Gene stable ID")
                 .agg({"Transcript count": "first",
                       "Transcript length (including UTRs and CDS)": "first",
                       "Exon stable ID": "nunique"})
                 .rename(columns={"Exon stable ID": "exon_count"})
                 .reset_index().rename(columns={"Gene stable ID": "ensg"}))

print(f"BioMart features tsv: {len(bm):,} rows, {len(bm_feat):,} genes")

cdna = parse_cdna(CDNA_FASTA)
print(f"cDNA FASTA: {len(cdna):,} genes, ORF found for {int(cdna['cds_length'].notna().sum()):,}")

feats = bm_feat.merge(cdna, on="ensg", how="inner")
feats.to_csv(TAB_DIR / "features_transcript.tsv", sep="\t", index=False)
print(f"features_transcript: {len(feats):,} genes with BioMart and cDNA features")

