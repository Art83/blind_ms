"""
Load PaxDb into a long table, mapped to ENSG.

note: PaxDb is ENSP based. We attach ENSG via HGNC symbol using the HPA gene dictionary
written by load_ihc.py. ENSP is retained for QC.
"""
from __future__ import annotations

import sys

import pandas as pd
from config import PAXDB_DIR, TAB_DIR
from utils import load_paxdb

# gene dictionary from previous script
gd = TAB_DIR / "gene_dict.tsv"

if not PAXDB_DIR.exists():
    print("Paxdb folder is not found")
    sys.exit()

cross = pd.read_csv(gd, sep="\t", dtype=str)
long, wb = load_paxdb(cross)
long.to_csv(TAB_DIR / "paxdb_long.tsv", sep="\t", index=False)
wb.to_csv(TAB_DIR / "paxdb_wholebody.tsv", sep="\t", index=False)
mapped = long["ensg"].notna().mean() if len(long) else 0

# --- summary ---
print(f"paxdb_long: {len(long):,} rows across {long['tissue'].nunique()} tissues"
          if len(long) else "paxdb_long: empty")
print(f"symbol -> ENSG mapped: {mapped:.1%}")
print(f"paxdb_wholebody: {len(wb):,} proteins, "
              f"{wb['ensg'].notna().mean():.1%} mapped to ENSG")
