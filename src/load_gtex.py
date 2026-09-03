"""
Convert GTEx-MS (Jiang, et al., 2020 Cell) into a long table.

Key vars (important):
  gtex_measured = True  -> protein quantified in this tissue (real value)
  gtex_measured = False -> protein is in the GTEx matrix but NA in this tissue
                           (measured-absent: detected elsewhere, not here)
  A protein absent from the matrix entirely is "not-measured" and simply does
  not appear in gtex_long. build_grid.py turns that distinction into the
  not-measured vs measured-absent strata.
"""
from __future__ import annotations
import pandas as pd
from config import TAB_DIR, GTEX_MEDIAN, GTEX_TS

from utils import load_gtex


# --- data ---
med = pd.read_csv(GTEX_MEDIAN, dtype=str)
ts = pd.read_csv(GTEX_TS, dtype=str, usecols=["ensembl_id", "entrez_id", "hgnc_name", "hgnc_symbol"])

long, cross = load_gtex(med, ts)
long.to_csv(TAB_DIR / "gtex_long.tsv", sep="\t", index=False)
cross.to_csv(TAB_DIR / "gtex_crosswalk.tsv", sep="\t", index=False)

# --- Output ---
n_meas = int(long["gtex_measured"].sum())
print(f"gtex_long: {len(long):,} rows  ({long['tissue'].nunique()} tissues)")
print(f"measured (real value): {n_meas:,}")
print(f"measured-absent (NA in tissue): {len(long) - n_meas:,}")
print(f"gtex_crosswalk: {len(cross):,} rows, "
      f"{cross['symbol'].notna().sum():,} with symbol")
