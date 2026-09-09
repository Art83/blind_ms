"""
Two additional feature sets (alphafold and half-life):

For alphafold derived features:
  af_plddt_mean        mean pLDDT
  af_frac_lt50         fraction of residues below 50 (AlphaFold's
                       own disordered threshold), the measured
                        replacement for disorder_fraction_proxy
  af_frac_lt70         fraction below 70 (low conf)
  af_frac_gt90         fraction above 90 (high conf)
  af_longest_lt50_run  longest run of residues below 50
  af_coverage          residues with a pLDDT / mature length

for half-life   source(Mathieson et al. 2018 (Nat Comms) ):
 hl_log2_hours        log2 of the median half-life across human replicates
 hl_n_values          how many replicate values went in.

Note: HL doesn't enter ML given that protein has measured HL only if MS detected it
"""
from __future__ import annotations
import re
import sys
import pandas as pd
from config import DATA_DIR, TAB_DIR
from utils_features import read_alphafold, af_summary, read_half_life
from utils import symbol_to_ensg

AF_DIR = DATA_DIR / "alphafold"
HALF_LIFE = DATA_DIR / "half_life_protein.csv"
_MEMBER = re.compile(r"AF-([A-Z0-9]+)-F(\d+)-model_v\d+\.cif\.gz$")
# setup ffor HL
HL_HUMAN_TYPES = ("Bcells", "NK cells", "Hepatocytes", "Monocytes")
HL_QUALITY_OK = {"good", "weak"}

fp = pd.read_csv(TAB_DIR / "features_protein.tsv", sep="\t", low_memory=False,
                 usecols=["ensg", "uniprot", "mature_start", "mature_end", "length"]).drop_duplicates("uniprot")
tar = AF_DIR / "UP000005640_9606_HUMAN_v6.tar"
L_res = ["STRUCTURE AND TURNOVER FEATURES", ""]
feats = fp[["uniprot", "ensg"]].copy()

# the tar processing is the slow step, so as with eclips
# summaries are cached (--refresh-alphafold to recompile)
af_cache = TAB_DIR / "alphafold_plddt_summary.tsv"
refresh = "--refresh-alphafold" in sys.argv
prev = TAB_DIR / "features_structure.tsv"

# --- AlphaFold ---
if not af_cache.exists() and not refresh and prev.exists():
    p = pd.read_csv(prev, sep="\t")
    af_cols = [c for c in p.columns if c.startswith("af_")]
    if af_cols and p["af_plddt_mean"].notna().any():
        p[["uniprot"] + af_cols].dropna(subset=["af_plddt_mean"]).to_csv(af_cache, sep="\t", index=False)
        print(f"AlphaFold cache seeded from the previous {prev.name}")
if af_cache.exists() and not refresh:
    af = pd.read_csv(af_cache, sep="\t")
    feats = feats.merge(af.drop(columns=[c for c in ("ensg",) if c in af.columns]), on="uniprot", how="left")
    L_res.append(f"AlphaFold: cached {af_cache.name}, {int(feats['af_plddt_mean'].notna().sum()):,} of {len(feats):,} "
                 f"UniProt-mapped genes covered")
    print("  " + L_res[-1])
else:
    print(f"AlphaFold reading {tar} (streaming, mmCIF members only)")
    pl, n_models = read_alphafold(tar, _MEMBER)
    rows = []
    for r in fp.itertuples(index=False):
        if r.uniprot in pl:
            rows.append(dict(uniprot=r.uniprot, **af_summary(pl[r.uniprot], int(r.mature_start), int(r.mature_end))))
    af = pd.DataFrame(rows)
    af.to_csv(af_cache, sep="\t", index=False)
    feats = feats.merge(af, on="uniprot", how="left")
    cov = feats["af_plddt_mean"].notna().mean()
    L_res.append(f"AlphaFold {tar.name}: {n_models:,} models read, {len(pl):,} accessions, "
                 f"{int(feats['af_plddt_mean'].notna().sum()):,} of {len(feats):,} UniProt-mapped genes covered ({cov:.1%})")
    L_res.append(
        f"median pLDDT {af['af_plddt_mean'].median():.1f}, median fraction below 50 {af['af_frac_lt50'].median():.3f}, "
        f"fragmented entries {int(sum(1 for a, d in pl.items() if max(d) > 1400)):,}")
    print("  " + L_res[-2])

# ---Half-life---
gd = pd.read_csv(TAB_DIR / "gene_dict.tsv", sep="\t", dtype=str)
hl = read_half_life(symbol_to_ensg(gd, tag="half-life"), HALF_LIFE, HL_HUMAN_TYPES, HL_QUALITY_OK)
if hl is not None:
    feats = feats.merge(hl, on="ensg", how="left")
    L_res.append(f"Half-life: {int(feats['hl_log2_hours'].notna().sum()):,} of "
                 f"{len(feats):,} genes covered ({feats['hl_log2_hours'].notna().mean():.1%}), "
                 f"median {2 ** feats['hl_log2_hours'].median():.0f} h")

feats.to_csv(TAB_DIR / "features_structure.tsv", sep="\t", index=False)
(TAB_DIR / "features_structure.txt").write_text("\n".join(L_res))
print("\n".join(L_res))
