"""
Building a per-protein biophysics feature table keyed to MS-detectability model.

Features and why (subject to change):
  gravy            Kyte-Doolittle hydropathy  -> hydrophobic = MS-hostile
  length, mw_da    Length and mol weight (probably ratio) -> small proteins evade MS
  pI               isoelectric point -> chromatography effect, binding to resins
  n_tm             transmembrane region count  -> MS-hostile (aggregation, poor digest)
  has_signal       signal peptide present -> secreted/membrane routing, MS-hostile
  n_tryptic_7_30   in-silico tryptic peptides 7-30 aa -> MS optimum range
  tryptic_per_kda  peptide density -> few peptides = few chances to detect

  mature_start/end  mature chain
  n_glyco_sites   glycosylation sites
  n_mod_res       modified residues. Doesn't go to ML: 27% of MOD_RES
                  sites are transfers from orthologs observed
                  by MS
  n_disulfide     disulfide bonds.
  n_lipid_sites   lipidation sites.
  n_*_observed    the same counts restricted to experimental or large-scale
                  proteomics evidence (ECO:0000269, ECO:0007744). Mot feature
  has_propeptide  any propep annotation
  pe_level        Protein existence, 1-5
  localisation    from uniprot's Subcellular location [CC]
"""

from __future__ import annotations
from utils_features import build_features
from config import DATA_DIR, TAB_DIR

UNIPROT_TSV = DATA_DIR / "uniprot_human.tsv"
GENE_DICT = TAB_DIR / "gene_dict.tsv"
GENE_FEATURES = DATA_DIR / "gene_features.tsv"

feats = build_features(UNIPROT_TSV, GENE_DICT, GENE_FEATURES)
feats.to_csv(TAB_DIR / "features_protein.tsv", sep="\t", index=False)
# Summary for qc
print(f"features_protein: {len(feats):,} ENSG")
print(f"with TM>0: {int((feats['n_tm'] > 0).sum()):,}  "
      f"| with signal: {int(feats['has_signal'].sum()):,}  "
      f"| mature chain shorter than precursor: {int((feats['mature_length'] < feats['length']).sum()):,}")
print(f"predicted (features): glyco sites {int((feats['n_glyco_sites'] > 0).sum()):,}, "
      f"modified residues {int((feats['n_mod_res'] > 0).sum()):,}, "
      f"disulfides {int((feats['n_disulfide'] > 0).sum()):,}, "
      f"lipidation {int((feats['n_lipid_sites'] > 0).sum()):,}, "
      f"propeptide {int(feats['has_propeptide'].sum()):,}")
print(f"observed (never features): modified residues {int((feats['n_mod_res_observed'] > 0).sum()):,}, "
      f"glyco {int((feats['n_glyco_sites_observed'] > 0).sum()):,}, "
      f"disulfides {int((feats['n_disulfide_observed'] > 0).sum()):,}")
print(f"median length {feats['length'].median():.0f} aa, "
      f"median tryptic(7-30) {feats['n_tryptic_7_30'].median():.0f}")
