"""
Configs and general setup for the project
"""
from __future__ import annotations
from pathlib import Path

# --- where the data lives
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RES_DIR = PROJECT_DIR / "results"
ML_DIR = RES_DIR / "ml"
TAB_DIR = RES_DIR / "tables"
PIC_DIR = RES_DIR / "pictures"
QC_DIR = TAB_DIR / "qc"


# --- Sources of protein data (all open access)
GTEX_MEDIAN = DATA_DIR / "gtex_protein_tissue_median.csv"
GTEX_TS = DATA_DIR / "gtex_protein_ts_score.csv"
IHC_FILE = DATA_DIR / "normal_ihc_data.tsv"
PAXDB_DIR = DATA_DIR / "paxdb"

# --- gene key -------------------------------------------------------------
# ENSG is the unifying key. GTEx is ENSG-native, HPA is ENSG-native, PaxDb is
# ENSP and is mapped to ENSG by HGNC symbol via the HPA gene dictionary (load_ihc.py).
GENE_KEY = "ensg"

# The tri-source intersection is 10 tissues. kidney + lymph_node have no
# GTEx-MS column, so they are IHC+PaxDb only (gtex = None).
#
# SUBSITE_AGG controls how multi-column GTEx tissues collapse: "mean" or "first".
SUBSITE_AGG = "mean"

TISSUES = {
    "cerebral_cortex": dict(paxdb="cerebral-cortex", gtex=["Brain Cortex"], hpa="Cerebral cortex"),
    "colon": dict(paxdb="colon", gtex=["Colon Sigmoid", "Colon Transverse"], hpa="Colon"),
    "heart": dict(paxdb="heart", gtex=["Heart Atrial", "Heart Ventricle"], hpa="Heart muscle"),
    "liver": dict(paxdb="liver", gtex=["Liver"], hpa="Liver"),
    "lung": dict(paxdb="lung", gtex=["Lung"], hpa="Lung"),
    "pancreas": dict(paxdb="pancreas", gtex=["Pancreas"], hpa="Pancreas"),
    "skin": dict(paxdb="skin", gtex=["Skin Unexpo", "Skin SunExpo"], hpa="Skin"),
    "spleen": dict(paxdb="spleen", gtex=["Spleen"], hpa="Spleen"),
    "stomach": dict(paxdb="stomach", gtex=["Stomach"], hpa="Stomach"),
    "testis": dict(paxdb="testis", gtex=["Testis"], hpa="Testis"),
    # two source organs (no gtex):
    "kidney": dict(paxdb="kidney", gtex=None, hpa="Kidney"),
    "lymph_node": dict(paxdb="lymph_node", gtex=None, hpa="Lymph node")
}

# --- HPA IHC Level handling ----------------------------------------------
# present = positive call, absent = real negative, NA = unscored/unknown.
# "Ascending"/"Descending" are non-standard -> treated as NA
IHC_PRESENT = {"Low", "Medium", "High"}
IHC_ABSENT = {"Not detected"}
IHC_NA = {"N/A", "Not representative", "Ascending", "Descending"}

# reliability confidence, high -> low
RELIABILITY_ORDER = ["Enhanced", "Supported", "Approved", "Uncertain"]
LEVEL_ORD = {"Low": 1, "Medium": 2, "High": 3}

# GTEx missingness tokens
GTEX_NA = {"NA", "NaN", ""}

# "longest" || "shortest" isoform sensitivity
ISOFORM_PICK = "longest"

# paxdb related variable. whole body ("global") vs "tissue"
ABUNDANCE_SOURCE = "global"

# Features for ML
GROUPS = {
    "size":            ["length", "mw_da", "mature_length"],
    "hydrophobicity":  ["gravy"],
    "transmembrane":   ["n_tm"],
    "tryptic_yield":   ["n_tryptic_7_30", "tryptic_per_kda"],
    "composition":     ["cysteine_fraction", "charged_fraction", "proline_fraction",
                        "glycine_fraction", "aromatic_fraction",
                        "low_complexity_fraction", "disorder_fraction_proxy"],
    "glycosylation":   ["n_glyco_motif_count"],
    "ptm":             ["n_glyco_sites", "n_disulfide", "n_lipid_sites",
                        "has_propeptide", "pep_frac_ptm_site", "pep_frac_disulfide"],
    "degradation_pest": ["pest_score_max", "pest_score_sum", "n_pest_regions", "pest_fraction"],
    "signal_secreted": ["has_signal", "is_secreted"],
    "localization":    ["is_membrane", "is_nuclear", "is_cytoplasmic", "is_mitochondrial",
                        "is_er_golgi", "is_cytoskeleton", "is_projection",
                        "n_compartments", "is_multilocal",
                        "is_ecm", "is_chromatin", "is_lysosome", "is_peroxisome", "is_endosome",
                        "is_vesicle", "is_lipid_droplet"],
    "structure_af":    ["af_plddt_mean", "af_frac_lt50", "af_frac_lt70", "af_frac_gt90", "af_longest_lt50_run"],
    "topology":        ["topo_multipass", "topo_single_type1", "topo_single_type2", "topo_single_other",
                        "topo_peripheral", "topo_lipid_anchor", "topo_extracellular_aa", "topo_cytoplasmic_aa"],
    "complex":         ["is_in_complex", "n_complexes", "complex_size_mean",
                        "complex_size_max", "in_large_complex"],
    "charge_pI":       ["pI"],
    "proteotypicity":  ["pep_frac_unique", "pep_frac_shared", "pep_max_shared_genes",
                        "pep_n_unique", "pep_n_clean", "pep_clean_per_kda", "pep_coverage_clean"],
    "peptide_chemistry": ["pep_frac_met", "pep_frac_cys", "pep_frac_sequon", "pep_frac_slow_cleave",
                          "pep_gravy_mean", "pep_frac_hydrophilic", "pep_frac_hydrophobic",
                          "pep_max_gap", "pep_coverage_7_30"],
}


EXCLUDED = ["Transcript count", "Transcript length (including UTRs and CDS)", "exon_count",
            "manual_gc_content", "cai", "kozak_score", "utr5_length", "utr3_length",
            "cds_length", "utr5_gc", "utr3_gc", "utr3_to_cds_ratio",
            "n_conserved_mirna_sites", "n_mirna_families", "mirna_site_density",
            "uorf_count", "uorf_max_length_aa", "uorf_mean_length_aa", "has_uorf",
            "uorf_mean_phastcon", "rbp_n_binding_rbps", "rbp_hur_binding",
            "rbp_hnrnp_binding", "rbp_pumilio_binding", "rbp_destab_binding"]

NEVER = ["pe_level", "mature_start", "mature_end", "isoform_rule",
         "loc_n_clauses", "loc_n_observed", "loc_go_n_terms", "loc_source"]

OBSERVED_PTM = ["n_glyco_sites_observed", "n_mod_res_observed", "n_disulfide_observed",
                "n_lipid_sites_observed", "n_mod_res"]

HALF_LIFE_LEAK = ["hl_log2_hours"]


