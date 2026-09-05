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
IHC_ABSENT  = {"Not detected"}
IHC_NA = {"N/A", "Not representative", "Ascending", "Descending"}

# reliability confidence, high -> low
RELIABILITY_ORDER = ["Enhanced", "Supported", "Approved", "Uncertain"]

# GTEx missingness tokens
GTEX_NA = {"NA", "NaN", ""}

# "longest" || "shortest" isoform sensitivity
ISOFORM_PICK = "longest"