"""
MS-detectability modeling (simple logistic regressions, all AUC are in-sample ones)

two lines of inquiry:
1. Does protein biophysics features predict MS-darkness in addition to abundance (if so, that's a chemistry signal)
2. Does fraction of cell types where proteins is detected predict MS-darkness (is so, this is "dilution" effect)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import TAB_DIR, TISSUES, ABUNDANCE_SOURCE
from utils_modeling import _zscore, fit, REL_ORD, feature_contrasts, _other_source, _swap_abundance



grid = pd.read_csv(TAB_DIR / "grid.tsv", sep="\t")
feats = pd.read_csv(TAB_DIR / "features_protein.tsv", sep="\t")

tri = {t for t, s in TISSUES.items() if s["gtex"] is not None}
m = grid[(grid["ihc_status"] == "present") & (grid["tissue"].isin(tri))].copy()
src = ABUNDANCE_SOURCE
t_ppm = m["paxdb_ppm"]
g_ppm = m["paxdb_ppm_global"]
if src == "global":
    ab = g_ppm.where(g_ppm.notna(), t_ppm)
    m["abundance_source"] = np.where(g_ppm.notna(), "global", "tissue")
else:
    ab = t_ppm.where(t_ppm.notna(), g_ppm)
    m["abundance_source"] = np.where(t_ppm.notna(), "tissue", "global")

m["abundance"] = ab
m = m[m["abundance"].notna() & (m["abundance"] > 0)].copy()
m["log_abundance"] = np.log10(m["abundance"])
m["log_abundance_tissue"] = np.log10(m["paxdb_ppm"].where(m["paxdb_ppm"] > 0))
m["log_abundance_global"] = np.log10(m["paxdb_ppm_global"].where(m["paxdb_ppm_global"] > 0))

print(f"abundance source = {src}: "
      f"{(m['abundance_source'] == src).mean():.1%} of rows from the primary source")

m = m.merge(feats, on="ensg", how="left")
before = len(m)
m = m[m["gravy"].notna()].copy()
feat_cov = len(m) / before if before else float("nan")

m["log_mw"] = np.log10(m["mw_da"].clip(lower=1))
m["log_tryptic"] = np.log10(m["n_tryptic_7_30"].clip(lower=1))
m["gtex_detected"] = (m["gtex_status"] == "measured").astype(int)
m["type"] = m["gtex_status"]
m["frac_pos_celltypes"] = m["frac_pos_celltypes"].fillna(1.0)

# IHC reliability as an ordinal antibody-QC covariate
m["reliability_ord"] = m["best_reliability"].map(REL_ORD)
m["reliability_ord"] = m["reliability_ord"].fillna(m["reliability_ord"].median())

# feature columns carried into model_table
gf_carry = ["disorder_fraction_proxy", "low_complexity_fraction",
            "n_glyco_motif_count", "charged_fraction", "proline_fraction",
            "is_membrane", "is_secreted", "is_nuclear", "n_compartments",
            "pest_score_max", "pest_fraction", "is_in_complex", "n_complexes",
            "complex_size_max", "cai", "kozak_score", "utr3_length",
            "n_conserved_mirna_sites", "uorf_count", "rbp_n_binding_rbps"]
keep = ["ensg", "symbol", "tissue", "gtex_detected", "type",
        "log_abundance", "log_abundance_tissue", "log_abundance_global", "abundance_source",
        "frac_pos_celltypes",
        "n_tm", "has_signal", "gravy", "log_mw", "log_tryptic",
        "best_reliability", "reliability_ord",
        "length", "mw_da", "n_tryptic_7_30", "pI", "paxdb_present"] + gf_carry
keep = [c for c in keep if c in m.columns]
tbl = m[keep]
tbl.to_csv(TAB_DIR / "model_table.tsv", sep="\t", index=False)

summ = [f"model_table: {len(tbl):,} IHC-present tri-source observations "
        f"({tbl['ensg'].nunique():,} genes)",
        f"abundance covariate: {ABUNDANCE_SOURCE} PaxDb ppm "
        f"({(tbl['abundance_source'] == ABUNDANCE_SOURCE).mean():.1%} primary, rest fallback)",
        f"biophysics feature coverage: {feat_cov:.1%}",
        f"outcome gtex_detected mean (=fraction MS-bright): {tbl['gtex_detected'].mean():.3f}",
        "by type:",
        tbl["type"].value_counts().to_string(),
        "",
        "=== feature contrasts by type ===",
        feature_contrasts(tbl).to_string(index=False),
        "",
        "=== models ===",
        fit(tbl, "pooled: measured vs (measured_absent|not_measured)"),
        "",
        fit(tbl[tbl["type"].isin(["measured", "not_measured"])],
            "Chemistry: measured vs not_measured"),
        "",
        fit(tbl[tbl["type"].isin(["measured", "measured_absent"])],
            "Dilution: measured vs measured_absent"),
        "",
        "=== abundance-source sensitivity (Chemistry, other PaxDb source) ===",
        fit(_swap_abundance(tbl[tbl["type"].isin(["measured", "not_measured"])]),
            f"Chemistry with {_other_source()} abundance"),
        ]
text = "\n".join(summ)
(TAB_DIR / "model_summary.txt").write_text(text)
print(text)
