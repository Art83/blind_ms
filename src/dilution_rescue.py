"""
Dilution rescue (Diagnostics).

The original dilution covariate in model_dark (frac_pos_celltypes from IHC alone) was
degenerate because IHC gives no cell-type proportions. This sctipt computes the
real covariate using Tabula Sapiens cell counts bridged to HPA cell types,
then reruns the dilution model AND reports two
diagnostics that decide whether dilution is real or unresolvable here.
"""
from __future__ import annotations
import pandas as pd
from config import DATA_DIR, TAB_DIR, TISSUES

PROP = DATA_DIR / "celltype_proportions.csv"
BRIDGE = DATA_DIR / "consensus_bridge_map.csv" # comes from hpa to tabula project
MIN_HPA_CELLTYPES = 2   # dilution resolution is bounded by IHC granularity
SPREAD_TISSUES = ["skin", "testis"]  # the only tri-source tissues with >2 HPA cell types


def build_covariate(out):
    prop = pd.read_csv(PROP).rename(columns={"ts_tissue": "TS_Tissue",
                                             "ts_cell_type": "TS_Cell_Type"})
    bridge = pd.read_csv(BRIDGE)
    ihc = pd.read_csv(out / "ihc_celltype_long.tsv", sep="\t")

    canon2hpa = {t: s["hpa"] for t, s in TISSUES.items() if s["gtex"] is not None}
    hpa2canon = {v: k for k, v in canon2hpa.items()}

    br = bridge[bridge["HPA_Tissue"].isin(canon2hpa.values())].copy()
    br = br.merge(prop, on=["TS_Tissue", "TS_Cell_Type"], how="left")
    br["tissue"] = br["HPA_Tissue"].map(hpa2canon)
    br = br.dropna(subset=["n_cells"])

    hpa_per_tissue = br.groupby("tissue")["HPA_Cell_Type"].nunique()
    ts_per_tissue = br.groupby("tissue")["TS_Cell_Type"].nunique()
    usable = set(hpa_per_tissue[hpa_per_tissue >= MIN_HPA_CELLTYPES].index)

    ihc = ihc[ihc["tissue"].isin(canon2hpa)].copy()
    ihc["is_pos"] = (ihc["ihc_call"] == "present")
    ihc["is_scored"] = ihc["ihc_call"].isin(["present", "absent"])

    # bridge coverage of IHC-positive calls + unbridged breakdown
    bridge_pairs = set(zip(br["tissue"], br["HPA_Cell_Type"]))
    ihc_pos = ihc[ihc["is_pos"]].copy()
    ihc_pos["matched"] = [(t, c) in bridge_pairs
                          for t, c in zip(ihc_pos["tissue"], ihc_pos["cell_type"])]
    cov = ihc_pos["matched"].mean() if len(ihc_pos) else float("nan")
    unbridged = (ihc_pos[~ihc_pos["matched"]]
                 .groupby(["tissue", "cell_type"]).size()
                 .reset_index(name="n_ihc_pos_calls")
                 .sort_values("n_ihc_pos_calls", ascending=False))

    j = br.merge(ihc[["ensg", "tissue", "cell_type", "is_pos", "is_scored"]],
                 left_on=["tissue", "HPA_Cell_Type"],
                 right_on=["tissue", "cell_type"], how="inner")
    ts = (j.groupby(["ensg", "tissue", "TS_Cell_Type", "n_cells"])
            .agg(expr=("is_pos", "any"), scored=("is_scored", "any"))
            .reset_index())

    def emf(g):
        scored = g[g["scored"]]
        den = scored["n_cells"].sum()
        num = scored[scored["expr"]]["n_cells"].sum()
        return pd.Series({"expressing_mass_fraction": num / den if den else float("nan"),
                          "n_scored_ts": len(scored),
                          "n_expr_ts": int((scored["expr"]).sum()),
                          "tissue_cells_scored": den})

    cova = ts.groupby(["ensg", "tissue"]).apply(emf).reset_index()
    cova["dilution_usable"] = cova["tissue"].isin(usable)
    counts = pd.DataFrame({"hpa_celltypes": hpa_per_tissue,
                           "tabula_celltypes": ts_per_tissue}).fillna(0).astype(int)
    return cova, cov, sorted(usable), counts, unbridged


def emf_variance(cova):
    ud = cova[cova["dilution_usable"]].copy()
    g = ud.groupby("tissue")["expressing_mass_fraction"]
    diag = pd.DataFrame({
        "n": g.size(),
        "median": g.median().round(3),
        "std": g.std().round(3),
        "frac_below_0.99": g.apply(lambda s: (s < 0.99).mean()).round(3),
    }).reset_index()
    return diag


def run_model(out, cova, tissues=None, label="usable tissues"):
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    tbl = pd.read_csv(out / "model_table.tsv", sep="\t")
    cu = cova[cova["dilution_usable"]]
    if tissues:
        cu = cu[cu["tissue"].isin(tissues)]
    dil = tbl[tbl["type"].isin(["measured", "measured_absent"])].merge(
        cu[["ensg", "tissue", "expressing_mass_fraction"]], on=["ensg", "tissue"], how="inner")
    dil = dil.dropna(subset=["expressing_mass_fraction"])

    lines = [f"--- DILUTION model [{label}] ---",
             f"rows {len(dil):,}  ({dil['tissue'].nunique()} tissues, {dil['ensg'].nunique():,} genes), "
             f"measured_absent frac {1 - dil['gtex_detected'].mean():.3f}",
             "emf by outcome (median | std):"]
    for o, s in dil.groupby("gtex_detected")["expressing_mass_fraction"]:
        lines.append(f"  gtex_detected={o}:  median {s.median():.3f}  std {s.std():.3f}")
    if dil["expressing_mass_fraction"].nunique() < 2 or len(dil) < 50:
        lines.append("emf has no usable variance here -> model skipped")
        return "\n".join(lines)

    feats = ["log_abundance", "expressing_mass_fraction", "log_tryptic", "n_tm"]
    z = dil.copy()
    for c in feats:
        sd = z[c].astype(float).std(ddof=0)
        z[c] = (z[c].astype(float) - z[c].astype(float).mean()) / sd if sd else 0.0
    tis = pd.get_dummies(z["tissue"], prefix="t", drop_first=True).astype(float)
    use = [c for c in feats if dil[c].nunique() > 1]
    X = sm.add_constant(pd.concat([z[use], tis], axis=1))
    y = dil["gtex_detected"].values
    f = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": dil["ensg"].values})
    lines.append(f"AUC {roc_auc_score(y, f.predict(X)):.4f}  | coef (gene-clustered SE):")
    for name in use:
        lines.append(f"  {name:>26s}  beta={f.params[name]:+.3f}  p={f.pvalues[name]:.2e}")
    return "\n".join(lines)


if __name__ == "__main__":
    cova, cov, usable, counts, unbridged = build_covariate(TAB_DIR)
    cova.to_csv(TAB_DIR / "dilution_covariate.tsv", sep="\t", index=False)
    unbridged.to_csv(TAB_DIR / "dilution_unbridged.tsv", sep="\t", index=False)

    L = [f"dilution_covariate: {len(cova):,} (ensg,tissue) rows",
         f"bridge coverage of IHC-positive calls: {cov:.1%}",
         f"dilution-usable tissues (>= {MIN_HPA_CELLTYPES} HPA cell types): {usable}",
         "cell types per tissue (HPA granularity bounds the test):",
         counts.to_string(), "",
         "=== (1) emf variance per usable tissue (is the covariate degenerate?) ===",
         emf_variance(cova).to_string(index=False), "",
         "=== (2) models ===",
         run_model(TAB_DIR, cova, None, "all usable (6 tissues)"), "",
         run_model(TAB_DIR, cova, SPREAD_TISSUES, "spread tissues only: skin+testis"), "",
         "=== (3) top unbridged IHC-positive cell types (the 46% miss) ===",
         unbridged.head(25).to_string(index=False)]
    text = "\n".join(L)
    (TAB_DIR / "dilution_summary.txt").write_text(text)
    print(text)
