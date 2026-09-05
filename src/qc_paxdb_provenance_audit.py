"""
PaxDb provenance audit: how much of the abundance covariate is the label?

PaxDb integrated tissue datasets are weighted averages of public MS studies.
PXD016999 (Jiang et al. 2020) is the gtex-MS body-map proteome that is one of
this pipeline's detection labels, and it is one of paxdb's inputs for most of
the tissues. Where its weight is high, "conditioning on abundance"
is partly conditioning on the outcome.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import TAB_DIR, TISSUES, PAXDB_DIR, QC_DIR
from utils import _read_weights, _auc

JIANG_TAG = "PXD016999"

grid = pd.read_csv(TAB_DIR / "grid.tsv", sep="\t", low_memory=False)
tri = [t for t, s in TISSUES.items() if s["gtex"]]

rows = []
for t, spec in TISSUES.items():
    w = _read_weights(PAXDB_DIR / f"{spec['paxdb']}.txt")
    jw = sum(v for k, v in w.items() if JIANG_TAG in k)
    top = max(w.items(), key=lambda kv: kv[1]) if w else ("", float("nan"))
    r = dict(tissue=t, n_input_datasets=len(w),
             jiang_weight=jw,
             top_dataset=top[0],
             top_weight=top[1])
    if t in tri:
        g = grid[(grid["tissue"] == t)
                 & (grid["ihc_status"] == "present")
                 & grid["gtex_status"].isin(["measured", "measured_absent", "not_measured"])]
        y = (g["gtex_status"] == "measured").astype(int).values
        r.update(n_ihc_present=len(g),
                 auc_tissue_ppm=_auc(y, np.log10(g["paxdb_ppm"].astype(float).values)),
                 auc_global_ppm=_auc(y, np.log10(g["paxdb_ppm_global"].astype(float).values)),
                 cov_tissue_ppm=float(g["paxdb_ppm"].notna().mean()),
                 cov_global_ppm=float(g["paxdb_ppm_global"].notna().mean()))
        rows.append(r)

    wb = _read_weights(PAXDB_DIR / "hs_whole_body.txt")
    tab = pd.DataFrame(rows)
    tab.to_csv(QC_DIR / "paxdb_provenance.tsv", sep="\t", index=False)
    tri_tab = tab[tab["tissue"].isin(tri)].sort_values("jiang_weight", ascending=False)
    L = ["paxdb provenance: Jiang 2020 (GTEx-MS, the label) inside the abundance covariate", "",
         f"{'tissue':16s} {'Jiang w':>8s} {'inputs':>6s}  {'AUC tissue ppm':>14s} {'AUC global ppm':>14s}"
         f"{'cov tissue':>10s} {'cov global':>10s} top input"]
    for _, r in tri_tab.iterrows():
        L.append(f"{r['tissue']:16s} {r['jiang_weight']:8.3f} {int(r['n_input_datasets']):6d}"
                 f"{r['auc_tissue_ppm']:14.3f} {r['auc_global_ppm']:14.3f}"
                 f"{r['cov_tissue_ppm']:10.1%} {r['cov_global_ppm']:10.1%}"
                 f"{r['top_dataset']} ({r['top_weight']:.2f})")
    rt, pt = spearmanr(tri_tab["jiang_weight"], tri_tab["auc_tissue_ppm"])
    rg, pg = spearmanr(tri_tab["jiang_weight"], tri_tab["auc_global_ppm"])
    L.append("")
    L.append(f"Spearman(Jiang weight, abundance only AUC): tissue ppm r={rt:+.2f} p={pt:.3f} "
             f"global ppm r={rg:+.2f} p={pg:.3f}")
    jw_wb = sum(v for k, v in wb.items() if JIANG_TAG in k)
    pa_wb = sum(v for k, v in wb.items() if "PeptideAtlas" in k or "PA_" in k or "Build" in k or "peptideatla" in k)
    L.append(f"whole body dataset: {len(wb)} inputs, Jiang weight {jw_wb:.3f}, "
             f"PeptideAtlas-derived weight {pa_wb:.2f}")
    L.append("")
    L.append("side panels (no gtex column): " + ", ".join(
        (f"{r['tissue']} Jiang w {r['jiang_weight']:.3f}" if r["n_input_datasets"] else f"{r['tissue']}")
        for _, r in tab[~tab["tissue"].isin(tri)].iterrows()))
    text = "\n".join(L)
    (QC_DIR / "paxdb_provenance.txt").write_text(text)
    print(text)
