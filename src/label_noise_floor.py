"""
Label noise floor: how reproducible is the MS-detection label the models
are trained on?

"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import ABUNDANCE_SOURCE, TAB_DIR, GTEX_MEDIAN, TISSUES

TRI = [t for t, s in TISSUES.items() if s["gtex"]]
PAIRED = {t: s["gtex"] for t, s in TISSUES.items() if s["gtex"] and len(s["gtex"]) == 2}


def _auc(y, x):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); x = np.asarray(x, dtype=float)
    return float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 else float("nan")


def _kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def load():
    med = pd.read_csv(GTEX_MEDIAN, dtype=str)
    med = med.rename(columns={med.columns[0]: "ensg"})
    med["ensg"] = med["ensg"].str.strip()
    for c in med.columns[1:]:
        med[c] = pd.to_numeric(med[c], errors="coerce")
    grid = pd.read_csv(TAB_DIR / "grid.tsv", sep="\t", low_memory=False)
    return med, grid


def per_tissue(med, grid):
    L = ["=== (1) per-tissue label: paired GTEx subsites, IHC-present proteins in the GTEx matrix ==="]
    rows = []
    for t, (a, b) in PAIRED.items():
        ihc = grid[(grid["tissue"] == t) & (grid["ihc_status"] == "present")]
        d = ihc[["ensg", "paxdb_ppm", "paxdb_ppm_global"]].merge(med[["ensg", a, b]], on="ensg")
        da = d[a].notna().astype(int).values
        db = d[b].notna().astype(int).values
        both, neither = int((da & db).sum()), int(((1 - da) & (1 - db)).sum())
        only_a, only_b = int((da & (1 - db)).sum()), int(((1 - da) & db).sum())
        either = int((da | db).sum())
        # same abundance source as the models (config.ABUNDANCE_SOURCE), other as fallback
        if ABUNDANCE_SOURCE == "global":
            abund = np.log10(d["paxdb_ppm_global"].fillna(d["paxdb_ppm"]))
        else:
            abund = np.log10(d["paxdb_ppm"].fillna(d["paxdb_ppm_global"]))
        ok = abund.notna().values
        disagree = (da != db)
        q = pd.qcut(abund[ok], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
        by_q = pd.Series(disagree[ok]).groupby(q.values, observed=True).mean()
        rows.append(dict(tissue=t, subsite_a=a, subsite_b=b, n=len(d), both=both,
                         neither=neither, only_a=only_a, only_b=only_b,
                         agreement=(both + neither) / len(d), kappa=_kappa(da, db),
                         frac_detected_once=(only_a + only_b) / either,
                         auc_a_to_b=_auc(db, da), auc_b_to_a=_auc(da, db),
                         auc_abundance_to_a=_auc(da[ok], abund[ok]),
                         auc_abundance_to_b=_auc(db[ok], abund[ok]),
                         **{f"disagree_{k.split()[0]}": float(v) for k, v in by_q.items()}))
        r = rows[-1]
        L.append(f"[{t}] {a} vs {b}   n={r['n']:,}")
        L.append(f"both {both:,}  neither {neither:,}  {a} only {only_a:,}  {b} only {only_b:,}")
        L.append(f"agreement {r['agreement']:.3f}   kappa {r['kappa']:.3f}   "
                 f"detected in one subsite only: {r['frac_detected_once']:.1%} of proteins detected in either")
        L.append(f"AUC one subsite -> other: {r['auc_a_to_b']:.3f} / {r['auc_b_to_a']:.3f}   "
                 f"(binary replicate, cannot rank)")
        L.append(f"AUC PaxDb abundance -> subsite: {r['auc_abundance_to_a']:.3f} / "
                 f"{r['auc_abundance_to_b']:.3f}   (continuous comparator, n={int(ok.sum()):,})")
        L.append("disagreement by abundance quartile: " +
                 "  ".join(f"{k} {v:.1%}" for k, v in by_q.items()))
    L.append("")
    lo = min(r["frac_detected_once"] for r in rows); hi = max(r["frac_detected_once"] for r in rows)
    L.append(f"  reading: two MS samples of the same organ disagree on {lo:.0%}-{hi:.0%} of the proteins")
    return "\n".join(L), pd.DataFrame(rows)


def gene_level_subsite(med, grid):
    single = [s["gtex"][0] for t, s in TISSUES.items() if s["gtex"] and len(s["gtex"]) == 1]
    A = single + [v[0] for v in PAIRED.values()]
    B = single + [v[1] for v in PAIRED.values()]
    present = grid[(grid["ihc_status"] == "present") & grid["tissue"].isin(TRI)]["ensg"].unique()
    m = med[med["ensg"].isin(present)]
    ya = m[A].notna().any(axis=1).astype(int)
    yb = m[B].notna().any(axis=1).astype(int)
    L = ["=== (2a) gene-level label: 'ever detected' is insensitive to subsite choice ===",
         f"IHC-present-somewhere genes in the GTEx matrix: {len(m):,}",
         f"ever-detected rate using first subsites {ya.mean():.3f}, second subsites {yb.mean():.3f}, "
         f"genes that flip: {int((ya != yb).sum())} ({(ya != yb).mean():.2%})"]
    return "\n".join(L)


def gene_level_comparison():
    pa = pd.read_csv(TAB_DIR / "peptideatlas_per_gene.tsv", sep="\t")
    pa = (pa.sort_values("pa_n_peptides", ascending=False)
            .drop_duplicates("ensg")[["ensg", "pa_n_peptides"]])
    ml = pd.read_csv(TAB_DIR / "ml_per_gene_predictions.tsv", sep="\t")
    n_dup = int(ml["ensg"].duplicated().sum())
    ml = ml.drop_duplicates("ensg")
    d = ml.merge(pa, on="ensg", how="inner")
    y = d["y"].values
    thr = float(np.quantile(d["pa_n_peptides"], 1 - y.mean()))
    ypa = (d["pa_n_peptides"] > thr).astype(int).values
    agree = ypa == y
    L = ["=== (2b) gene-level label: GTEx vs PeptideAtlas at matched positive rate ==="]
    if n_dup:
        L.append(f"note: ml_per_gene_predictions.tsv had {n_dup} duplicate ENSG rows, first kept")
    L.append(f"genes with both labels: {len(d):,}   GTEx positive rate {y.mean():.3f}")
    L.append(f"PeptideAtlas pseudo-label: observed peptides > {thr:.0f} (same positive rate)")
    L.append(f"agreement {agree.mean():.3f}   kappa {_kappa(y, ypa):.3f}   "
             f"GTEx-dark & PA-bright {int(((y == 0) & (ypa == 1)).sum()):,}   "
             f"GTEx-bright & PA-dark {int(((y == 1) & (ypa == 0)).sum()):,}")
    L.append(f"AUC PeptideAtlas peptide count -> GTEx label: {_auc(y, d['pa_n_peptides']):.3f}")
    L.append(f"AUC PaxDb abundance -> GTEx label:            {_auc(y, d['log_abundance']):.3f}")
    L.append("")
    L.append("model performance, all genes vs the consensus subset (both sources agree):")
    for col, lab in [("p_detect_intrinsic", "intrinsic (no abundance)"), ("p_detect_full", "+abundance")]:
        L.append(f"{lab:26s} AUC vs GTEx {_auc(y, d[col]):.3f}   vs PeptideAtlas {_auc(ypa, d[col]):.3f}   "
                 f"on consensus (n={int(agree.sum()):,}) {_auc(y[agree], d[col][agree]):.3f}")
    return "\n".join(L)


med, grid = load()
t1, tab = per_tissue(med, grid)
t2 = gene_level_subsite(med, grid)
t3 = gene_level_comparison()
text = "LABEL NOISE FLOOR\n\n" + t1 + "\n\n" + t2 + "\n\n" + t3
(TAB_DIR / "label_noise_floor.txt").write_text(text)
tab.to_csv(TAB_DIR / "label_noise_subsites.tsv", sep="\t", index=False)
print(text)

