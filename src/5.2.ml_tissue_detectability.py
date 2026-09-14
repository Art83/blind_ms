"""
Per-tissue MS-detectability model.


This script fits the same feature groups on the gene x tissue table
(model_table.tsv, IHC-present rows only) with the tissue as a categorical
covariate, cross-validated with StratifiedGroupKFold grouped by gene so no
gene contributes to both training and test.
Three models as in ml_detectability.py: intrinsic (no abundance, no
translation block), practical (intrinsic + translation block) and +abundance
(log_abundance as built by model_msdark.py from config.ABUNDANCE_SOURCE,
whole-body PaxDb by default). Fixed GBM defaults, no tuning: the stage 11
nested grid changed nothing.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils import _merge_size_tryptic, present, _cols, _boot_ci, _diff_ci
from config import TAB_DIR, GROUPS, ABUNDANCE_SOURCE, TISSUES, EXCLUDED

N_SPLITS = 5
TISSUES = [t for t, s in TISSUES.items() if s["gtex"]]
ABUNDANCE_GROUP = {"abundance": ["log_abundance"]}

def build_table(out):
    mt = pd.read_csv(out / "model_table.tsv", sep="\t", low_memory=False)
    mt = mt[["ensg", "tissue", "gtex_detected", "type", "log_abundance"]].dropna(subset=["log_abundance"])
    feats = pd.read_csv(out / "features_protein.tsv", sep="\t", low_memory=False).drop_duplicates("ensg")
    pep_path = out / "features_peptides.tsv"
    if pep_path.exists():
        feats = feats.merge(pd.read_csv(pep_path, sep="\t"), on="uniprot", how="left")
    df = mt.merge(feats, on="ensg", how="left")
    for t in TISSUES:
        df[f"tissue_{t}"] = (df["tissue"] == t).astype(int)
    return df.reset_index(drop=True)


def oof(df, groups, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = _cols(groups) + [f"tissue_{t}" for t in TISSUES]
    X = df[cols].astype(float).values
    y = df["gtex_detected"].values
    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
    p = np.full(len(df), np.nan)
    for tr, te in cv.split(X, y, groups=df["ensg"].values):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                           random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


df = build_table(TAB_DIR)
y = df["gtex_detected"].values
intr = _merge_size_tryptic(present(GROUPS, df))
trans = [c for c in EXCLUDED if c in df.columns]
prac = _merge_size_tryptic(present({**GROUPS, "translation": trans}, df))
full = _merge_size_tryptic(present({**GROUPS, **ABUNDANCE_GROUP}, df))

p_i = oof(df, intr)
p_p = oof(df, prac)
p_f = oof(df, full)
df["p_tissue_intrinsic"] = p_i
df["p_tissue_practical"] = p_p
df["p_tissue_full"] = p_f

L = ["Per tissue detetcability score (gene x tissue, gene-grouped CV, bootstrap CIs)", "",
     f"rows: {len(df):,} IHC-present gene x tissue observations, {df['ensg'].nunique():,} genes, "
     f"{len(TISSUES)} tissues   detected rate {y.mean():.3f}", "",
     "=== overall (out-of-fold GBM, 95% bootstrap CI) ==="]
src = ABUNDANCE_SOURCE
for lab, p in [("intrinsic (no abundance)", p_i), ("practical (+translation block)", p_p),
               (f"+abundance ({src} PaxDb)", p_f)]:
    a = _boot_ci(y, p, roc_auc_score)
    L.append(f"  {lab:32s} AUC {a[0]:.3f}  [{a[1]:.3f}, {a[2]:.3f}]")
dd = _diff_ci(y, p_f, p_i, roc_auc_score)
L.append(f"AUC gain from abundance: {dd[0]:+.3f}  [{dd[1]:+.3f}, {dd[2]:+.3f}]")
dp = _diff_ci(y, p_p, p_i, roc_auc_score)
L.append(f"AUC gain from the translation block: {dp[0]:+.3f}  [{dp[1]:+.3f}, {dp[2]:+.3f}]")
L.append(f"abundance alone (no model): {_auc(y, df['log_abundance']):.3f}")
L.append("")

L.append("=== by flavour of MS absence ===")
for flav in ("not_measured", "measured_absent"):
    m = (df["type"] == "measured") | (df["type"] == flav)
    L.append(f"measured vs {flav:16s} n={int(m.sum()):7,}   intrinsic {_auc(y[m], p_i[m]):.3f}   "
             f"practical {_auc(y[m], p_p[m]):.3f}   "
             f"+abundance {_auc(y[m], p_f[m]):.3f}   abundance alone {_auc(y[m], df.loc[m, 'log_abundance']):.3f}")
L.append("measured_absent is the per-tissue case the gene-level label cannot see.")
L.append("")

floor_path = TAB_DIR / "label_noise_subsites.tsv"
floor = pd.read_csv(floor_path, sep="\t").set_index("tissue") if floor_path.exists() else None
L.append("=== per tissue ===")
L.append(f"{'tissue':16s} {'n':>7s}  {'det':>5s}  {'intrinsic':>9s}  {'practical':>9s}  {'+abund':>7s}  {'abund only':>10s}"
         + ("   subsite kappa    abund->subsite" if floor is not None else ""))
for t in TISSUES:
    m = (df["tissue"] == t).values
    line = (f"{t:16s} {int(m.sum()):7,}  {y[m].mean():.3f}  {_auc(y[m], p_i[m]):9.3f}  "
            f"{_auc(y[m], p_p[m]):9.3f}  "
            f"{_auc(y[m], p_f[m]):7.3f}  {_auc(y[m], df.loc[m, 'log_abundance']):10.3f}")
    if floor is not None and t in floor.index:
        r = floor.loc[t]
        line += f"   {r['kappa']:13.3f}  {np.mean([r['auc_abundance_to_a'], r['auc_abundance_to_b']]):15.3f}"
    L.append(line)
L.append("")

# gene-level aggregate for comparison with ml_detectability.py
g = df.groupby("ensg").agg(y=("gtex_detected", "max"), p_i=("p_tissue_intrinsic", "max"),
                           p_p=("p_tissue_practical", "max"), p_f=("p_tissue_full", "max"))
L.append("=== gene-level aggregate (max over tissues) vs the gene-level model ===")
L.append(f"genes {len(g):,}   intrinsic {_auc(g['y'], g['p_i']):.3f}   practical {_auc(g['y'], g['p_p']):.3f}   "
         f"+abundance {_auc(g['y'], g['p_f']):.3f}")
mlp = TAB_DIR / "ml_per_gene_predictions.tsv"
if mlp.exists():
    ml = pd.read_csv(mlp, sep="\t").drop_duplicates("ensg").set_index("ensg")
    cols = [c for c in ("p_detect_intrinsic", "p_detect_practical", "p_detect_full") if c in ml.columns]
    j = g.join(ml[cols], how="inner")
    L.append(f"ml_detectability.py on the same {len(j):,} genes: "
             + "   ".join(f"{c.replace('p_detect_', '')} {_auc(j['y'], j[c]):.3f}" for c in cols))
L.append("")

keep = ["ensg", "tissue", "gtex_detected", "type", "log_abundance",
        "p_tissue_intrinsic", "p_tissue_practical", "p_tissue_full"]
df[keep].to_csv(TAB_DIR / "tissue_detectability_predictions.tsv", sep="\t", index=False)
text = "\n".join(L)
(TAB_DIR / "tissue_detectability.txt").write_text(text)
print(text)
