"""
MS-detectability ML model

Gene level binary prediction: given a protein is present (IHC-positive in >=1
tri-source tissue), does GTEx shotgun MS ever detect it? y=0 = present but MS-dark.
One row per gene and split with StratifiedGroupKFold grouped by gene.

Two main models: intrinsic (sequence/structure/localisation/complex, no abundance)
and +abundance.

Feature selection: nested CV, with stability selection run inside each outer training fold
All imputation/scaling is fit inside the fold. Correlated near-duplicate
features are grouped; size+tryptic are merged for selection/importance because
they are of one physical nature (more residues -> more peptides).

Every metric has a bootstrap 95% CI. The intrinsic vs +abundance gap has
a paired-bootstrap CI so "abundance helps" is a tested claim.
"""

from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from config import GROUPS, EXCLUDED, OBSERVED_PTM, HALF_LIFE_LEAK, TAB_DIR
from utils import _build_table, _merge_size_tryptic, nested_oof, _boot_ci, _diff_ci, calibration, perm_importance, \
    actionable_readout, transfer, present, per_gene_attribution
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PEPTIDE_GROUPS = ("proteotypicity", "peptide_chemistry")
PTM_GROUPS = ("ptm",)
ABUNDANCE_GROUP = {"abundance": ["log_abundance"]}
SEL_THRESH = 0.8
N_BOOT_SEL = 50
N_BOOT_CI = 1000
N_PERM = 20
N_SPLITS = 5

# Fine tuning.
TUNE = True

GRID = [(d, lr) for d in (3, 4, 6) for lr in (0.03, 0.05, 0.1)]
GBM_DEFAULT = (4, 0.05)
N_INNER = 3

df = _build_table(TAB_DIR).dropna(subset=["y", "log_abundance"]).reset_index(drop=True)
y = df["y"].values

intr = _merge_size_tryptic(present(GROUPS, df))
full = _merge_size_tryptic(present({**GROUPS, **ABUNDANCE_GROUP}, df))
trans = [c for c in EXCLUDED if c in df.columns]
prac = _merge_size_tryptic(present({**GROUPS, "translation": trans}, df)) if trans else None

L = ["MS-DETECTABILITY PREDICTOR (nested CV, gene-grouped, bootstrap CIs)", "",
     f"genes: {len(df):,}   positive rate (detected somewhere): {y.mean():.3f}   "
     f"MS-dark n={int((y == 0).sum()):,}",
     f"excluded a priori (translation regulation): {len(EXCLUDED)} features",
     f"selection: stability >= {SEL_THRESH} inside each outer fold; "
     f"reported metrics are the selected model's", ""]

oof_i_gbm, oof_i_log, sel_i, hp_i = nested_oof(df, intr, N_SPLITS, GBM_DEFAULT, GRID, N_INNER, tune=TUNE)
oof_f_gbm, oof_f_log, sel_f, hp_f = nested_oof(df, full, N_SPLITS, GBM_DEFAULT, GRID, N_INNER, tune=TUNE)
if prac is not None:
    oof_p_gbm, oof_p_log, sel_p, hp_p = nested_oof(df, prac, N_SPLITS, GBM_DEFAULT, GRID, N_INNER, tune=TUNE)

ablations = []
for name, drop in (("peptide-level groups (proteotypicity, peptide_chemistry)", PEPTIDE_GROUPS),
                   ("annotated PTM group", PTM_GROUPS),
                   ("topology group", ("topology",)),
                   ("AlphaFold pLDDT group", ("structure_af",))):
    if any(k in intr for k in drop):
        ablations.append((f"without {name}", "drop",
                          {k: v for k, v in intr.items() if k not in drop},
                          {k: v for k, v in full.items() if k not in drop}))

fine = [c for c in ("is_ecm", "is_chromatin", "is_lysosome", "is_peroxisome", "is_endosome",
                    "is_vesicle", "is_lipid_droplet") if c in df.columns]

if fine and "localization" in intr:
    coarse = [c for c in intr["localization"] if c not in fine]
    ablations.append((
        "without the extra compartments",
        "drop",
        {**intr, "localization": coarse}, {**full, "localization": coarse}))

if "structure_af" in intr and "disorder_fraction_proxy" in intr.get("composition", []):
    comp0 = [c for c in intr["composition"] if c != "disorder_fraction_proxy"]
    ablations.append((
        "without the sequence disorder proxy (pLDDT retained)",
        "drop",
        {**intr, "composition": comp0}, {**full, "composition": comp0}))

obs = [c for c in OBSERVED_PTM if c in df.columns]
if obs:
    ablations.append(("with OBSERVED-PTM counts added (MS observation history, a leak by construction)", "add",
                      {**intr, "ptm_observed": obs}, {**full, "ptm_observed": obs}))
hl = [c for c in HALF_LIFE_LEAK if c in df.columns]
if hl:
    ablations.append((
        "with Mathieson half-life added",
        "add",
        {**intr, "half_life": hl}, {**full, "half_life": hl}))
abl_oof = [(lab, kind, nested_oof(df, gi, N_SPLITS, GBM_DEFAULT, GRID, N_INNER,)[0], nested_oof(df, gf, N_SPLITS, GBM_DEFAULT, GRID, N_INNER,)[0]) for lab, kind, gi, gf in
           ablations]

L.append("=== performance (nested out-of-fold, GBM 95% bootstrap CI) ===")
models = [("intrinsic (no abundance, no translation block)", oof_i_gbm)]
if prac is not None:
    models.append(("practical (intrinsic + translation block; all inputs from the gene)", oof_p_gbm))
models.append(("+abundance", oof_f_gbm))

for lab, oof in models:
    auc = _boot_ci(y, oof, roc_auc_score)
    ap = _boot_ci(y, oof, average_precision_score)
    br = _boot_ci(y, oof, brier_score_loss)
    L.append(f" {lab} ")
    L.append(f"AUC {auc[0]:.3f}  [{auc[1]:.3f}, {auc[2]:.3f}]")
    L.append(f"PR-AUC {ap[0]:.3f}  [{ap[1]:.3f}, {ap[2]:.3f}]")
    L.append(f"Brier {br[0]:.3f}  [{br[1]:.3f}, {br[2]:.3f}]")
dd = _diff_ci(y, oof_f_gbm, oof_i_gbm, roc_auc_score)
sig = "abundance adds signal" if dd[1] > 0 else "abundance gain not significant"
L.append(f"AUC gain from abundance: {dd[0]:+.3f}  [{dd[1]:+.3f}, {dd[2]:+.3f}]  -> {sig}")
if prac is not None:
    dp = _diff_ci(y, oof_p_gbm, oof_i_gbm, roc_auc_score)
    sigp = "translation block adds signal" if dp[1] > 0 else "translation block gain not significant"
    L.append(f"AUC gain from the translation block: "
             f"{dp[0]:+.3f}  [{dp[1]:+.3f}, {dp[2]:+.3f}]  -> {sigp}")
L.append("")
L.append("logistic baseline AUC: "
         f"intrinsic {roc_auc_score(y, oof_i_log):.3f} | "
         + (f"practical {roc_auc_score(y, oof_p_log):.3f} | " if prac is not None else "")
         + f"+abundance {roc_auc_score(y, oof_f_log):.3f}")
L.append("")
L.append("hyperparameters chosen per outer fold (depth, lr) by inner %d-fold grouped CV over %d grid points; "
         "ablations use the fixed default %s:" % (N_INNER, len(GRID), GBM_DEFAULT))
hp_rows = [("intrinsic", hp_i)] + ([("practical", hp_p)] if prac is not None else []) + [("+abundance", hp_f)]
for lab, hp in hp_rows:
    L.append(f"  [{lab}] " + ", ".join(f"({d},{lr})" for d, lr in hp))
L.append("")

for lab, kind, oi, of in abl_oof:
    L.append(f"=== ablation: {lab} ===")
    for m, main, alt in [("intrinsic", oof_i_gbm, oi), ("+abundance", oof_f_gbm, of)]:
        a = _boot_ci(y, alt, roc_auc_score)
        dd = _diff_ci(y, main, alt, roc_auc_score) if kind == "drop" else _diff_ci(y, alt, main, roc_auc_score)
        sig = "ADDS signal" if dd[1] > 0 else "gain NOT significant"
        L.append(f"  [{m}] comparator AUC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]   "
                 f"gain from the group {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]  -> {sig}")
    L.append("")
if hl:
    cov = df[df["hl_log2_hours"].notna() & df["log_abundance"].notna()]
    L.append("=== half-life, covered subset only (value, not coverage) ===")
    L.append(f"genes with a Mathieson value: {len(cov):,} of {len(df):,} ({len(cov) / len(df):.1%}), "
             f"MS-dark among them {(cov['y'] == 0).mean():.1%} vs {(df['y'] == 0).mean():.1%} overall")
    try:
        import statsmodels.api as sm

        Z = pd.DataFrame({"hl": (cov["hl_log2_hours"] - cov["hl_log2_hours"].mean()) / cov["hl_log2_hours"].std(),
                          "ab": (cov["log_abundance"] - cov["log_abundance"].mean()) / cov["log_abundance"].std()})
        fit = sm.Logit(cov["y"].values, sm.add_constant(Z)).fit(disp=0)
        L.append(
            f"logit(detected) ~ half-life + abundance: half-life b={fit.params['hl']:+.3f} p={fit.pvalues['hl']:.1e}, "
            f"abundance b={fit.params['ab']:+.3f} p={fit.pvalues['ab']:.1e}")
        L.append("reading: the coverage of a pulse-SILAC dataset is MS detectability.")
    except Exception as e:
        L.append(f"covered-subset fit failed ({e.__class__.__name__})")
    L.append("")
L.append("calibration (+abundance GBM, quantile bins mean_pred->obs):")
L.append("  " + "  ".join(f"{mp:.2f}->{fp:.2f}" for mp, fp in calibration(y, oof_f_gbm)))
L.append("")

L.append("=== selection frequency across outer folds (kept if >= %.2f inner) ===" % SEL_THRESH)
sel_rows = [("intrinsic", sel_i)] + ([("practical", sel_p)] if prac is not None else []) + [("+abundance", sel_f)]
for lab, sf in sel_rows:
    L.append(f"  [{lab}] " + ", ".join(f"{g}:{f:.1f}" for g, f
                                       in sorted(sf.items(), key=lambda kv: -kv[1])))
L.append("")

imp_rows = [("intrinsic", intr)] + ([("practical", prac)] if prac is not None else []) + [("+abundance", full)]
for lab, groups in imp_rows:
    base, imp = perm_importance(df, groups)
    L.append(f"=== grouped permutation importance [{lab}] (GBM base AUC {base:.3f}; mean +/- sd over {N_PERM}) ===")
    for g, (mu, sd) in imp.items():
        flag = "" if mu > 2 * sd else "  ~0 (indistinguishable from noise)"
        L.append(f"  {g:18s} {mu:+.4f} +/- {sd:.4f}{flag}")
    L.append("")

L.append("=== external transfer (on abundance-free intrinsic predictions) ===")
L.append(transfer(TAB_DIR, df, oof_i_gbm))
L.append("")

L.append("=== actionable: structural risk bands + triage ===")
L.append(actionable_readout(df, oof_i_gbm))
L.append("")

attr = per_gene_attribution(df, present({**GROUPS, **ABUNDANCE_GROUP}, df))
pred = df[["ensg", "y", "log_abundance"]].copy()
pred["p_detect_intrinsic"] = oof_i_gbm
if prac is not None:
    pred["p_detect_practical"] = oof_p_gbm
pred["p_detect_full"] = oof_f_gbm
pred["tryptic_quartile"] = pd.qcut(df["n_tryptic_7_30"], 4,
                                   labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
pred = pred.merge(attr, on="ensg", how="left")
pred.to_csv(TAB_DIR / "ml_per_gene_predictions.tsv", sep="\t", index=False)
L.append("dominant 'why MS-dark' reason (per-gene logistic attribution):")
L.append(attr["top_dark_reason"].value_counts().to_string())
if "length" in df.columns:
    d2 = df.merge(attr[["ensg", "top_dark_reason"]], on="ensg", how="left")
    L.append("")
    L.append(f"size-direction check: median length all {d2['length'].median():.0f} aa; "
             f"size-flagged-dark {d2.loc[d2['top_dark_reason'] == 'size', 'length'].median():.0f} aa ")

text = "\n".join(L)
(TAB_DIR / "ml_summary.txt").write_text(text)
print(text)
