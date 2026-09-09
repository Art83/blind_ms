"""
Funcs for statistical modeling and ML
"""
import pandas as pd
from config import ABUNDANCE_SOURCE

# Features for logistic regression
CONT = ["log_abundance", "gravy", "log_mw", "log_tryptic", "n_tm",
        "frac_pos_celltypes", "reliability_ord"]

BIOPHYS = ["gravy", "log_mw", "log_tryptic", "n_tm", "has_signal"]

# Reliability
REL_ORD = {"Enhanced": 3, "Supported": 2, "Approved": 1, "Uncertain": 0}
EXTRA = ["reliability_ord"]


def _zscore(df, cols):
    z = df.copy()
    for c in cols:
        s = z[c].astype(float)
        sd = s.std(ddof=0)
        z[c] = (s - s.mean()) / sd if sd else 0.0
    return z


def _other_source():
    return "tissue" if ABUNDANCE_SOURCE == "global" else "global"


def _swap_abundance(df):
    col = f"log_abundance_{_other_source()}"
    d = df[df[col].notna()].copy()
    d["log_abundance"] = d[col]
    return d


def feature_contrasts(tbl):
    rows = []
    for fl, sub in tbl.groupby("type"):
        rows.append(dict(
            type=fl, n=len(sub),
            frac_TM_pos=round((sub["n_tm"] > 0).mean(), 3),
            frac_signal=round(sub["has_signal"].mean(), 3),
            med_gravy=round(sub["gravy"].median(), 3),
            med_log_tryptic=round(sub["log_tryptic"].median(), 3),
            med_log_abundance=round(sub["log_abundance"].median(), 3),
            med_frac_pos_ct=round(sub["frac_pos_celltypes"].median(), 3),
            med_reliability_ord=round(sub["reliability_ord"].median(), 3),
            frac_uncertain=round((sub["best_reliability"] == "Uncertain").mean(), 3),
        ))
    order = {"measured": 0, "measured_absent": 1, "not_measured": 2}
    return pd.DataFrame(rows).sort_values("type", key=lambda s: s.map(order))


def fit(df, label="pooled"):
    lines = [f"--- {label}  (n={len(df):,}, MS-bright frac {df['gtex_detected'].mean():.3f}) ---"]
    if df["gtex_detected"].nunique() < 2:
        return lines[0] + "  [single outcome class, skipped]"
    # drop zero-variance covariates so the design can't go singular
    cand = ["log_abundance", "frac_pos_celltypes"] + BIOPHYS + EXTRA
    full_terms = [t for t in cand if t in df.columns and df[t].nunique() > 1]
    z = _zscore(df, [c for c in CONT if c in df.columns])
    tis = pd.get_dummies(z["tissue"], prefix="t", drop_first=True).astype(float)
    base_X = pd.concat([z[["log_abundance"]], tis], axis=1)
    full_X = pd.concat([z[full_terms], tis], axis=1)
    y = df["gtex_detected"].values

    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    groups = df["ensg"].values
    b = sm.Logit(y, sm.add_constant(base_X)).fit(disp=0, cov_type="cluster",
                                                 cov_kwds={"groups": groups})
    f = sm.Logit(y, sm.add_constant(full_X)).fit(disp=0, cov_type="cluster",
                                                 cov_kwds={"groups": groups})
    lines.append(f"AUC baseline {roc_auc_score(y, b.predict(sm.add_constant(base_X))):.4f} "
                 f"-> full {roc_auc_score(y, f.predict(sm.add_constant(full_X))):.4f}")
    lines.append("coef (full, gene-clustered SE):")
    for name in full_terms:
        if name in f.params.index:
            lines.append(f"  {name:>20s}  beta={f.params[name]:+.3f}  p={f.pvalues[name]:.2e}")
    return "\n".join(lines)

