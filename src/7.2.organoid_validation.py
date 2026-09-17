"""
Orthogonal validation of the MS-detectability model in iPSC cerebral organoids
measured on three platforms (DIA-NN MS, SomaScan 11k, Olink Explore HT) in two
compartments (organoid lysate, conditioned media), two cell lines (TDR1
control, TDR4 AD), five organoids each.

Consumes organoid_platforms.py output (LoD-based per-sample detection with
line carried). Genotype is confounded with line, so every test is reported
pooled and within each line, and the line split is a replication axis. No
disease claim is made anywhere in this script.

Tests this scripts runs:
  T0  concordance      cross-platform abundance agreement, per arm
  T0b reliability      per-feature SD, CV and ICC of affinity abundance across the
                       five organoids of a line: measurement error in the covariate

  T1  presence         structural score, MS-caught vs MS-missed among affinity-present

  T2  core             MS detection ~ structure + affinity abundance, per platform, per arm,
                       pooled and within line.

  T2b raw feature      same with tryptic_per_kda instead of the trained score

  T2c abundance strata score with affinity abundance as quintile dummies and within
                       each quintilet (residual abundance check)

  T3  specificity      the same predictors against SomaScan vs Olink disagreement.
                       If structure predicts MS loss and not affinity disagreement,
                       the score measures MS chemistry, not "difficult protein".

  T4  graded outcome   MS detection rate across replicates: never / intermittent /
                       always, among affinity-present, pooled (k of 10) and per line
                       (k of 5).

  T5  quantitative     among triple-detected proteins, MS intensity residual after
                       affinity abundance vs structure.

  T6  inference-dark   proteins with no unique tryptic peptide, or most peptides shared,
                       vs MS detection at matched affinity abundance.

  T7  lysate to media  paired: affinity-present in both compartments, MS keeps the
                       protein in media or loses it. Secretion route vs structure.

  T8  calibration      GTEx-trained probability vs observed organoid MS detection, by decile

  T9  missing proteome PE2-4 proteins affinity-present in organoids
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from config import TAB_DIR

PRESENT_FRAC = 0.70
LINES = ("TDR1", "TDR4")
ARMS = ("Organoid", "Media")
PLATS = ("MS", "SomaScan", "Olink")
SHORT = {"MS": "ms", "SomaScan": "soma", "Olink": "olink"}
PEP_COLS = ["pep_n_unique", "pep_frac_shared", "pep_max_shared_genes", "pep_n_clean"]
FEAT_COLS = ["uniprot", "tryptic_per_kda", "n_tryptic_7_30", "mol_weight_kDa", "n_tm",
             "is_secreted", "has_signal", "is_membrane", "disorder_fraction_proxy"]


def z(x):
    x = pd.to_numeric(x, errors="coerce")
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s and s > 0 else x * 0.0


def auc(y, x):
    y = np.asarray(y)
    m = np.isfinite(np.asarray(x, dtype=float))
    return float(roc_auc_score(y[m], np.asarray(x, dtype=float)[m])) if len(np.unique(y[m])) > 1 else float("nan")


def logit(d, y, preds, label, L, min_n=40):
    dd = d[[y] + preds].replace([np.inf, -np.inf], np.nan).dropna()
    if dd[y].nunique() < 2 or len(dd) < min_n or dd[y].sum() < 10 or (dd[y] == 0).sum() < 10:
        L.append(f"     {label}: too few to fit (n={len(dd)}, pos={int(dd[y].sum()) if len(dd) else 0})")
        return None
    X = sm.add_constant(pd.DataFrame({p: z(dd[p]) for p in preds}))
    try:
        m = sm.Logit(dd[y].values, X).fit(disp=0, maxiter=200)
        if not m.mle_retvals.get("converged", True):
            m = sm.Logit(dd[y].values, X).fit(disp=0, maxiter=500, method="bfgs")
    except Exception as e:
        L.append(f"     {label}: fit failed ({e.__class__.__name__})")
        return None
    bad = (not m.mle_retvals.get("converged", True)) or bool(np.isnan(m.pvalues[preds]).any())
    flag = "DID NOT CONVERGE or separated" if bad else ""
    parts = "  ".join(f"{p} b={m.params[p]:+.3f} p={m.pvalues[p]:.1e}" for p in preds)
    L.append(f" {label} (n={len(dd):,}, pos={int(dd[y].sum()):,}): {parts}{flag}")
    return m


def gene_table(out):
    feat = pd.read_csv(out / "organoid_features.tsv", sep="\t").dropna(subset=["ensg"])
    long = pd.read_csv(out / "organoid_long.tsv", sep="\t", dtype={"sample": str}).dropna(subset=["ensg"])
    best = (feat.sort_values(["det_rate", "mean_value"], ascending=False)
                .drop_duplicates(["platform", "arm", "ensg"]))
    line_val = (long.merge(best[["platform", "arm", "feature_id"]], on=["platform", "arm", "feature_id"])
                    .groupby(["platform", "arm", "ensg", "line"])["value"].mean().unstack("line"))
    line_val.columns = [f"val_{c}" for c in line_val.columns]
    best = best.merge(line_val.reset_index(), on=["platform", "arm", "ensg"], how="left")
    rows = []
    for arm in ARMS:
        wide = None
        for plat in PLATS:
            s = SHORT[plat]
            b = best[(best["platform"] == plat) & (best["arm"] == arm)]
            cols = {"det_rate": f"{s}_det", "mean_value": f"{s}_abund", "present": f"{s}_present",
                    "det_rate_TDR1": f"{s}_det_TDR1", "det_rate_TDR4": f"{s}_det_TDR4",
                    "present_TDR1": f"{s}_present_TDR1", "present_TDR4": f"{s}_present_TDR4",
                    "val_TDR1": f"{s}_abund_TDR1", "val_TDR4": f"{s}_abund_TDR4"}
            if plat == "MS":
                cols["n_proteotypic"] = "ms_n_proteotypic"
                cols["n_sequences"] = "ms_n_sequences"
            b = b[["ensg"] + list(cols)].rename(columns=cols)
            wide = b if wide is None else wide.merge(b, on="ensg", how="outer")
        wide["arm"] = arm
        rows.append(wide)
    g = pd.concat(rows, ignore_index=True)
    for s in SHORT.values():
        for suf in ("", "_TDR1", "_TDR4"):
            g[f"{s}_present{suf}"] = g[f"{s}_present{suf}"].fillna(0).astype(int)
            g[f"{s}_det{suf}"] = g[f"{s}_det{suf}"].fillna(0.0)
    return g


def annotate(g, out):
    pred = pd.read_csv(out / "ml_per_gene_predictions.tsv", sep="\t").drop_duplicates("ensg")
    keep = ["ensg", "p_detect_intrinsic", "p_detect_full", "top_dark_reason"]
    g = g.merge(pred[keep], on="ensg", how="left")
    fp = pd.read_csv(out / "features_protein.tsv", sep="\t", low_memory=False).drop_duplicates("ensg")
    g = g.merge(fp[["ensg"] + [c for c in FEAT_COLS if c in fp.columns]], on="ensg", how="left")
    pp = out / "features_peptides.tsv"
    if pp.exists():
        pep = pd.read_csv(pp, sep="\t")[["uniprot"] + PEP_COLS]
        g = g.merge(pep, on="uniprot", how="left")
    g["aff_present"] = ((g["soma_present"] == 1) | (g["olink_present"] == 1)).astype(int)
    g["aff_abund_z"] = pd.concat([z(g["soma_abund"]), z(g["olink_abund"])], axis=1).mean(axis=1)
    return g


# ---------------------------------------------------------------- tests
def t0_concordance(g, L):
    L.append("T0 concordance. Spearman of abundance across platforms (shared ENSG):")
    for arm in ARMS:
        d = g[g["arm"] == arm]
        parts = []
        for a, b in [("soma_abund", "olink_abund"), ("soma_abund", "ms_abund"), ("olink_abund", "ms_abund")]:
            sub = d[(d[a.split("_")[0] + "_present"] == 1) & (d[b.split("_")[0] + "_present"] == 1)][[a, b]].dropna()
            if len(sub) > 30:
                r, _ = spearmanr(sub[a], sub[b])
                parts.append(f"{a.split('_')[0]}~{b.split('_')[0]} rho={r:.2f} (n={len(sub):,})")
        L.append(f" {arm:9s} " + "   ".join(parts))
    L.append("")


def t_reliability(out, L):
    L.append("T0b covariate reliability -- affinity abundance across the five replicates per line:")
    long = pd.read_csv(out / "organoid_long.tsv", sep="\t", dtype={"sample": str})
    long = long[(long["platform"].isin(["SomaScan", "Olink"])) & (long["above_lod"] == 1)].dropna(subset=["value"])
    for (plat, arm), d in long.groupby(["platform", "arm"]):
        parts = []
        for ln, dl in d.groupby("line"):
            gstat = dl.groupby("feature_id")["value"].agg(["mean", "std", "count"])
            gstat = gstat[gstat["count"] >= 3]
            cv = (gstat["std"] / gstat["mean"].abs()).replace([np.inf, -np.inf], np.nan)
            k = gstat["count"].mean()
            msw = (gstat["std"] ** 2).mean()
            msb = gstat["mean"].var(ddof=1) * k
            icc = (msb - msw) / (msb + (k - 1) * msw) if msb + (k - 1) * msw > 0 else np.nan
            parts.append(f"{ln}: features {len(gstat):,}, median within-line SD {gstat['std'].median():.2f} log2, "
                         f"median CV {cv.median():.3f}, ICC(1) {icc:.3f}")
        L.append(f"{plat:9s} {arm:9s} " + " | ".join(parts))
    L.append(" between-line agreement of the line means (Spearman across features present in both lines):")
    for (plat, arm), d in long.groupby(["platform", "arm"]):
        m = d.groupby(["feature_id", "line"])["value"].mean().unstack("line").dropna()
        if len(m) > 30 and set(LINES) <= set(m.columns):
            r, _ = spearmanr(m[LINES[0]], m[LINES[1]])
            L.append(f"   {plat:9s} {arm:9s} n={len(m):,}  rho={r:.3f}")
    L.append("")


def t1_presence(g, L):
    L.append("T1 presence. structural score (abundance-free), MS-caught vs MS-missed among affinity-present:")
    for arm in ARMS:
        for ref, lab in (("olink", "Olink-present"), ("soma", "SomaScan-present")):
            d = g[(g["arm"] == arm) & (g[f"{ref}_present"] == 1)].dropna(subset=["p_detect_intrinsic"])
            if d["ms_present"].nunique() < 2 or (d["ms_present"] == 0).sum() < 10:
                continue
            a_pool = auc(d["ms_present"], d["p_detect_intrinsic"])
            line_aucs = []
            for ln in LINES:
                dl = d[d[f"{ref}_present_{ln}"] == 1]
                line_aucs.append(f"{ln} {auc(dl[f'ms_present_{ln}'], dl['p_detect_intrinsic']):.3f}")
            L.append(f"{arm:9s} among {lab:16s} n={len(d):,}  MS-missed {int((d['ms_present'] == 0).sum()):,}  "
                     f"AUC {a_pool:.3f}   within line: " + ", ".join(line_aucs))
    L.append("")


def t2_core(g, L, pred_col="p_detect_intrinsic", head="T2 CORE", note=True):
    L.append(f"{head}MS detection ~ {pred_col} + affinity abundance, per platform:")
    for arm in ARMS:
        L.append(f"   [{arm}]")
        d = g[g["arm"] == arm].copy()
        d["z_struct"] = d[pred_col]
        for ref, lab in (("soma", "SomaScan"), ("olink", "Olink")):
            sub = d[d[f"{ref}_present"] == 1].copy()
            sub["z_ab"] = sub[f"{ref}_abund"]
            logit(sub, "ms_present", ["z_struct", "z_ab"], f"{lab:9s} pooled ", L)
            for ln in LINES:
                sl = d[d[f"{ref}_present_{ln}"] == 1].copy()
                sl["z_ab"] = sl[f"{ref}_abund_{ln}"]
                logit(sl, f"ms_present_{ln}", ["z_struct", "z_ab"], f"{lab:9s} {ln}   ", L)
        both = d[(d["soma_present"] == 1) & (d["olink_present"] == 1)].copy()
        both["z_soma"] = both["soma_abund"]
        both["z_olink"] = both["olink_abund"]
        logit(both, "ms_present", ["z_struct", "z_soma", "z_olink"], "both      pooled ", L)
    if note:
        L.append(" -> z_struct positive and significant with a non-MS abundance term in the model,")
        L.append("in both compartments and both lines, is the chemistry claim.")
    L.append("")


def t2_abundance_strata(g, L, pred_col="p_detect_intrinsic"):
    L.append("T2c abundance strata score with affinity abundance as quintile dummies, and within each quintile:")
    for arm in ARMS:
        d = g[g["arm"] == arm].copy()
        d["z_struct"] = d[pred_col]
        for ref, lab in (("soma", "SomaScan"), ("olink", "Olink")):
            sub = d[d[f"{ref}_present"] == 1].dropna(subset=[f"{ref}_abund", pred_col]).copy()
            if len(sub) < 200:
                continue
            sub["q"] = pd.qcut(sub[f"{ref}_abund"], 5, labels=False, duplicates="drop")
            dum = pd.get_dummies(sub["q"], prefix="q", drop_first=True).astype(float)
            X = sm.add_constant(pd.concat([z(sub["z_struct"]).rename("z_struct"), dum], axis=1))
            try:
                m = sm.Logit(sub["ms_present"].values, X).fit(disp=0, maxiter=200)
                L.append(f"{arm:9s} {lab:9s} quintile dummies (n={len(sub):,}): z_struct b={m.params['z_struct']:+.3f} p={m.pvalues['z_struct']:.1e}")
            except Exception as e:
                L.append(f"{arm:9s} {lab:9s} quintile dummies: fit failed ({e.__class__.__name__})")
            parts = []
            for qq, s in sub.groupby("q"):
                s = s.copy()
                s["z_ab"] = s[f"{ref}_abund"]
                mm = logit(s, "ms_present", ["z_struct", "z_ab"], f"{lab} Q{int(qq)+1}", [], min_n=40)
                parts.append(f"Q{int(qq)+1} {mm.params['z_struct']:+.2f}" + ("*" if mm.pvalues['z_struct'] < 0.05 else "") if mm is not None else f"Q{int(qq)+1} n/a")
            L.append(f"   {arm:9s} {lab:9s} within quintile (low to high): " + ", ".join(parts) + "   (* p<0.05)")
    L.append("")


def t3_specificity(g, L):
    L.append("T3 platform specificity, same predictors against SomaScan vs Olink disagreement:")
    L.append("outcome = present on exactly one affinity platform, among proteins on both panels.")
    L.append(" a score that predicts MS loss but not affinity disagreement measures MS chemistry.")
    for arm in ARMS:
        d = g[g["arm"] == arm].copy()
        panel = d.dropna(subset=["soma_abund", "olink_abund"]).copy()
        panel["aff_disagree"] = (panel["soma_present"] != panel["olink_present"]).astype(int)
        panel["z_struct"] = panel["p_detect_intrinsic"]
        panel["z_ab"] = panel["aff_abund_z"]
        L.append(f"{arm}] n on both panels {len(panel):,}, disagree {int(panel['aff_disagree'].sum()):,}")
        logit(panel, "aff_disagree", ["z_struct", "z_ab"], "affinity disagreement", L)
        aff = panel[panel["aff_present"] == 1].copy()
        logit(aff, "ms_present", ["z_struct", "z_ab"], "MS loss (same rows)  ", L)
        # continuous version: platform gap magnitude vs structure
        panel["gap"] = (z(panel["soma_abund"]) - z(panel["olink_abund"])).abs()
        r, p = spearmanr(panel["gap"], panel["p_detect_intrinsic"], nan_policy="omit")
        L.append(f"|z_soma - z_olink| vs structural score: Spearman r={r:+.3f} p={p:.1e}")
    L.append("")


def t4_graded(g, L):
    L.append("T4 graded outcome, MS replicate detection rate among affinity present proteins:")
    L.append("   pooled: never = 0 of 10, intermittent = 1-9 of 10, always = 10 of 10; per line: k of 5")
    scopes = (("pooled", "ms_det", ("soma_present", "olink_present")),
              ("TDR1", "ms_det_TDR1", ("soma_present_TDR1", "olink_present_TDR1")),
              ("TDR4", "ms_det_TDR4", ("soma_present_TDR4", "olink_present_TDR4")))
    for arm in ARMS:
        for scope, det_col, aff_cols in scopes:
            d = g[(g["arm"] == arm) & ((g[aff_cols[0]] == 1) | (g[aff_cols[1]] == 1))].dropna(subset=["p_detect_intrinsic"]).copy()
            d["cls"] = np.where(d[det_col] == 0, "never", np.where(d[det_col] >= 0.999, "always", "intermittent"))
            L.append(f"   [{arm} {scope}]")
            for c in ("never", "intermittent", "always"):
                s = d[d["cls"] == c]
                L.append(f"     {c:12s} n={len(s):5,}  median score {s['p_detect_intrinsic'].median():.3f}  "
                         f"median affinity abundance z {s['aff_abund_z'].median():+.2f}  median tryptic/kDa {s['tryptic_per_kda'].median():.3f}")
            sub = d[d["cls"] != "always"].copy()
            sub["never"] = (sub["cls"] == "never").astype(int)
            sub["z_struct"] = sub["p_detect_intrinsic"]
            sub["z_ab"] = sub["aff_abund_z"]
            logit(sub, "never", ["z_struct", "z_ab"], "never vs intermittent", L)
            hi = d[d["aff_abund_z"] >= d["aff_abund_z"].quantile(0.66)]
            L.append(f"top-tertile affinity abundance: never {int((hi['cls'] == 'never').sum()):,}, "
                     f"intermittent {int((hi['cls'] == 'intermittent').sum()):,}, always {int((hi['cls'] == 'always').sum()):,}")
    L.append("outcome is never=1, so a negative z_struct means higher structural score")
    L.append("")


def t5_quant(g, L):
    L.append("T5 quantitative deficit, among MS-detected, MS intensity residual after affinity abundance:")
    for arm in ARMS:
        d = g[(g["arm"] == arm) & (g["ms_present"] == 1)]
        for ref, lab in (("olink", "Olink"), ("soma", "SomaScan")):
            s = d[d[f"{ref}_present"] == 1].dropna(subset=["ms_abund", f"{ref}_abund", "p_detect_intrinsic"])
            if len(s) < 50:
                continue
            X = sm.add_constant(s[f"{ref}_abund"].values)
            res = s["ms_abund"].values - sm.OLS(s["ms_abund"].values, X).fit().predict(X)
            r1, p1 = spearmanr(res, s["p_detect_intrinsic"])
            r2, p2 = spearmanr(res, s["tryptic_per_kda"], nan_policy="omit")
            r3, p3 = spearmanr(res, s["pep_n_unique"], nan_policy="omit") if "pep_n_unique" in s else (np.nan, np.nan)
            L.append(f"{arm:9s} vs {lab:9s} n={len(s):,}  residual~score r={r1:+.3f} p={p1:.1e}   "
                     f"~tryptic/kDa r={r2:+.3f} p={p2:.1e}   ~unique peptides r={r3:+.3f} p={p3:.1e}")
    L.append("")


def t6_inference(g, L, out):
    L.append("T6 inference-dark -- no unique tryptic peptide, vs MS detection at matched affinity abundance:")
    exp = []
    for arm in ARMS:
        d = g[(g["arm"] == arm) & (g["aff_present"] == 1)].dropna(subset=["pep_n_unique"]).copy()
        d["zero_unique"] = (d["pep_n_unique"] == 0).astype(int)
        d["mostly_shared"] = (d["pep_frac_shared"] >= 0.5).astype(int)
        d["z_ab"] = d["aff_abund_z"]
        base = d["ms_present"].mean()
        for flag in ("zero_unique", "mostly_shared"):
            s = d[d[flag] == 1]
            L.append(f"   {arm:9s} {flag:14s} n={len(s):4,}  MS-present {s['ms_present'].mean():.1%} "
                     f"(all affinity-present {base:.1%})  median affinity abundance z {s['aff_abund_z'].median():+.2f}")
            logit(d, "ms_present", [flag, "z_ab"], f"{flag:14s} + abundance", L)
        det = d[d["ms_present"] == 1].dropna(subset=["ms_n_proteotypic"])
        r, p = spearmanr(det["ms_n_proteotypic"], det["pep_n_unique"])
        L.append(f"{arm:9s} DIA-NN N.Proteotypic.Sequences vs in-silico unique peptides (MS-detected, n={len(det):,}): "
                 f"Spearman r={r:+.3f} p={p:.1e}   groups with 0 proteotypic sequences {int((det['ms_n_proteotypic'] == 0).sum()):,}")
        exp.append(d[(d["zero_unique"] == 1) | (d["mostly_shared"] == 1)].assign(arm=arm))
    pd.concat(exp).sort_values("aff_abund_z", ascending=False).to_csv(out / "organoid_inference_dark.csv", index=False)
    L.append("")


def t7_paired(g, L):
    L.append("T7 lysate to media (paired), affinity-present in both compartments and MS-present in lysate:")
    L.append("outcome = MS still present in media. Secretion route vs structure vs abundance change.")
    o = g[g["arm"] == "Organoid"].set_index("ensg")
    m = g[g["arm"] == "Media"].set_index("ensg")
    common = o.index.intersection(m.index)
    for ref, lab in (("olink", "Olink"), ("soma", "SomaScan")):
        idx = [e for e in common if o.at[e, f"{ref}_present"] == 1 and m.at[e, f"{ref}_present"] == 1 and o.at[e, "ms_present"] == 1]
        d = pd.DataFrame({
            "ms_media": m.loc[idx, "ms_present"].values,
            "z_struct": o.loc[idx, "p_detect_intrinsic"].values,
            "d_ab": (m.loc[idx, f"{ref}_abund"].values - o.loc[idx, f"{ref}_abund"].values),
            "secreted": o.loc[idx, "is_secreted"].fillna(0).values,
            "signal": o.loc[idx, "has_signal"].fillna(0).values,
        })
        L.append(f"   {lab:9s} n={len(d):,}  MS kept in media {int(d['ms_media'].sum()):,} ({d['ms_media'].mean():.1%})  "
                 f"secreted among kept {d.loc[d['ms_media'] == 1, 'secreted'].mean():.1%} vs lost {d.loc[d['ms_media'] == 0, 'secreted'].mean():.1%}")
        logit(d, "ms_media", ["z_struct", "d_ab", "secreted"], "kept ~ structure + affinity change + secreted", L)
        logit(d, "ms_media", ["z_struct", "d_ab", "signal"], "kept ~ structure + affinity change + signal  ", L)
    L.append("")


def t8_calibration(g, L):
    L.append("T8 calibration. GTEx-trained P(detect) vs observed organoid MS detection, by decile (affinity-present):")
    for arm in ARMS:
        d = g[(g["arm"] == arm) & (g["aff_present"] == 1)].dropna(subset=["p_detect_intrinsic"]).copy()
        d["dec"] = pd.qcut(d["p_detect_intrinsic"], 10, labels=False, duplicates="drop")
        t = d.groupby("dec").agg(pred=("p_detect_intrinsic", "mean"), obs=("ms_present", "mean"), n=("ensg", "size"))
        slope = np.polyfit(t["pred"], t["obs"], 1)[0]
        L.append(f"{arm:9s} n={len(d):,}  " + "  ".join(f"{r.pred:.2f}->{r.obs:.2f}" for r in t.itertuples())
                 + f" slope {slope:.2f}")
    L.append("")


def t9_missing(g, L, out):
    p = out / "proteome_census.tsv"
    pe = pd.read_csv(p, sep="\t", usecols=["uniprot", "pe", "p_seq"])
    d = g.merge(pe, on="uniprot", how="inner")
    cand = d[(d["pe"].isin([2, 3, 4])) & (d["aff_present"] == 1)]
    L.append("T9 missing proteome. PE2-4 proteins above affinity LoD in organoids (candidates, cross-reactivity not excluded):")
    for arm in ARMS:
        c = cand[cand["arm"] == arm]
        L.append(f"   {arm:9s} PE2-4 on panels {int((d[d['arm'] == arm]['pe'].isin([2, 3, 4])).sum()):,}, "
                 f"affinity-present {len(c):,}, on both affinity platforms {int(((c['soma_present'] == 1) & (c['olink_present'] == 1)).sum()):,}, "
                 f"MS-detected {int(c['ms_present'].sum()):,}, median sequence only P(detect) {c['p_seq'].median():.2f}")
    cols = ["arm", "ensg", "uniprot", "pe", "p_seq", "soma_present", "olink_present", "ms_present", "aff_abund_z", "pep_n_unique"]
    cand[[c for c in cols if c in cand.columns]].sort_values("aff_abund_z", ascending=False).to_csv(
        out / "organoid_missing_proteome_candidates.csv", index=False)
    L.append("")


g = annotate(gene_table(TAB_DIR), TAB_DIR)
g.to_csv(TAB_DIR / "organoid_gene_table.tsv", sep="\t", index=False)

L = ["ogranoid detectability (three platforms, two compartments, two lines)", "",
     "design: 5 TDR1 (control) + 5 TDR4 (AD) organoids per compartment on every platform.",
     "genotype is confounded with line, so line is a replication tool and no disease claim is made.",
     f"presence = above LoD in >= {PRESENT_FRAC:.0%} of samples (organoid_platforms.py).", ""]
for arm in ARMS:
    d = g[g["arm"] == arm]
    ms, so, ol = (set(d.loc[d[f"{s}_present"] == 1, "ensg"]) for s in ("ms", "soma", "olink"))
    L.append(f"[{arm}] present ENSG: MS {len(ms):,}  SomaScan {len(so):,}  Olink {len(ol):,}   "
             f"MS&Soma&Olink {len(ms & so & ol):,}  Soma&Olink {len(so & ol):,}  MS-only {len(ms - so - ol):,}  "
             f"affinity-only {len((so | ol) - ms):,}")
L.append("")
t0_concordance(g, L)
t_reliability(TAB_DIR, L)
t1_presence(g, L)
t2_core(g, L)
t2_core(g, L, pred_col="tryptic_per_kda", head="T2b raw feature", note=False)
t2_abundance_strata(g, L)
t3_specificity(g, L)
t4_graded(g, L)
t5_quant(g, L)
t6_inference(g, L, TAB_DIR)
t7_paired(g, L)
t8_calibration(g, L)
t9_missing(g, L, TAB_DIR)
text = "\n".join(L)
(TAB_DIR / "organoid_validation.txt").write_text(text)
print(text)
