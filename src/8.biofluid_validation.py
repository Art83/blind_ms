"""
Validation of the MS-detectability model in human CSF and
plasma. Data: Dammer et al. 2022 (Emory), 36 CSF and 36 plasma specimens each
measured by TMT-MS (neat: 3 batches. depleted of top-14 abundant proteins: 5
batches) SomaScan (7,596 aptamers, raw RFU) and Olink Target 96 (13 panels, 1,196
assays, NPX).


Tests (Match T1-T9 in organoid_validation.py)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_DIR, TAB_DIR
from utils_validation import PEP_COLS, FEAT_COLS, z, auc, build_resolver, logit

EXT = DATA_DIR / "external_validation"
AFFINITY = {
    "CSF": {"olink": EXT / "2e1b.Olink.NPX.CSFneat.36samp.csv", "soma": EXT / "2g0b.Soma.rawRFU.CSFneat.36samp.csv"},
    "Plasma": {"olink": EXT / "2f1b.Olink.NPX.PlasmaNeat.36samp.csv", "soma": EXT / "2h0b.Soma.rawRFU.plasmaNeat.36samp.csv"},
}
MS_ARMS = {
    ("CSF", "neat"): (EXT / "2a2.BatchCorr.relAbundance.CSFneat.csv",      "batch.channel.1"),
    ("CSF", "depleted"): (EXT / "2c2.BatchCorr.relAbundance.CSFdepleted.csv",  "batch.channel"),
    ("Plasma", "neat"): (EXT / "2b2.BatchCorr.relAbundance.PlasmaNeat.csv",   "batch.channel.3"),
    ("Plasma", "depleted"): (EXT / "2d2.BatchCorr.relAbundance.PlasmaDepleted.csv", "batch.channel.2"),
}
SETS = [f"{f} {a}" for f, a in MS_ARMS]   # "CSF neat", "CSF depleted", "Plasma neat", "Plasma depleted"
MASTER = EXT / "SuppTable1-MasterTraits-Candidate-05032022_v01.xlsx"
FLUIDS = ("CSF", "Plasma")
ARMS = ("neat", "depleted")
GIS_CHANNEL = "127N"
CONC_P = 0.05


def load_master():
    d = pd.read_excel(MASTER, header=1)
    d = d[d["GUID"].notna()].copy()
    d["GUID"] = d["GUID"].astype(int).astype(str)
    d["dx"] = d["ClinDx"].map({"AD": "AD", "Normal": "Control"})
    return d


def load_ms(fluid, arm, master, notes):
    path, chan_col = MS_ARMS[(fluid, arm)]
    ms = pd.read_csv(path, index_col=0)
    gis = [c for c in ms.columns if c.endswith("." + GIS_CHANNEL)]
    chan2guid = dict(zip(master[chan_col].dropna().astype(str), master.loc[master[chan_col].notna(), "GUID"]))
    keep = [c for c in ms.columns if c in chan2guid]
    unmapped = [c for c in ms.columns if c not in chan2guid and c not in gis]
    notes.append(f"[{fluid} {arm} MS] proteins {len(ms):,}, channels {len(ms.columns)}, GIS {len(gis)} dropped, "
                 f"mapped to GUID {len(keep)}, batches {len(set(c.split('.')[0] for c in keep))}"
                 + (f", other-cohort channels dropped {len(unmapped)}" if unmapped else ""))
    wide = ms[keep].copy()
    wide.columns = [chan2guid[c] for c in keep]
    batch = pd.Series({chan2guid[c]: c.split(".")[0] for c in keep})
    log2 = np.log2(wide.where(wide > 0))
    # batch quantified if any channel of that batch is non-NA
    bq = pd.DataFrame({b: log2[batch[batch == b].index].notna().any(axis=1) for b in batch.unique()})
    summ = pd.DataFrame({
        "ms_batches": bq.sum(axis=1).astype(int),
        "ms_frac": log2.notna().mean(axis=1),
        "ms_abund": log2.mean(axis=1),
    })
    summ["gene"] = [i.split("|")[0].split(";")[0].strip() for i in summ.index]
    summ["uniprot"] = [i.split("|")[1].strip() if "|" in i else None for i in summ.index]
    return log2, summ, int(bq.shape[1])


def load_affinity(fluid, plat, notes):
    f = pd.read_csv(AFFINITY[fluid][plat], index_col=0)
    f.columns = f.columns.astype(str)
    if plat == "soma":
        vals = np.log2(f.where(f > 0))
    else:
        vals = f
    summ = pd.DataFrame({f"{plat}_abund": vals.mean(axis=1), f"{plat}_n": vals.notna().sum(axis=1)})
    summ["gene"] = [i.split("|")[0].strip() for i in summ.index]
    summ["uniprot"] = [i.split("|")[1].strip() for i in summ.index]
    notes.append(f"[{fluid} {plat}] features {len(f):,}, samples {f.shape[1]}, unique UniProt {summ['uniprot'].nunique():,}, "
                 f"NA fraction {f.isna().mean().mean():.3%}")
    return vals, summ


def best_per_gene(summ, resolve, rank_cols):
    summ = summ.copy()
    summ["ensg"] = [resolve(u, g) for u, g in zip(summ["uniprot"], summ["gene"])]
    summ = summ.dropna(subset=["ensg"])
    return summ.sort_values(rank_cols, ascending=False).drop_duplicates("ensg")


def per_protein_corr(a_vals, a_best, b_vals, b_best, ensg_list):
    shared = [c for c in a_vals.columns if c in b_vals.columns]
    out = {}
    a_idx = a_best.reset_index().set_index("ensg")["index"]
    b_idx = b_best.reset_index().set_index("ensg")["index"]
    for e in ensg_list:
        if e not in a_idx or e not in b_idx:
            continue
        x = a_vals.loc[a_idx[e], shared]
        y = b_vals.loc[b_idx[e], shared]
        m = x.notna() & y.notna()
        if m.sum() < 10:
            continue
        r, p = spearmanr(x[m], y[m])
        out[e] = (r, p, int(m.sum()))
    return pd.DataFrame(out, index=["r", "p", "n"]).T


def build_set(fluid, arm, master, resolve, notes, aff_cache):
    ms_vals, ms_summ, nb = load_ms(fluid, arm, master, notes)
    if fluid not in aff_cache:
        so_vals, so_summ = load_affinity(fluid, "soma", notes)
        ol_vals, ol_summ = load_affinity(fluid, "olink", notes)
        aff_cache[fluid] = (so_vals, best_per_gene(so_summ, resolve, ["soma_abund"]),
                            ol_vals, best_per_gene(ol_summ, resolve, ["olink_abund"]), len(so_summ), len(ol_summ))
    so_vals, so_b, ol_vals, ol_b, n_so, n_ol = aff_cache[fluid]
    ms_b = best_per_gene(ms_summ, resolve, ["ms_batches", "ms_frac", "ms_abund"])
    notes.append(f"[{fluid} {arm}] ENSG-mapped: MS {len(ms_b):,} of {len(ms_summ):,}, SomaScan {len(so_b):,} of {n_so:,}, "
                 f"Olink {len(ol_b):,} of {n_ol:,}")
    g = (so_b[["ensg", "soma_abund"]].merge(ol_b[["ensg", "olink_abund"]], on="ensg", how="outer")
             .merge(ms_b[["ensg", "ms_batches", "ms_frac", "ms_abund", "uniprot"]], on="ensg", how="left"))
    g["ms_batches"] = g["ms_batches"].fillna(0).astype(int)
    g["ms_n_batches"] = nb
    g["ms_frac"] = g["ms_frac"].fillna(0.0)
    g["ms_present"] = (g["ms_batches"] == nb).astype(int)
    g["ms_present2"] = (g["ms_batches"] >= int(np.ceil(2 * nb / 3))).astype(int)
    g["ms_any"] = (g["ms_batches"] >= 1).astype(int)
    g["soma_present"] = g["soma_abund"].notna().astype(int)
    g["olink_present"] = g["olink_abund"].notna().astype(int)
    g["aff_present"] = ((g["soma_present"] == 1) | (g["olink_present"] == 1)).astype(int)
    g["aff_abund_z"] = pd.concat([z(g["soma_abund"]), z(g["olink_abund"])], axis=1).mean(axis=1)
    # specimen-matched agreement
    both = g.loc[(g["soma_present"] == 1) & (g["olink_present"] == 1), "ensg"].tolist()
    c_so = per_protein_corr(so_vals, so_b, ol_vals, ol_b, both)
    g = g.merge(c_so.rename(columns={"r": "r_soma_olink", "p": "p_soma_olink", "n": "n_soma_olink"}),
                left_on="ensg", right_index=True, how="left")
    g["concordant"] = ((g["p_soma_olink"] < CONC_P) & (g["r_soma_olink"] > 0)).astype(int)
    quant = g.loc[g["ms_present"] == 1, "ensg"].tolist()
    c_ms_so = per_protein_corr(ms_vals, ms_b, so_vals, so_b, quant)
    c_ms_ol = per_protein_corr(ms_vals, ms_b, ol_vals, ol_b, quant)
    g = g.merge(c_ms_so[["r"]].rename(columns={"r": "r_ms_soma"}), left_on="ensg", right_index=True, how="left")
    g = g.merge(c_ms_ol[["r"]].rename(columns={"r": "r_ms_olink"}), left_on="ensg", right_index=True, how="left")
    # AD / control abundance covariates
    ad = set(master.loc[master["dx"] == "AD", "GUID"])
    for plat, vals, b in (("soma", so_vals, so_b), ("olink", ol_vals, ol_b)):
        idx = b.reset_index().set_index("ensg")["index"]
        for grp, cols in (("AD", [c for c in vals.columns if c in ad]), ("Control", [c for c in vals.columns if c not in ad])):
            sv = vals.loc[idx.values, cols].mean(axis=1)
            sv.index = idx.index
            g[f"{plat}_abund_{grp}"] = g["ensg"].map(sv)
    g["fluid"], g["arm"], g["set"] = fluid, arm, f"{fluid} {arm}"
    up = pd.concat([so_b[["ensg", "uniprot"]], ol_b[["ensg", "uniprot"]]]).drop_duplicates("ensg").set_index("ensg")["uniprot"]
    g["uniprot"] = g["uniprot"].fillna(g["ensg"].map(up))
    return g


def annotate(g, out):
    pred = pd.read_csv(out / "ml_per_gene_predictions.tsv", sep="\t").drop_duplicates("ensg")
    keep = [c for c in ("ensg", "p_detect_intrinsic", "p_detect_practical", "p_detect_full", "top_dark_reason") if c in pred.columns]
    g = g.merge(pred[keep], on="ensg", how="left")
    fp = pd.read_csv(out / "features_protein.tsv", sep="\t", low_memory=False).drop_duplicates("ensg")
    cols = [c for c in FEAT_COLS if c in fp.columns and c != "uniprot"]
    g = g.merge(fp[["ensg"] + cols], on="ensg", how="left")
    pp = out / "features_peptides.tsv"
    if pp.exists():
        pep = pd.read_csv(pp, sep="\t")[["uniprot"] + PEP_COLS].drop_duplicates("uniprot")
        g = g.merge(pep, on="uniprot", how="left")
    return g


ANCHORS = (("soma", "SomaScan"), ("olink", "Olink"))


def t0_concordance(g, L):
    L.append("T0 specimen-matched agreement, per protein Spearman across shared specimens:")
    for fl in FLUIDS:
        b = g[(g["fluid"] == fl) & (g["arm"] == "neat")].dropna(subset=["r_soma_olink"])
        L.append(f"{fl:16s} SomaScan~Olink on both panels n={len(b):,}  median r {b['r_soma_olink'].median():+.2f}  "
                 f"concordant (p<{CONC_P}, r>0) {int(b['concordant'].sum()):,} ({b['concordant'].mean():.1%})")
    for fl in SETS:
        d = g[g["set"] == fl]
        for plat in ("soma", "olink"):
            q = d.dropna(subset=[f"r_ms_{plat}"])
            if len(q):
                L.append(f"{fl:16s} MS~{plat:5s} among MS-quantified n={len(q):,}  median r {q[f'r_ms_{plat}'].median():+.2f}  "
                         f"r>0.3 {(q[f'r_ms_{plat}'] > 0.3).mean():.1%}")
    L.append("")


def t1_presence(g, L):
    L.append("T1 presence, intrinsic score AUC for MS detection (all batches) among affinity targets:")
    for fl in SETS:
        d = g[g["set"] == fl]
        parts = []
        for ref, lab in ANCHORS + (("concordant", "concordant"),):
            s = d[d[f"{ref}_present" if ref != "concordant" else "concordant"] == 1].dropna(subset=["p_detect_intrinsic"])
            if s["ms_present"].nunique() < 2:
                continue
            parts.append(f"{lab} n={len(s):,} MS-missed {int((s['ms_present'] == 0).sum()):,} AUC {auc(s['ms_present'], s['p_detect_intrinsic']):.3f}")
        L.append(f"   {fl:16s} " + "   ".join(parts))
    L.append("")


def t2_core(g, L, pred_col="p_detect_intrinsic", head="T2 CORE", y="ms_present"):
    lab_y = {"ms_present": "all batches", "ms_present2": ">= 2/3 of batches", "ms_any": ">= 1 batch"}[y]
    L.append(f"{head} MS present ({lab_y}) ~ {pred_col} + affinity abundance:")
    for fl in SETS:
        L.append(f"   [{fl}]")
        d = g[g["set"] == fl].copy()
        d["z_struct"] = d[pred_col]
        for ref, lab in ANCHORS:
            sub = d[d[f"{ref}_present"] == 1].copy()
            sub["z_ab"] = sub[f"{ref}_abund"]
            logit(sub, y, ["z_struct", "z_ab"], f"{lab:10s} on panel      ", L)
            hi = sub[sub["z_ab"] >= sub["z_ab"].median()].copy()
            logit(hi, y, ["z_struct", "z_ab"], f"{lab:10s} top half abund", L)
            for grp in ("AD", "Control"):
                sg = sub.copy()
                sg["z_ab"] = sg[f"{ref}_abund_{grp}"]
                logit(sg, y, ["z_struct", "z_ab"], f"{lab:10s} {grp:7s} abund ", L)
        both = d[(d["soma_present"] == 1) & (d["olink_present"] == 1)].copy()
        both["z_soma"], both["z_olink"] = both["soma_abund"], both["olink_abund"]
        logit(both, y, ["z_struct", "z_soma", "z_olink"], "both       on panels     ", L)
        conc = both[both["concordant"] == 1].copy()
        logit(conc, y, ["z_struct", "z_soma", "z_olink"], "concordant anchor        ", L)
    L.append("")


def t3_specificity(g, L):
    L.append("T3 specificity, score vs SomaScan-Olink agreement, against score vs MS loss on the same rows (per arm):")
    for fl in FLUIDS:
        d = g[(g["fluid"] == fl) & (g["arm"] == "neat")].dropna(subset=["r_soma_olink", "p_detect_intrinsic"]).copy()
        d["z_struct"], d["z_ab"] = d["p_detect_intrinsic"], d["aff_abund_z"]
        r, p = spearmanr(d["r_soma_olink"], d["p_detect_intrinsic"])
        L.append(f"{fl} n on both panels {len(d):,}   r(soma~olink) vs score: Spearman {r:+.3f} p={p:.1e}")
        d["discordant"] = 1 - d["concordant"]
        logit(d, "discordant", ["z_struct", "z_ab"], "discordant ~ score + abundance", L)
        for arm in ARMS:
            da = g[(g["fluid"] == fl) & (g["arm"] == arm)].dropna(subset=["r_soma_olink", "p_detect_intrinsic"]).copy()
            da["z_struct"], da["z_ab"] = da["p_detect_intrinsic"], da["aff_abund_z"]
            logit(da, "ms_present", ["z_struct", "z_ab"], f"MS present, {arm:8s} (same rows)", L)
    L.append("")


def t4_graded(g, L):
    L.append("T4 graded, TMT batches quantifying the protein, among affinity targets")
    for fl in SETS:
        d = g[(g["set"] == fl) & (g["aff_present"] == 1)].dropna(subset=["p_detect_intrinsic"]).copy()
        d["cls"] = np.where(d["ms_batches"] == 0, "never", np.where(d["ms_batches"] == d["ms_n_batches"], "always", "partial"))
        L.append(f"{fl}")
        for c in ("never", "partial", "always"):
            s = d[d["cls"] == c]
            L.append(f"{c:8s} n={len(s):5,}  median score {s['p_detect_intrinsic'].median():.3f}  "
                     f"median affinity z {s['aff_abund_z'].median():+.2f}  median tryptic/kDa {s['tryptic_per_kda'].median():.3f}")
        sub = d[d["cls"] != "always"].copy()
        sub["never"] = (sub["cls"] == "never").astype(int)
        sub["z_struct"], sub["z_ab"] = sub["p_detect_intrinsic"], sub["aff_abund_z"]
        logit(sub, "never", ["z_struct", "z_ab"], "never vs partial", L)
        hi = d[d["aff_abund_z"] >= d["aff_abund_z"].quantile(0.66)]
        L.append(f"top-tertile affinity abundance: never {int((hi['cls'] == 'never').sum()):,}, "
                 f"partial {int((hi['cls'] == 'partial').sum()):,}, always {int((hi['cls'] == 'always').sum()):,}")
    L.append("")


def t5_matched_quant(g, L):
    L.append("T5 matched-specimen quantification, among MS-quantified (all batches), per protein r vs score:")
    for fl in SETS:
        d = g[(g["set"] == fl) & (g["ms_present"] == 1)]
        for plat in ("soma", "olink"):
            s = d.dropna(subset=[f"r_ms_{plat}", "p_detect_intrinsic"])
            if len(s) < 30:
                continue
            r1, p1 = spearmanr(s[f"r_ms_{plat}"], s["p_detect_intrinsic"])
            r2, p2 = spearmanr(s[f"r_ms_{plat}"], s["tryptic_per_kda"], nan_policy="omit")
            r3, p3 = spearmanr(s[f"r_ms_{plat}"], s[f"{plat}_abund"])
            L.append(f"{fl:16s} vs {plat:5s} n={len(s):,}  r(MS~aff)~score {r1:+.3f} p={p1:.1e}  ~tryptic/kDa {r2:+.3f} p={p2:.1e}   "
                     f"~affinity abundance {r3:+.3f} p={p3:.1e}")
    L.append("")


def t6_inference(g, L):
    L.append("T6 inference-dark, no unique tryptic peptide vs MS detection, among affinity targets:")
    for fl in SETS:
        d = g[(g["set"] == fl) & (g["aff_present"] == 1)].dropna(subset=["pep_n_unique"]).copy()
        d["zero_unique"] = (d["pep_n_unique"] == 0).astype(int)
        d["mostly_shared"] = (d["pep_frac_shared"] >= 0.5).astype(int)
        d["z_ab"] = d["aff_abund_z"]
        base = d["ms_present"].mean()
        for flag in ("zero_unique", "mostly_shared"):
            s = d[d[flag] == 1]
            L.append(f"   {fl:16s} {flag:14s} n={len(s):4,}  MS-present {s['ms_present'].mean():.1%} (all affinity targets {base:.1%})")
            logit(d, "ms_present", [flag, "z_ab"], f"{flag:14s} + abundance", L)
    L.append("")


def t8_calibration(g, L):
    L.append("T8 calibration, GTEx-trained P(detect) vs observed MS detection (all batches):")
    for fl in SETS:
        d = g[(g["set"] == fl) & (g["aff_present"] == 1)].dropna(subset=["p_detect_intrinsic"]).copy()
        d["dec"] = pd.qcut(d["p_detect_intrinsic"], 10, labels=False, duplicates="drop")
        t = d.groupby("dec").agg(pred=("p_detect_intrinsic", "mean"), obs=("ms_present", "mean"))
        slope = np.polyfit(t["pred"], t["obs"], 1)[0]
        L.append(f"{fl:16s} n={len(d):,}  " + "  ".join(f"{r.pred:.2f}->{r.obs:.2f}" for r in t.itertuples()) + f"   slope {slope:.2f}")
    L.append("")


def t10_paired(g, L):
    L.append("T10 CSF to plasma (paired, same arm), on both affinity panels in both fluids: MS present in CSF vs plasma")
    for arm in ARMS:
        c = g[(g["fluid"] == "CSF") & (g["arm"] == arm)].set_index("ensg")
        p = g[(g["fluid"] == "Plasma") & (g["arm"] == arm)].set_index("ensg")
        idx = [e for e in c.index.intersection(p.index)
               if c.at[e, "soma_present"] == 1 and c.at[e, "olink_present"] == 1 and p.at[e, "soma_present"] == 1 and p.at[e, "olink_present"] == 1]
        d = pd.DataFrame({"csf": c.loc[idx, "ms_present"].values, "plasma": p.loc[idx, "ms_present"].values,
                          "z_struct": c.loc[idx, "p_detect_intrinsic"].values,
                          "d_ab": (p.loc[idx, "aff_abund_z"].values - c.loc[idx, "aff_abund_z"].values)})
        L.append(f"   [{arm}] n={len(d):,}   CSF-only {int(((d['csf'] == 1) & (d['plasma'] == 0)).sum()):,}   "
                 f"plasma-only {int(((d['csf'] == 0) & (d['plasma'] == 1)).sum()):,}   "
                 f"both {int(((d['csf'] == 1) & (d['plasma'] == 1)).sum()):,}   neither {int(((d['csf'] == 0) & (d['plasma'] == 0)).sum()):,}")
        both = d[(d["csf"] == 1) | (d["plasma"] == 1)].copy()
        logit(both, "plasma", ["z_struct", "d_ab"], "MS in plasma among MS in either ~ score + affinity change", L)
    L.append("")


def t11_depletion(g, L):
    L.append("T11 neat to depleted (paired, same fluid) among affinity targets: what depletion recovers")
    L.append("gained = never in neat, present (all batches) in depleted; among neat-never targets.")
    for fl in FLUIDS:
        n = g[(g["fluid"] == fl) & (g["arm"] == "neat")].set_index("ensg")
        d = g[(g["fluid"] == fl) & (g["arm"] == "depleted")].set_index("ensg")
        idx = n.index.intersection(d.index)
        idx = [e for e in idx if n.at[e, "aff_present"] == 1]
        t = pd.DataFrame({"neat": n.loc[idx, "ms_present"].values, "depl": d.loc[idx, "ms_present"].values,
                          "neat_any": n.loc[idx, "ms_any"].values, "depl_any": d.loc[idx, "ms_any"].values,
                          "z_struct": n.loc[idx, "p_detect_intrinsic"].values,
                          "z_yield": n.loc[idx, "tryptic_per_kda"].values,
                          "z_ab": n.loc[idx, "aff_abund_z"].values})
        L.append(f"{fl} affinity targets {len(t):,}   MS present neat {int(t['neat'].sum()):,} -> depleted {int(t['depl'].sum()):,}   "
                 f"never in neat {int((t['neat_any'] == 0).sum()):,}, of which any-batch in depleted {int(((t['neat_any'] == 0) & (t['depl_any'] == 1)).sum()):,} "
                 f"and all-batch in depleted {int(((t['neat_any'] == 0) & (t['depl'] == 1)).sum()):,}")
        nn = t[t["neat_any"] == 0].copy()
        nn["gained"] = nn["depl"]
        logit(nn, "gained", ["z_struct", "z_ab"], "gained (all batches) ~ score + affinity abundance", L)
        logit(nn, "gained", ["z_yield", "z_ab"], "gained (all batches) ~ tryptic/kDa + affinity abundance", L)
        nn["gained_any"] = nn["depl_any"]
        logit(nn, "gained_any", ["z_struct", "z_ab"], "gained (any batch)  ~ score + affinity abundance", L)
        lost = t[t["neat"] == 1].copy()
        lost["kept"] = lost["depl"]
        logit(lost, "kept", ["z_struct", "z_ab"], "kept after depletion ~ score + affinity abundance", L)
    L.append("")


notes = []
master = load_master()
resolve = build_resolver()
aff_cache = {}
g = pd.concat([build_set(fl, arm, master, resolve, notes, aff_cache) for fl, arm in MS_ARMS], ignore_index=True)
g = annotate(g, TAB_DIR)
g.to_csv(TAB_DIR / "biofluid_gene_table.tsv", sep="\t", index=False)

L = ["Biofluid detectability (Emory matched CSF and plasma, TMT-MS + SomaScan + Olink)", "",
     f"specimens: {master['Core Set'].eq(True).sum()} core, "
     f"AD {int((master.loc[master['Core Set'].eq(True), 'dx'] == 'AD').sum())} / Control "
     f"{int((master.loc[master['Core Set'].eq(True), 'dx'] == 'Control').sum())}; "
     f"diagnosis is a stratification check only, no disease claim is made.",
     "MS present = quantified in every TMT batch of the arm. Affinity presence = on the panel.",
     f"Concordant anchor = SomaScan and Olink correlate across shared specimens (Spearman p<{CONC_P}, r>0).", ""]
L += notes
L.append("")
for fl in SETS:
    d = g[g["set"] == fl]
    L.append(f"[{fl}] ENSG: on SomaScan {int(d['soma_present'].sum()):,}  on Olink {int(d['olink_present'].sum()):,}  "
             f"on both {int(((d['soma_present'] == 1) & (d['olink_present'] == 1)).sum()):,}  concordant {int(d['concordant'].sum()):,}  "
             f"MS all batches {int(d['ms_present'].sum()):,}  partial {int(((d['ms_batches'] > 0) & (d['ms_present'] == 0)).sum()):,}  "
             f"never {int((d['ms_batches'] == 0).sum()):,}")
L.append("")
t0_concordance(g, L)
t1_presence(g, L)
t2_core(g, L)
t2_core(g, L, head="T2 sensitivity", y="ms_present2")
t2_core(g, L, pred_col="tryptic_per_kda", head="T2b raw feature")
t3_specificity(g, L)
t4_graded(g, L)
t5_matched_quant(g, L)
t6_inference(g, L)
t8_calibration(g, L)
t10_paired(g, L)
t11_depletion(g, L)
text = "\n".join(L)
(TAB_DIR / "biofluid_validation.txt").write_text(text)
print(text)
