"""
Functions for project
"""
from typing import Tuple, Any

import pandas as pd
import numpy as np
from config import TISSUES, SUBSITE_AGG, IHC_ABSENT, IHC_PRESENT, RELIABILITY_ORDER, PAXDB_DIR, ISOFORM_PICK


# General handling
def symbol_to_ensg(cross, tag="", verbose=True):
    """symbol->ENSG map that reports ambiguity """
    c = (cross.dropna(subset=["symbol", "ensg"])[["symbol", "ensg"]]
              .drop_duplicates())
    dup = c["symbol"].duplicated(keep=False)
    n_ambig = c.loc[dup, "symbol"].nunique()
    n_alt_dropped = int(dup.sum()) - n_ambig
    if verbose:
        print(f"map {tag} {len(c):,} symbol-ENSG pairs; {n_ambig:,} symbols "
              f"are ambiguous (>1 ENSG); {n_alt_dropped:,} alt rows dropped (kept first)")
    return c.drop_duplicates("symbol").set_index("symbol")["ensg"]


def _attach_ensg(df: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    sym2ensg = symbol_to_ensg(cross, tag="paxdb")
    df = df.copy()
    df["ensg"] = df["symbol"].map(sym2ensg)
    return df


def _agg_subsites(df: pd.DataFrame, cols: list[str], how: str) -> pd.Series:
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    return sub.mean(axis=1) if how == "mean" else sub[cols[0]]


# Gtex processing
def load_gtex(med: pd.DataFrame, ts:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    med = med.rename(columns={med.columns[0]: "ensg"})
    med["ensg"] = med["ensg"].str.strip()

    cross = ts.rename(columns={"ensembl_id": "ensg", "entrez_id": "entrez",
                               "hgnc_symbol": "symbol", "hgnc_name": "hgnc_name"})
    cross["ensg"] = cross["ensg"].str.strip()

    rows = []
    for tname, spec in TISSUES.items():
        gcols = spec["gtex"]
        if not gcols:
            continue
        missing = [c for c in gcols if c not in med.columns]
        if missing:
            raise KeyError(f"{tname}: GTEx columns not found: {missing}")
        level = _agg_subsites(med, gcols, SUBSITE_AGG)
        block = pd.DataFrame({"ensg": med["ensg"], "tissue": tname, "gtex_level": level})
        block["gtex_measured"] = block["gtex_level"].notna()
        rows.append(block)
    long = pd.concat(rows, ignore_index=True)
    return long, cross


# HPA processing
def _call_from_level(level: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=level.index, dtype="object")
    out[level.isin(IHC_PRESENT)] = "present"
    out[level.isin(IHC_ABSENT)] = "absent"
    return out


def load_ihc(df: pd.DataFrame) -> tuple[Any, Any, Any, Any]:
    df = df.rename(columns={"Gene": "ensg", "Gene name": "symbol",
                            "Tissue": "hpa_tissue", "Cell type": "cell_type",
                            "Level": "level", "Reliability": "reliability"})
    for c in df.columns:
        df[c] = df[c].str.strip()

    level_audit = df["level"].value_counts(dropna=False).rename_axis("level").reset_index(name="rows")

    gene_dict = (df[["ensg", "symbol"]].dropna().drop_duplicates().drop_duplicates("ensg"))

    hpa2canon = {spec["hpa"]: t for t, spec in TISSUES.items()}
    df = df[df["hpa_tissue"].isin(hpa2canon)].copy()
    df["tissue"] = df["hpa_tissue"].map(hpa2canon)
    df["ihc_call"] = _call_from_level(df["level"])

    celltype = df[["ensg", "tissue", "cell_type", "ihc_call", "level", "reliability"]].copy()

    # collapse cell types -> tissue level call
    rel_rank = {r: i for i, r in enumerate(RELIABILITY_ORDER)}
    scored = df[df["ihc_call"].notna()].copy()
    scored["is_pos"] = (scored["ihc_call"] == "present").astype(int)
    scored["rel_rank"] = scored["reliability"].map(rel_rank)

    g = scored.groupby(["ensg", "tissue"])
    tissue_tbl = g.agg(
        n_celltypes=("ihc_call", "size"),
        n_pos_celltypes=("is_pos", "sum"),
        best_rel_rank=("rel_rank", "min"),
    ).reset_index()
    tissue_tbl["ihc_present"] = tissue_tbl["n_pos_celltypes"] > 0
    tissue_tbl["frac_pos_celltypes"] = (tissue_tbl["n_pos_celltypes"] / tissue_tbl["n_celltypes"])
    inv_rank = {i: r for r, i in rel_rank.items()}
    tissue_tbl["best_reliability"] = tissue_tbl["best_rel_rank"].map(inv_rank)
    tissue_tbl = tissue_tbl.drop(columns=["best_rel_rank"])

    return gene_dict, celltype, tissue_tbl, level_audit


# --- Paxdb processing
def _read_paxdb_file(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#",
                     names=["symbol", "string_id", "paxdb_ppm"], dtype=str)
    df["paxdb_ppm"] = pd.to_numeric(df["paxdb_ppm"], errors="coerce")
    df["ensp"] = df["string_id"].str.replace(r"^\d+\.", "", regex=True)  # drop 9606.
    df = df.drop(columns=["string_id"]).dropna(subset=["paxdb_ppm"])
    return df


def load_paxdb(cross: pd.DataFrame):
    rows = []
    for tname, spec in TISSUES.items():
        f = PAXDB_DIR / f"{spec['paxdb']}.txt"
        if not f.exists():
            print(f"  [skip] {tname}: {f.name} not present")
            continue
        d = _attach_ensg(_read_paxdb_file(f), cross)
        d["tissue"] = tname
        rows.append(d[["ensg", "tissue", "paxdb_ppm", "ensp", "symbol"]])
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    wb_path = PAXDB_DIR / "hs_whole_body.txt"
    wb = _attach_ensg(_read_paxdb_file(wb_path), cross)
    wb = wb.rename(columns={"paxdb_ppm": "paxdb_ppm_global"})[
            ["ensg", "paxdb_ppm_global", "ensp", "symbol"]]
    return long, wb


# Combining HPA, gtex, paxdb together
def _load(out):
    gene_dict = pd.read_csv(out / "gene_dict.tsv", sep="\t", dtype=str)
    gtex = pd.read_csv(out / "gtex_long.tsv", sep="\t",
                       dtype={"ensg": str, "tissue": str, "gtex_level": float})
    gtex["gtex_measured"] = gtex["gtex_measured"].astype(str).str.lower().eq("true")
    ihc = pd.read_csv(out / "ihc_tissue.tsv", sep="\t")
    ihc["ensg"] = ihc["ensg"].astype(str)
    ihc["ihc_present"] = ihc["ihc_present"].astype(str).str.lower().eq("true")
    pax = pd.read_csv(out / "paxdb_long.tsv", sep="\t",
                      dtype={"ensg": str, "tissue": str})
    pax = pax.dropna(subset=["ensg"])
    wb_path = out / "paxdb_wholebody.tsv"
    wb = pd.read_csv(wb_path, sep="\t", dtype={"ensg": str}) if wb_path.exists() \
        else pd.DataFrame(columns=["ensg", "paxdb_ppm_global"])
    wb = wb.dropna(subset=["ensg"]).drop_duplicates("ensg")
    return gene_dict, gtex, ihc, pax, wb


def build_grid(out):
    gene_dict, gtex, ihc, pax, wb = _load(out)

    ihc_universe = set(gene_dict["ensg"])
    gtex_universe = set(gtex["ensg"])
    master = sorted(ihc_universe | gtex_universe)

    tissues = list(TISSUES.keys())
    tri = {t for t, s in TISSUES.items() if s["gtex"] is not None}

    # skeleton
    import itertools
    grid = pd.DataFrame(itertools.product(master, tissues), columns=["ensg", "tissue"])
    grid = grid.merge(gene_dict, on="ensg", how="left")

    # --- IHC join ---
    grid = grid.merge(
        ihc[["ensg", "tissue", "ihc_present", "frac_pos_celltypes",
             "n_celltypes", "n_pos_celltypes", "best_reliability"]],
        on=["ensg", "tissue"], how="left")
    in_hpa = grid["ensg"].isin(ihc_universe)
    has_ihc_row = grid["ihc_present"].notna()
    grid["ihc_status"] = "unscored"
    grid.loc[~in_hpa, "ihc_status"] = "not_in_hpa"
    grid.loc[has_ihc_row & (grid["ihc_present"] == True), "ihc_status"] = "present"
    grid.loc[has_ihc_row & (grid["ihc_present"] == False), "ihc_status"] = "absent"

    # --- GTEx join ---
    grid = grid.merge(gtex[["ensg", "tissue", "gtex_level", "gtex_measured"]],
                      on=["ensg", "tissue"], how="left")
    in_gtex = grid["ensg"].isin(gtex_universe)
    is_tri = grid["tissue"].isin(tri)
    grid["gtex_status"] = "not_applicable"
    grid.loc[is_tri & ~in_gtex, "gtex_status"] = "not_measured"
    grid.loc[is_tri & in_gtex & (grid["gtex_measured"] == True), "gtex_status"] = "measured"
    grid.loc[is_tri & in_gtex & (grid["gtex_measured"] != True), "gtex_status"] = "measured_absent"

    # --- PaxDb join ---
    grid = grid.merge(pax[["ensg", "tissue", "paxdb_ppm"]].drop_duplicates(["ensg", "tissue"]),
                      on=["ensg", "tissue"], how="left")
    grid = grid.merge(wb[["ensg", "paxdb_ppm_global"]], on="ensg", how="left")
    grid["paxdb_present"] = grid["paxdb_ppm"].notna()

    # --- agreement class (ihc x gtex) ---
    ic = grid["ihc_status"]
    gc = grid["gtex_status"]
    grid["agreement_class"] = pd.NA
    grid.loc[(ic == "present") & (gc == "measured"), "agreement_class"] = "both_present"
    grid.loc[(ic == "absent") & (gc == "measured_absent"), "agreement_class"] = "both_absent"
    grid.loc[(ic == "present") & (gc == "measured_absent"), "agreement_class"] = "IHC_only"
    grid.loc[(ic == "absent") & (gc == "measured"), "agreement_class"] = "GTEx_only"

    grid["ms_dark_candidate"] = (ic == "present") & gc.isin(["measured_absent", "not_measured"])

    cols = ["ensg", "symbol", "tissue",
            "ihc_status", "ihc_present", "frac_pos_celltypes",
            "n_celltypes", "n_pos_celltypes", "best_reliability",
            "gtex_status", "gtex_level",
            "paxdb_present", "paxdb_ppm", "paxdb_ppm_global",
            "agreement_class", "ms_dark_candidate"]
    return grid[cols], tri


def _qc(grid) -> str:
    L = [f"grid rows: {len(grid):,}  "
         f"({grid['ensg'].nunique():,} genes x {grid['tissue'].nunique()} tissues)", "",
         "ihc_status:", grid["ihc_status"].value_counts(dropna=False).to_string(), "", "gtex_status:",
         grid["gtex_status"].value_counts(dropna=False).to_string(), "", "agreement_class (tri-source tissues only):",
         grid["agreement_class"].value_counts(dropna=False).to_string(), "",
         f"ms_dark_candidates: {int(grid['ms_dark_candidate'].sum()):,}"]
    md = grid[grid["ms_dark_candidate"]]
    L.append("by gtex_status:")
    L.append("  " + md["gtex_status"].value_counts().to_string().replace("\n", "\n  "))
    L.append("")
    # PaxDb abundance coverage among IHC-present grid rows
    ihc_pos = grid[grid["ihc_status"] == "present"]
    cov_tissue = ihc_pos["paxdb_ppm"].notna().mean() if len(ihc_pos) else float("nan")
    cov_any = (ihc_pos["paxdb_ppm"].notna() | ihc_pos["paxdb_ppm_global"].notna()).mean() if len(ihc_pos) else float("nan")
    L.append("PaxDb abundance coverage among IHC-present rows:")
    L.append(f"tissue-matched ppm: {cov_tissue:.1%}")
    L.append(f"tissue or whole-body ppm: {cov_any:.1%}")
    return "\n".join(L)


# QC utils
def audit(out):
    gd = pd.read_csv(out / "gene_dict.tsv", sep="\t", dtype=str)
    sym2ensg = (gd.dropna(subset=["symbol", "ensg"]).drop_duplicates("symbol")
                  .set_index("symbol")["ensg"])

    wb = _read_paxdb_file(PAXDB_DIR / "hs_whole_body.txt")
    wb = wb[wb["paxdb_ppm"] > 0].copy()
    wb["ensg"] = wb["symbol"].map(sym2ensg)
    wb["mapped"] = wb["ensg"].notna()
    wb["log_ppm"] = np.log10(wb["paxdb_ppm"])
    return wb


def compare(wb) -> str:
    m = wb.loc[wb["mapped"], "log_ppm"]
    u = wb.loc[~wb["mapped"], "log_ppm"]
    L = ["Mapping check: PaxDb whole body abundance, mapped vs unmapped", "",
         f"total proteins (ppm>0): {len(wb):,}",
         f"mapped to ENSG:   {len(m):,} ({len(m) / len(wb):.1%})",
         f"unmapped (dropped): {len(u):,} ({len(u) / len(wb):.1%})", ""]
    if len(u) < 20 or len(m) < 20:
        L.append("too few in one group to test")
        return "\n".join(L)

    def q(s):
        return (f"median {s.median():+.2f}  IQR [{s.quantile(.25):+.2f}, "
                f"{s.quantile(.75):+.2f}]  (log10 ppm)")
    L.append(f"mapped:   {q(m)}")
    L.append(f"unmapped: {q(u)}")
    L.append("")

    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(m, u, alternative="two-sided")
    rbc = 1 - 2 * U / (len(m) * len(u))
    direction = ("unmapped lower abundance" if m.median() > u.median()
                 else "unmapped higher abundance" if m.median() < u.median()
                 else "no median difference")
    L.append(f"Mann-Whitney U: p={p:.2e} rank-biserial={rbc:+.3f}   ({direction})")
    L.append("")
    mag = abs(rbc)
    band = ("negligible" if mag < 0.1 else "small" if mag < 0.3
            else "moderate" if mag < 0.5 else "large")
    L.append(f"effect size is {band}.")
    if mag < 0.1:
        L.append("=> unmapped proteins are not materially different in abundance;")
        L.append("the 16% loss is ignorable w.r.t. the dominant detectability axis.")
    else:
        lo = "lower" if m.median() > u.median() else "higher"
        L.append(f"=> unmapped proteins skew {lo}-abundance dropping them makes the")
        L.append(f"blind-spot estimate {'conservative' if lo=='lower' else 'anti-conservative'}.")
    return "\n".join(L)


# coverage bias
def _build(out):
    grid = pd.read_csv(out / "grid.tsv", sep="\t")
    tri = {t for t, s in TISSUES.items() if s["gtex"] is not None}
    m = grid[(grid["ihc_status"] == "present") & (grid["tissue"].isin(tri))].copy()

    # same abundance definition as model_msdark (tissue ppm, whole-body fallback),
    # but don't drop the uncovered
    ab = m["paxdb_ppm"].where(m["paxdb_ppm"].notna(), m["paxdb_ppm_global"])
    m["abundance_covered"] = ab.notna() & (ab > 0)
    # MS-dark outcome, identical coding to model_msdark (0 = dark here)
    m["gtex_detected"] = (m["gtex_status"] == "measured").astype(int)
    m["ms_dark"] = 1 - m["gtex_detected"]
    return m


def _report(m) -> str:
    cov = m[m["abundance_covered"]]
    drop = m[~m["abundance_covered"]]
    N = len(m)
    L = ["Abundance coverage bias control (IHC-present tri-source rows)", "", f"IHC-present rows: {N:,}",
         f"abundance-covered (modelled): {len(cov):,} ({len(cov) / N:.1%})",
         f"dropped for no ppm: {len(drop):,} ({len(drop) / N:.1%})", ""]
    if len(drop) < 20:
        L.append("too few dropped rows to test -> loss is negligible by count")
        return "\n".join(L)

    dark_cov = cov["ms_dark"].mean()
    dark_drop = drop["ms_dark"].mean()
    L.append(f"MS-dark rate, covered: {dark_cov:.1%}")
    L.append(f"MS-dark rate, dropped: {dark_drop:.1%}")
    L.append(f"difference (dropped - covered): {dark_drop - dark_cov:+.1%}")
    L.append("")

    # 2x2: covered/dropped x dark/detected.  Fisher for enrichment
    from scipy.stats import fisher_exact
    a = int((drop["ms_dark"] == 1).sum()); b = int((drop["ms_dark"] == 0).sum())
    c = int((cov["ms_dark"] == 1).sum());  d = int((cov["ms_dark"] == 0).sum())
    orr, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    L.append(f"Fisher exact (dropped enriched for dark?): OR={orr:.2f}  p={p:.2e}")
    L.append("")
    if p >= 0.05:
        L.append("dropped rows are not differentially MS-dark")
        L.append("loss is inert w.r.t. the blind-spot estimate.")
    elif orr > 1:
        L.append("dropped rows are enriched for MS-dark: the cross-atlas identifier")
    else:
        L.append("dropped rows are less often MS-dark (OR<1), loss mildly inflates")
    return "\n".join(L)


# PAxdb vs gtex weight
def _auc(y, x):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(x)
    return float(roc_auc_score(y[m], x[m])) if len(np.unique(y[m])) > 1 else float("nan")


def _read_weights(path):
    if not path.exists():
        print(f"missing {path.name}]")
        return {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("#weights:"):
                body = line.split(":", 1)[1].strip().rstrip(";")
                out = {}
                for item in body.split(";"):
                    if ":" in item:
                        k, v = item.rsplit(":", 1)
                        out[k.strip()] = float(v)
                return out
    return {}


# ML
def _build_table(out):
    feats = pd.read_csv(out / "features_protein.tsv", sep="\t")
    mt = pd.read_csv(out / "model_table.tsv", sep="\t")
    g = mt.groupby("ensg").agg(frac_measured=("gtex_detected", "mean"),
                               log_abundance=("log_abundance", "median"),
                               n_tissues=("gtex_detected", "size")).reset_index()
    g["y"] = (g["frac_measured"] > 0).astype(int)
    pep_path = out / "features_peptides.tsv"
    if pep_path.exists() and "uniprot" in feats.columns:
        pep = pd.read_csv(pep_path, sep="\t")
        feats = feats.merge(pep, on="uniprot", how="left")
    else:
        print("features_peptides.tsv not found -> peptide-level groups unavailable")
    st_path = out / "features_structure.tsv"
    if st_path.exists() and "uniprot" in feats.columns:
        st = pd.read_csv(st_path, sep="\t").drop(columns=["ensg"], errors="ignore")
        feats = feats.merge(st, on="uniprot", how="left")
    else:
        print("features_structure.tsv not found -> AlphaFold and half-life groups unavailable")
    df = g.merge(feats, on="ensg", how="left")
    n_dup = int(df["ensg"].duplicated().sum())
    if n_dup:
        print(f"dropped {n_dup} duplicate ENSG rows after feature merge (first kept)")
        df = df.drop_duplicates("ensg")
    return df

def _merge_size_tryptic(groups):
    g = {k: v for k, v in groups.items() if k not in ("size", "tryptic_yield")}
    g["size_peptide_axis"] = groups.get("size", []) + groups.get("tryptic_yield", [])
    return g


def select_groups(df_tr, groups, thresh=0.8, n_boot=50, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    cols = _cols(groups)
    imp = SimpleImputer(strategy="median").fit(df_tr[cols].astype(float))
    sc = StandardScaler().fit(imp.transform(df_tr[cols].astype(float)))
    X = sc.transform(imp.transform(df_tr[cols].astype(float)))
    y = df_tr["y"].values
    rng = np.random.default_rng(seed)
    sel = np.zeros(len(cols))
    n = len(df_tr)
    for _ in range(n_boot):
        idx = rng.choice(n, int(0.6 * n), replace=False)
        m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=500)
        m.fit(X[idx], y[idx])
        sel += np.abs(m.coef_[0]) > 1e-8
    freq = dict(zip(cols, sel / n_boot))
    grp_freq = {g: float(np.max([freq[c] for c in v])) for g, v in groups.items()}
    selected = [g for g, f in grp_freq.items() if f >= thresh] or list(groups)
    return selected, grp_freq


def _cols(groups):
    return [c for v in groups.values() for c in v]


def nested_oof(df, groups, n_splits, default, grid, n_inner, seed=0, tune=False):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    y = df["y"].values
    grp = df["ensg"].values
    oof_gbm = np.full(len(df), np.nan)
    oof_log = np.full(len(df), np.nan)
    sel_count = {g: 0 for g in groups}
    chosen = []
    for tr, te in cv.split(df, y, groups=grp):
        dtr, dte = df.iloc[tr], df.iloc[te]
        selected, _ = select_groups(dtr, groups, seed=seed)
        for g in selected:
            sel_count[g] += 1
        scols = _cols({g: groups[g] for g in selected})
        Xtr = dtr[scols].astype(float).values
        Xte = dte[scols].astype(float).values
        d, lr = tune_gbm(Xtr, dtr["y"].values, dtr["ensg"].values, default, grid, n_inner, seed) if tune else default
        chosen.append((d, lr))
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=lr,
                                             max_depth=d, random_state=0).fit(Xtr, dtr["y"].values)
        oof_gbm[te] = gbm.predict_proba(Xte)[:, 1]
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000)).fit(Xtr, dtr["y"].values)
        oof_log[te] = pipe.predict_proba(Xte)[:, 1]
    sel_freq = {g: sel_count[g] / n_splits for g in groups}
    return oof_gbm, oof_log, sel_freq, chosen


def _boot_ci(y, p, fn, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return float(fn(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _diff_ci(y, p_a, p_b, fn, n=1000, seed=2):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx]))
    return float(fn(y, p_a) - fn(y, p_b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def present(group_dict, df):
    return {k: [c for c in v if c in df.columns] for k, v in group_dict.items()
            if any(c in df.columns for c in v)}


def calibration(y, p):
    from sklearn.calibration import calibration_curve
    fp, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return list(zip(mp, fp))


def perm_importance(df, groups, n_rep=20, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    cols = _cols(groups)
    Xtr, Xte, ytr, yte = train_test_split(df[cols].astype(float).values, df["y"].values,
                                          test_size=0.3, stratify=df["y"].values, random_state=seed)
    mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=4, random_state=0).fit(Xtr, ytr)
    base = roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1])
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(cols)}
    res = {}
    for g, members in groups.items():
        gi = [idx[c] for c in members]
        drops = []
        for _ in range(n_rep):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            for j in gi:
                Xp[:, j] = Xp[perm, j]
            drops.append(base - roc_auc_score(yte, mdl.predict_proba(Xp)[:, 1]))
        res[g] = (float(np.mean(drops)), float(np.std(drops)))
    return base, dict(sorted(res.items(), key=lambda kv: -kv[1][0]))


def per_gene_attribution(df, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    cols = _cols(groups)
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Z = sc.fit_transform(imp.fit_transform(df[cols].astype(float)))
    m = LogisticRegression(max_iter=2000).fit(Z, df["y"].values)
    coef = dict(zip(cols, m.coef_[0]))
    contrib = {g: (Z[:, [cols.index(c) for c in v]] *
                   np.array([coef[c] for c in v])).sum(axis=1) for g, v in groups.items()}
    out = pd.DataFrame(contrib, index=df["ensg"].values)
    out["top_dark_reason"] = out.idxmin(axis=1)
    return out.reset_index().rename(columns={"index": "ensg"})


def transfer(out, df, oof_intrinsic):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    d = df[["ensg"]].copy(); d["p"] = oof_intrinsic
    lines = []
    pa = pd.read_csv(out / "peptideatlas_per_gene.tsv", sep="\t")[
        ["ensg", "pa_n_peptides", "pa_observed"]]
    a = d.merge(pa, on="ensg", how="inner")
    r, pv = spearmanr(a["p"], a["pa_n_peptides"])
    lines.append(f"PeptideAtlas (n={len(a):,}): Spearman(P_intrinsic, observed peptides) "
                 f"r={r:+.3f} p={pv:.1e}")
    if a["pa_observed"].nunique() > 1:
        lines.append(f"AUC predicting PeptideAtlas observed: "
                     f"{roc_auc_score(a['pa_observed'], a['p']):.3f}")
    px = pd.read_csv(out / "paxdb_wholebody.tsv", sep="\t")[["ensg", "paxdb_ppm_global"]]
    px = px.dropna().drop_duplicates("ensg")
    pres = d.merge(px, on="ensg", how="inner")
    r, pv = spearmanr(pres["p"], np.log10(pres["paxdb_ppm_global"]))
    lines.append(f"PaxDb whole-body abundance (n={len(pres):,}; supporting, not load-bearing):")
    lines.append(f"Spearman(P_intrinsic, log PaxDb abundance): r={r:+.3f} p={pv:.1e}")
    return "\n".join(lines)


def actionable_readout(df, oof_intrinsic):
    L = []
    t = df["n_tryptic_7_30"]
    qlab = ["Q1 fewest", "Q2", "Q3", "Q4 most"]
    q = pd.qcut(t, 4, labels=qlab)
    L.append("detection rate by tryptic-fragment yield (the 'looks like X -> odds Y' rule):")
    for lev in qlab:
        m = (q == lev).values
        L.append(f"{lev:10s}  median fragments {t[m].median():4.0f}   "
                 f"detected {df.loc[m, 'y'].mean():.1%}   (n={int(m.sum()):,})")
    ab = pd.qcut(df["log_abundance"], 3, labels=["lo", "mid", "hi"])
    sub = df[(ab == "mid").values].copy()
    sub["tq"] = pd.qcut(sub["n_tryptic_7_30"], 4, labels=qlab)
    L.append("same, within the middle abundance third (same abundance, varying yield):")
    for lev in qlab:
        m = (sub["tq"] == lev).values
        L.append(f"{lev:10s}  detected {sub.loc[m, 'y'].mean():.1%}   (n={int(m.sum()):,})")
    d = df[["y"]].copy(); d["p"] = oof_intrinsic
    base = float((d["y"] == 0).mean())
    d = d.sort_values("p")
    L.append(f"triage by abundance-free score (base MS-dark rate {base:.1%}):")
    for frac in (0.05, 0.10, 0.20):
        k = int(len(d) * frac)
        dr = float((d.head(k)["y"] == 0).mean())
        L.append(f"lowest {int(frac*100):2d}% scored -> {dr:.1%} are MS-dark "
                 f"= {dr/base:.1f}x base rate")
    return "\n".join(L)


def tune_gbm(Xtr, ytr, gtr, default, grid, n_inner, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
    splits = list(inner.split(Xtr, ytr, groups=gtr))
    best, best_auc = default, -1.0
    for d, lr in grid:
        aucs = []
        for a, b in splits:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=lr, max_depth=d,
                                               random_state=0).fit(Xtr[a], ytr[a])
            aucs.append(roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))
        if np.mean(aucs) > best_auc:
            best, best_auc = (d, lr), float(np.mean(aucs))
    return best


