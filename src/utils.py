"""
Functions for project
"""
from typing import Tuple, Any

import pandas as pd
from config import TISSUES, SUBSITE_AGG, IHC_ABSENT, IHC_PRESENT, RELIABILITY_ORDER, PAXDB_DIR


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
    import numpy as np
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