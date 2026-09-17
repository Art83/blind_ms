"""
For every protein on an affinity panel or in the nominated biomarker lists,
the experiment that could actually validate it

"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd


import uniprot_annot as UA
from utils_features import WIN, MZ_WIN, SEQUON
from utils_features import _digest, _pep_mass, PROTON, ENZYMES, ALTERNATIVES, digest_with as enzyme_digest
from config import DATA_DIR, TAB_DIR

UNIPROT_TSV = DATA_DIR / "uniprot_human.tsv"
TOP_N = 5
ROUTE = {
    "confirmable":       "shotgun MS replication",
    "chemistry_limited": "targeted MS (PRM/SRM) on listed unique peptides",
    "inference_limited": "targeted MS on listed peptides, or affinity replication",
    "inference_locked":  "alternative protease + targeted MS if listed, else affinity/immunoassay only",
    "unscored":          "not scored (no reviewed sequence)",
}


def load_proteome():
    df = pd.read_csv(UNIPROT_TSV, sep="\t", dtype=str, keep_default_na=False)
    acc = df["Entry"].str.strip()
    gene = df["Gene Names (primary)"].str.strip().str.split("[;\\s]+", regex=True).str[0].fillna("")
    gene = gene.where(gene != "", acc)
    seqs = df["Sequence"].str.strip().str.upper()
    return df, acc.tolist(), gene.tolist(), seqs.tolist()


def window_peptides(seq, rule):
    """(peptide, start, end) in the observable window, precursor coordinates."""
    if rule is None:
        return [(p, s, e) for p, s, e in _digest(seq) if WIN[0] <= len(p) <= WIN[1]]
    pos, out = 0, []
    for frag in enzyme_digest(seq, rule):
        if not frag:
            continue
        s, e = pos, pos + len(frag)
        pos = e
        if WIN[0] <= len(frag) <= WIN[1]:
            out.append((frag, s, e))
    return out


def build_indices(genes, seqs):
    """Per enzyme: peptide (L/I collapsed) -> set of genes."""
    idx = {}
    for enz in ["trypsin"] + ALTERNATIVES:
        rule = None if enz == "trypsin" else ENZYMES[enz]
        d = defaultdict(set)
        for g, s in zip(genes, seqs):
            for p, _, _ in window_peptides(s, rule):
                d[p.replace("I", "L")].add(g)
        idx[enz] = d
        print(f"    index {enz}: {len(d):,} peptides")
    return idx


def mz_ok(p):
    m = _pep_mass(p)
    return any(MZ_WIN[0] <= (m + z * PROTON) / z <= MZ_WIN[1] for z in (2, 3))


def describe(acc, gene, seq, row, idx):
    """Unique peptides, clean peptides, paralogues, best alternative protease."""
    rd = row.to_dict()
    m_start, m_end = UA.mature_range(rd, len(seq))
    sites = UA.site_positions(rd)
    ptm_pos = sites["glyco"] | sites["lipid"]
    out = {}
    shared_with = set()
    for enz in ["trypsin"] + ALTERNATIVES:
        rule = None if enz == "trypsin" else ENZYMES[enz]
        peps = [(p, s, e) for p, s, e in window_peptides(seq, rule) if s + 1 >= m_start and e <= m_end]
        uniq, clean = [], []
        for p, s, e in peps:
            owners = idx[enz].get(p.replace("I", "L"), set())
            if owners - {gene}:
                if enz == "trypsin":
                    shared_with |= (owners - {gene})
                continue
            uniq.append(p)
            if (mz_ok(p) and "M" not in p and "C" not in p and not SEQUON.search(p)
                    and not any(q in ptm_pos for q in range(s + 1, e + 1))):
                clean.append(p)
        out[enz] = (uniq, clean)
    tu, tc = out["trypsin"]
    tc_sorted = sorted(tc, key=len)
    best_alt, best_n = None, 0
    for enz in ALTERNATIVES:
        n = len(out[enz][1])
        if n > best_n:
            best_alt, best_n = enz, n
    return dict(
        n_unique_tryptic=len(tu),
        n_clean_tryptic=len(tc),
        clean_tryptic_peptides=";".join(tc_sorted[:TOP_N]),
        unique_tryptic_peptides=";".join(sorted(tu, key=len)[:TOP_N]) if not tc else "",
        shared_with=";".join(sorted(shared_with)[:8]),
        n_shared_genes=len(shared_with),
        best_alt_protease=best_alt or "",
        n_clean_best_alt=best_n,
        best_alt_peptides=";".join(sorted(out[best_alt][1], key=len)[:TOP_N]) if best_alt else "",
        **{f"n_clean_{enz}": len(out[enz][1]) for enz in ALTERNATIVES},
    )


def route_for(cls, d):
    if cls == "inference_locked":
        if d["n_clean_best_alt"] >= 2:
            return f"targeted MS after {d['best_alt_protease']} digest on listed peptides"
        return "affinity or immunoassay only (no unique peptide under any listed protease)"
    return ROUTE.get(cls, "")


pa = pd.read_csv(TAB_DIR / "panel_audit.tsv", sep="\t")
nom_path = TAB_DIR / "nominated_audit.tsv"
nom = pd.read_csv(nom_path, sep="\t") if nom_path.exists() else pd.DataFrame()
targets = pd.concat([pa[["uniprot", "gene", "ms_class"]].assign(source="panel"),
                     nom[["uniprot", "gene", "ms_class"]].assign(source="nominated") if len(nom) else pd.DataFrame()],
                    ignore_index=True).dropna(subset=["uniprot"])
targets["gene"] = targets["gene"].where(targets["gene"].notna() & (targets["gene"].astype(str) != "nan"), targets["uniprot"])
targets["source"] = targets.groupby("uniprot")["source"].transform(lambda s: "+".join(sorted(set(s))))
targets = targets.drop_duplicates("uniprot").reset_index(drop=True)
print(f"proteins to route: {len(targets):,}")

df, accs, genes, seqs = load_proteome()
print("building uniqueness indices for trypsin and alternatives")
idx = build_indices(genes, seqs)
by_acc = {a: i for i, a in enumerate(accs)}
rows = []
for r in targets.itertuples(index=False):
    i = by_acc.get(r.uniprot)
    if i is None:
        rows.append(dict(uniprot=r.uniprot, gene=r.gene, ms_class=r.ms_class, source=r.source,
                         route=ROUTE["unscored"]))
        continue
    d = describe(accs[i], genes[i], seqs[i], df.iloc[i], idx)
    d.update(uniprot=r.uniprot, gene=r.gene, ms_class=r.ms_class, source=r.source, route=route_for(r.ms_class, d))
    rows.append(d)
t = pd.DataFrame(rows)
lead = ["uniprot", "gene", "source", "ms_class", "route", "n_unique_tryptic", "n_clean_tryptic",
        "clean_tryptic_peptides", "unique_tryptic_peptides", "n_shared_genes", "shared_with",
        "best_alt_protease", "n_clean_best_alt", "best_alt_peptides"]
t = t[lead + [c for c in t.columns if c not in lead]]
t.to_csv(TAB_DIR / "validation_routes.tsv", sep="\t", index=False)

L = ["Validation: what experiment could confirm each protein, with the sequences it needs", ""]
for src in ("panel", "nominated"):
    s = t[t["source"].str.contains(src)]
    L.append(f"[{src}] proteins {len(s):,}")
    for cls in ["confirmable", "chemistry_limited", "inference_limited", "inference_locked", "unscored"]:
        c = s[s["ms_class"] == cls]
        if len(c):
            line = f"{cls:18s} {len(c):6,}  route: {ROUTE.get(cls, '')}"
            if cls == "inference_locked":
                resc = int((c["n_clean_best_alt"] >= 2).sum())
                line += f"   [in silico: {resc} gain >=2 clean unique peptides under an alternative protease, {len(c) - resc} affinity-only]"
            L.append(line)
    L.append("")
if len(nom):
    L.append("nominated proteins in the not-confirmable classes, with their route and peptides:")
    n2 = t[t["source"].str.contains("nominated") & t["ms_class"].isin(["chemistry_limited", "inference_limited", "inference_locked"])]
    n2 = n2.merge(nom[["uniprot", "fluid", "group"]].drop_duplicates("uniprot"), on="uniprot", how="left")
    for r in n2.sort_values(["ms_class", "gene"]).itertuples(index=False):
        peps = next((x for x in (r.clean_tryptic_peptides, r.unique_tryptic_peptides, r.best_alt_peptides)
                     if isinstance(x, str) and x), "")
        extra = f"shared with {r.shared_with}" if r.ms_class == "inference_locked" and isinstance(r.shared_with, str) and r.shared_with else ""
        L.append(f"{str(r.gene):10s} {str(r.fluid):6s} {str(r.group):13s} {r.ms_class:18s} {r.route}")
        L.append(f"{'peptides: ' + peps if peps else 'no unique peptide'}{extra}")
    L.append("")
text = "\n".join(L)
(TAB_DIR / "validation_routes.txt").write_text(text)
print(text[:4000])
