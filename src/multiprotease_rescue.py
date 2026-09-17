"""
Question: of MS-dark, IHC-present genes that are trypsin-limited, how many
gain observable peptides under an alternative protease and is
the non-rescue explained by abundance / size rather than by peptide yield?
"""
import re
from collections import Counter, defaultdict


import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from config import DATA_DIR, TAB_DIR

# --- paths ---
PG_FILE = DATA_DIR / "proteinGroups.txt"
PEP_FILE = DATA_DIR / "peptides.txt"
ENSP2ENSG = DATA_DIR / "ensp2ensg.txt"
UNIPROT = DATA_DIR / "uniprot_human.tsv"
MODEL = TAB_DIR / "model_table.tsv"
FEAT = TAB_DIR / "features_protein.tsv"
GENEDICT = TAB_DIR / "gene_dict.tsv"
OUT = TAB_DIR / "multiprotease_rescue.tsv"


PROTEASES = ["Trypsin", "LysC", "LysN", "AspN", "GluC", "Chymotrypsin"]
ALT = [p for p in PROTEASES if p != "Trypsin"]
COUNT_STEM = "Razor + unique peptides"
MIN_PEP = 2
VOTE_MIN = 0.60

pg = pd.read_csv(PG_FILE, sep="\t", low_memory=False)
pg = pg[(pg.get("Reverse") != "+") & (pg.get("Potential contaminant") != "+")].copy()


def protease_max(df, protease):
    pat = re.compile(rf"^{re.escape(COUNT_STEM)} .*_{protease}_(ETD|HCD|CAD)$")
    cols = [c for c in df.columns if pat.match(c)]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)


for p in PROTEASES:
    pg[f"n_{p}"] = protease_max(pg, p)

fp = pd.read_csv(FEAT, sep="\t")[["ensg", "uniprot"]].dropna().drop_duplicates("uniprot")
uni2ensg = dict(zip(fp["uniprot"], fp["ensg"]))

e2 = pd.read_csv(ENSP2ENSG, sep="\t")
gcol = next(c for c in e2.columns if "Gene" in c)
pcol = next(c for c in e2.columns if "Protein" in c)
e2 = e2.dropna(subset=[pcol])
ensp2ensg = dict(zip(e2[pcol].astype(str).str.split(".").str[0], e2[gcol]))

gsym = pd.read_csv(GENEDICT, sep="\t").dropna(subset=["symbol", "ensg"]).drop_duplicates("symbol")
sym2ensg = dict(zip(gsym["symbol"], gsym["ensg"]))

usym = pd.read_csv(UNIPROT, sep="\t", usecols=["Entry", "Gene Names (primary)"]).dropna()
acc2sym = dict(zip(usym["Entry"], usym["Gene Names (primary)"]))


def token_to_ensg(tok):
    tok = tok.strip()
    if not tok:
        return None
    if tok.startswith("ENSG"):
        return tok.split(".")[0]
    if tok.startswith("ENSP"):
        return ensp2ensg.get(tok.split(".")[0])
    if tok.startswith("GENSCAN"):
        return None
    acc = tok.split("-")[0]
    hit = uni2ensg.get(acc)
    if hit:
        return hit
    sym = acc2sym.get(acc)
    return sym2ensg.get(sym) if sym else None


def map_group(maj_ids, gene_names):
    for tok in str(maj_ids).split(";"):
        hit = token_to_ensg(tok)
        if hit:
            return hit
    for gn in str(gene_names).split(";"):
        if gn.strip() in sym2ensg:
            return sym2ensg[gn.strip()]
    return None


gene_names = pg["Gene names"] if "Gene names" in pg.columns else pd.Series("", index=pg.index)
pg["ensg"] = [map_group(mi, gn) for mi, gn in zip(pg["Majority protein IDs"], gene_names)]
n_route1 = int(pg["ensg"].notna().sum())


unmapped_ids = set(pg.loc[pg["ensg"].isna(), "id"].astype(str))

pep = pd.read_csv(PEP_FILE, sep="\t", low_memory=False,
                  usecols=["Proteins", "Protein group IDs"]).dropna(subset=["Protein group IDs"])
pe = pep.assign(gid=pep["Protein group IDs"].astype(str).str.split(";")).explode("gid")
pe["gid"] = pe["gid"].str.strip()
pe = pe[pe["gid"].isin(unmapped_ids)]


_cache = {}


def proteins_to_ensgs(s):
    if s in _cache:
        return _cache[s]
    out = set()
    for tok in str(s).split(";"):
        e = token_to_ensg(tok)
        if e:
            out.add(e)
    _cache[s] = out
    return out


votes = defaultdict(Counter)
for gid, proteins in zip(pe["gid"], pe["Proteins"].fillna("")):
    for e in proteins_to_ensgs(proteins):
        votes[gid][e] += 1

assigned, ambiguous = {}, 0
for gid, c in votes.items():
    total = sum(c.values())
    ensg, top = c.most_common(1)[0]
    if total and top / total >= VOTE_MIN:
        assigned[gid] = ensg
    else:
        ambiguous += 1

pg["ensg"] = pg["ensg"].fillna(pg["id"].astype(str).map(assigned))
n_route2 = int(pg["ensg"].notna().sum())


n_groups = len(pg)
still_unmap = pg["ensg"].isna().sum()
print(f"protein groups (filtered): {n_groups}")
print(f"mapped by own accessions (route 1): {n_route1} ({n_route1/n_groups*100:.0f}%)")
print(f" + recovered by peptide evidence (route 2): {n_route2-n_route1}")
print(f"= mapped total: {n_route2} ({n_route2/n_groups*100:.0f}%)")
print(f"left unmapped: {still_unmap}  (of which {ambiguous} peptide-ambiguous, rest no evidence)")

g = (pg.dropna(subset=["ensg"])
       .groupby("ensg")[[f"n_{p}" for p in PROTEASES]]
       .max()
       .reset_index())

m = pd.read_csv(MODEL, sep="\t")
gene_det = m.groupby("ensg")["gtex_detected"].max()
dark = set(gene_det[gene_det == 0].index)
all_genes = set(gene_det.index)

cov = (m.groupby("ensg")
         .agg(log_abundance_peak=("log_abundance", "max"),
              log_abundance_med=("log_abundance", "median"),
              n_tm=("n_tm", "max"),
              log_mw=("log_mw", "first"),
              log_tryptic=("log_tryptic", "first"),
              length=("length", "first"),
              is_membrane=("is_membrane", "max"))
         .reset_index())

g = g.merge(cov, on="ensg", how="left")
g["ms_dark"] = g["ensg"].isin(dark)

tested = g[g["ensg"].isin(all_genes)]
print(f"\nMS-dark-everywhere genes: {len(dark)}")
print(f"present + ID-mapped in the 6 cell lines: {int(tested['ms_dark'].sum())}")


d = g[g["ms_dark"]].copy()
d["trypsin_limited"] = d["n_Trypsin"] < MIN_PEP
lim = d[d["trypsin_limited"]].copy()
lim["rescued_any"] = (lim[[f"n_{p}" for p in ALT]] >= MIN_PEP).any(axis=1)

n_lim, n_res = len(lim), int(lim["rescued_any"].sum())
print(f"\ntrypsin-limited dark genes (n_Trypsin<{MIN_PEP}): {n_lim}")
print(f"rescued by >=1 alternative protease: {n_res} ({n_res/max(n_lim,1)*100:.0f}%)")
print("per-protease rescue of the trypsin-limited set:")
for p in ALT:
    print(f"{p:12s} {(lim[f'n_{p}'] >= MIN_PEP).mean()*100:5.1f}%")


def compare(col, label):
    a = lim.loc[lim["rescued_any"], col].dropna()
    b = lim.loc[~lim["rescued_any"], col].dropna()
    if len(a) < 5 or len(b) < 5:
        print(f"{label:16s} too few to test (n={len(a)},{len(b)})")
        return
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    print(f"{label:16s} rescued {a.median():8.2f} (n={len(a)}) | "
          f"non-rescued {b.median():8.2f} (n={len(b)}) | p={p:.3g}")


print("\nrescued vs non-rescued within the trypsin-limited set:")
compare("log_abundance_peak", "abundance(peak)")
compare("log_abundance_med", "abundance(med)")
compare("log_mw", "log_mw")
compare("length", "length")
compare("log_tryptic", "log_tryptic")
compare("n_tm", "n_tm")


lim["membrane"] = lim["is_membrane"] >= 1
ct = pd.crosstab(lim["membrane"], lim["rescued_any"])
print("\nmembrane (rows) vs rescued (cols):")
print(ct)
if ct.shape == (2, 2):
    _, pf = fisher_exact(ct)
    print(f"Fisher exact p={pf:.3g}")

mem = lim[lim["membrane"]]
if len(mem):
    print(f"\nchymotrypsin rescue within membrane trypsin-limited "
          f"(n={len(mem)}): {(mem['n_Chymotrypsin'] >= MIN_PEP).mean()*100:.0f}%")

lim.to_csv(OUT, sep="\t", index=False)

