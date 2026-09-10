"""
Peptide level features from an in-silico tryptic digest of the
whole reviewed human proteome.

The existing feature table (features_protein.tsv) describes chemistry at the protein level and
counts tryptic peptides in the observable length window.
That representation is blind to two things:
  - whether a peptide is unique to one gene
  - whether instrument can "see" each peptide
"""
from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import uniprot_annot as UA
from config import DATA_DIR, TAB_DIR
from utils_features import WIN, MZ_WIN, PROTON, SEQUON, GRAVY_HYDROPHILIC, GRAVY_HYDROPHOBIC, _digest, _pep_mass, \
    _slow_cleave, _gravy

UNIPROT_TSV = DATA_DIR / "uniprot_human.tsv"

df = pd.read_csv(UNIPROT_TSV, sep="\t", dtype=str, keep_default_na=False)
acc = df["Entry"].str.strip()
gene = df["Gene Names (primary)"].str.strip().str.split("[;\\s]+", regex=True).str[0].fillna("")
gene = gene.where(gene != "", acc)
seqs = df["Sequence"].str.strip().str.upper()
have_chain = all(c in df.columns for c in ("Chain", "Propeptide"))
have_ptm = all(c in df.columns for c in ("Glycosylation", "Modified residue", "Disulfide bond", "Lipidation"))

print(f"mature-chain annotation {'present' if have_chain else 'absent: precursor digest'}, "
      f"PTM annotation {'present' if have_ptm else 'absent: no PTM flags'}")

pep_genes = defaultdict(set)
per_protein = []
for i, (a, g, s) in enumerate(zip(acc, gene, seqs)):
    rd = df.iloc[i].to_dict()
    m_start, m_end = UA.mature_range(rd, len(s)) if have_chain else (1, len(s))
    sites = UA.site_positions(rd) if have_ptm else dict(glyco=set(), mod=set(), lipid=set(), disulfide=set())
    allp = [(p, st, en) for p, st, en in _digest(s) if WIN[0] <= len(p) <= WIN[1]]
    for p, _, _ in allp:
        pep_genes[p.replace("I", "L")].add(g)
    peps = [(p, st, en) for p, st, en in allp if st + 1 >= m_start and en <= m_end]
    per_protein.append((a, g, s, peps, sites, (m_start, m_end)))
print(f"digested {len(per_protein):,} entries, {len(pep_genes):,} distinct peptides")

# per-peptide flags and per-protein aggregates
rows = []
for a, g, s, peps, sites, (m_start, m_end) in per_protein:
    n = len(peps)
    L = len(s)
    ptm_pos = sites["glyco"] | sites["lipid"]
    ss_pos = sites["disulfide"]
    kda = _pep_mass(s) / 1000.0 if L else float("nan")
    sites = [m.start() for m in re.finditer(r"[KR](?!P)", s)]
    bounds = [0] + [i + 1 for i in sites] + [L]
    max_gap = max((b - a_ for a_, b in zip(bounds[:-1], bounds[1:])), default=L)
    if n == 0:
        rows.append(dict(uniprot=a, pep_n_7_30=0, pep_n_unique=0, pep_frac_unique=np.nan,
                         pep_n_unique_mz=0, pep_n_clean=0, pep_clean_per_kda=0.0,
                         pep_frac_met=np.nan, pep_frac_cys=np.nan, pep_frac_sequon=np.nan,
                         pep_frac_slow_cleave=np.nan, pep_frac_ptm_site=np.nan,
                         pep_frac_disulfide=np.nan, pep_gravy_mean=np.nan,
                         pep_frac_hydrophilic=np.nan, pep_frac_hydrophobic=np.nan,
                         pep_max_shared_genes=0, pep_frac_shared=np.nan,
                         pep_max_gap=max_gap, pep_coverage_7_30=0.0, pep_coverage_clean=0.0))
        continue
    uniq, mz_ok, met, cys, seqon, slow, ptm, ssb = (np.zeros(n, bool) for _ in range(8))
    gr = np.zeros(n)
    shared = np.zeros(n, int)
    cov, cov_clean = np.zeros(L, bool), np.zeros(L, bool)
    for i, (p, st, en) in enumerate(peps):
        others = len(pep_genes[p.replace("I", "L")]) - 1
        shared[i] = others
        uniq[i] = others == 0
        m = _pep_mass(p)
        mz_ok[i] = any(MZ_WIN[0] <= (m + z * PROTON) / z <= MZ_WIN[1] for z in (2, 3))
        met[i] = "M" in p
        cys[i] = "C" in p
        seqon[i] = bool(SEQUON.search(p))
        slow[i] = _slow_cleave(s, st, en)
        gr[i] = _gravy(p)
        cov[st:en] = True
        span = range(st + 1, en + 1)  # 1-based precursor positions
        ptm[i] = any(q in ptm_pos for q in span)
        ssb[i] = any(q in ss_pos for q in span)
    clean = uniq & mz_ok & ~met & ~cys & ~seqon & ~ptm
    for i, (p, st, en) in enumerate(peps):
        if clean[i]:
            cov_clean[st:en] = True
    rows.append(dict(
        uniprot=a,
        pep_n_7_30=n,
        pep_n_unique=int(uniq.sum()),
        pep_frac_unique=float(uniq.mean()),
        pep_n_unique_mz=int((uniq & mz_ok).sum()),
        pep_n_clean=int(clean.sum()),
        pep_clean_per_kda=float(clean.sum() / kda) if kda else np.nan,
        pep_frac_met=float(met.mean()),
        pep_frac_cys=float(cys.mean()),
        pep_frac_sequon=float(seqon.mean()),
        pep_frac_slow_cleave=float(slow.mean()),
        pep_frac_ptm_site=float(ptm.mean()),
        pep_frac_disulfide=float(ssb.mean()),
        pep_gravy_mean=float(np.nanmean(gr)),
        pep_frac_hydrophilic=float((gr < GRAVY_HYDROPHILIC).mean()),
        pep_frac_hydrophobic=float((gr > GRAVY_HYDROPHOBIC).mean()),
        pep_max_shared_genes=int(shared.max()),
        pep_frac_shared=float((shared > 0).mean()),
        pep_max_gap=int(max_gap),
        pep_coverage_7_30=float(cov.mean()),
        pep_coverage_clean=float(cov_clean.mean()),
    ))
feats = pd.DataFrame(rows)
feats.to_csv(TAB_DIR / "features_peptides.tsv", sep="\t", index=False)
print(f"features_peptides: {len(feats):,} accessions")
z = feats[feats["pep_n_7_30"] > 0]
print(f"median window peptides {z['pep_n_7_30'].median():.0f}, unique {z['pep_n_unique'].median():.0f}, "
      f"clean {z['pep_n_clean'].median():.0f}")
print(f"proteins with zero unique window peptides: {int((feats['pep_n_unique'] == 0).sum()):,}  "
      f"zero clean: {int((feats['pep_n_clean'] == 0).sum()):,}")
print(f"median frac shared with another gene {z['pep_frac_shared'].median():.3f}, "
      f"frac with any shared peptide {(z['pep_frac_shared'] > 0).mean():.1%}")
# cross-check
fp = TAB_DIR / "features_protein.tsv"
old = pd.read_csv(fp, sep="\t", usecols=["uniprot", "n_tryptic_7_30"])
m = old.merge(feats[["uniprot", "pep_n_7_30"]], on="uniprot")
print(f"check vs features_protein n_tryptic_7_30: {int((m['n_tryptic_7_30'] == m['pep_n_7_30']).sum()):,} "
      f"/ {len(m):,} identical")
