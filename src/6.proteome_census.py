"""
Full-proteome run + missing-proteome structural partition.

Extends the detectability finding from the labelled (IHC-present) universe to all
reviewed human proteins
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import re
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
import pandas as pd
import uniprot_annot as UA
from utils_features import AA_MASS, _mol_weight, _tryptic_count, _pI, _gravy, _low_complexity_fraction, _boot_auc, _boot_diff, _oof_auc
from config import DATA_DIR, TAB_DIR

PE_FILE = DATA_DIR / "uniprot_human.tsv"
PE_FILE_OLD = DATA_DIR / "uniprotkb_pe.tsv"
STD = set(AA_MASS)


def clean_seq(s):
    return re.sub(r"[^A-Z]", "", str(s).upper())


def gene_label(out):
    mt = pd.read_csv(out / "model_table.tsv", sep="\t")
    g = mt.groupby("ensg")["gtex_detected"].mean().reset_index()
    g["y"] = (g["gtex_detected"] > 0).astype(int)
    return g[["ensg", "y"]]


def seq_features(seq, m_start=1, m_end=None):
    m_end = m_end or len(seq)
    mat = seq[m_start - 1:m_end] or seq
    mw = _mol_weight(seq)
    nt = _tryptic_count(seq, start=m_start, end=m_end)
    L = len(mat) or 1
    return dict(length=len(seq), mw_da=mw, gravy=_gravy(mat), pI=_pI(mat),
                n_tryptic_7_30=nt, tryptic_per_kda=nt / (mw / 1000.0) if mw else 0.0,
                cysteine_fraction=mat.count("C") / L,
                charged_fraction=sum(mat.count(a) for a in "DEKR") / L,
                proline_fraction=mat.count("P") / L,
                glycine_fraction=mat.count("G") / L,
                aromatic_fraction=sum(mat.count(a) for a in "FYW") / L,
                n_glyco_motif_count=len(re.findall(r"(?=N[^P][ST])", mat)),
                low_complexity_fraction=_low_complexity_fraction(mat))


PE_MAP = {"Evidence at protein level": 1, "Evidence at transcript level": 2,
          "Inferred from homology": 3, "Predicted": 4, "Uncertain": 5}
SEQ_FEATS = ["length", "mw_da", "gravy", "pI", "n_tryptic_7_30", "tryptic_per_kda",
             "cysteine_fraction", "charged_fraction", "proline_fraction",
             "glycine_fraction", "aromatic_fraction", "n_glyco_motif_count",
             "low_complexity_fraction"]
PEP_FEATS = ["pep_frac_unique", "pep_frac_shared", "pep_max_shared_genes", "pep_n_unique",
             "pep_n_clean", "pep_clean_per_kda", "pep_coverage_clean", "pep_frac_met",
             "pep_frac_cys", "pep_frac_sequon", "pep_frac_slow_cleave", "pep_gravy_mean",
             "pep_frac_hydrophilic", "pep_frac_hydrophobic", "pep_max_gap", "pep_coverage_7_30"]


pe = pd.read_csv(PE_FILE, sep="\t", dtype=str, keep_default_na=False)
pe.columns = [c.strip() for c in pe.columns]
if not any("existence" in c.lower() for c in pe.columns):
    print(f"{PE_FILE.name} has no PE column, falling back to {PE_FILE_OLD.name}")
    pe = pd.read_csv(PE_FILE_OLD, sep="\t", dtype=str, keep_default_na=False)
    pe.columns = [c.strip() for c in pe.columns]
entry_col = [c for c in pe.columns if c.lower() == "entry"][0]
seq_col = [c for c in pe.columns if c.lower() == "sequence"][0]
peex_col = [c for c in pe.columns if "existence" in c.lower()][0]
pe = pe.rename(columns={entry_col: "uniprot", seq_col: "seq", peex_col: "pe_text"})
n_raw = len(pe)
pe["seq"] = pe["seq"].map(clean_seq)
pe = pe[pe["seq"].str.len() > 0].copy()
pe["pe"] = pe["pe_text"].map(PE_MAP)
nonstd = pe["seq"].map(lambda s: any(a not in STD for a in s)).sum()
maxlen = pe["seq"].str.len().max()

have_chain = all(c in pe.columns for c in ("Chain", "Propeptide"))
print(f"mature-chain annotation {'present: chemistry on the mature chain' if have_chain else 'absent: chemistry on the precursor'}")
rows = []
for i, s_ in zip(pe.index, pe["seq"]):
    m_start, m_end = UA.mature_range(pe.loc[i].to_dict(), len(s_))
    rows.append(seq_features(s_, m_start, m_end))
feat = pd.DataFrame(rows, index=pe.index)
pe = pd.concat([pe[["uniprot", "pe", "pe_text", "seq"]], feat], axis=1)

pep_path = TAB_DIR / "features_peptides.tsv"
have_pep = pep_path.exists()
if have_pep:
    pep = pd.read_csv(pep_path, sep="\t")
    pe = pe.merge(pep[["uniprot"] + PEP_FEATS], on="uniprot", how="left")
    pep_cov = pe[PEP_FEATS[0]].notna().mean()
    print(f"peptide-level features merged on accession: coverage {pep_cov:.1%}")
else:
    print("features_peptides.tsv not found")
FEATS = SEQ_FEATS + (PEP_FEATS if have_pep else [])

fp = pd.read_csv(TAB_DIR / "features_protein.tsv", sep="\t")[["ensg", "uniprot"]].dropna()
lab = gene_label(TAB_DIR)
bridge = fp.merge(lab, on="ensg", how="inner").drop_duplicates("uniprot")
pe = pe.merge(bridge[["uniprot", "ensg", "y"]], on="uniprot", how="left")

train = pe.dropna(subset=["y"]).copy()
Xtr = train[FEATS].astype(float).values
ytr = train["y"].astype(int).values
grp = train["ensg"].values
# group by gene so isoforms of one ENSG cannot straddle train/test (leakage)
oof = _oof_auc(Xtr, ytr, grp)
auc, lo, hi = _boot_auc(ytr, oof)
if have_pep:
    oof0 = _oof_auc(train[SEQ_FEATS].astype(float).values, ytr, grp)
    auc0, lo0, hi0 = _boot_auc(ytr, oof0)
    gain = _boot_diff(ytr, oof, oof0)
# comparator: the full intrinsic model's OOF AUC on the same genes, if present
comp = None
mlp = TAB_DIR / "ml_per_gene_predictions.tsv"
if mlp.exists():
    ml = pd.read_csv(mlp, sep="\t").drop_duplicates("ensg")[["ensg", "y", "p_detect_intrinsic"]]
    ml = ml[ml["ensg"].isin(train["ensg"])]
    if len(ml) and ml["y"].nunique() > 1:
        comp = (roc_auc_score(ml["y"], ml["p_detect_intrinsic"]), len(ml))

gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                     max_depth=4, random_state=0).fit(Xtr, ytr)
pe["p_seq"] = gbm.predict_proba(pe[FEATS].astype(float).values)[:, 1]
pe.drop(columns=["seq"]).to_csv(TAB_DIR / "proteome_census.tsv", sep="\t", index=False)

L = ["FULL-PROTEOME + MISSING-PROTEOME STRUCTURAL PARTITION", ""]
L += [f"proteins: {n_raw:,} read, {len(pe):,} with sequence; "
      f"non-standard-residue proteins {int(nonstd):,}; max length {int(maxlen):,}",
      "PE distribution: " + ", ".join(
          f"PE{k}:{int((pe['pe']==k).sum()):,}" for k in range(1, 6)),
      f"labelled (in GTEx-detection universe): {len(train):,}  "
      f"detected rate {ytr.mean():.3f}", ""]
L += ["sequence-only scorer (price of full-proteome coverage):",
      f"  5-fold OOF AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]"
      + (f"  (full intrinsic model on the same {comp[1]:,} genes: {comp[0]:.3f})" if comp else "")]
if have_pep:
    L.append(f" without peptide-level block {auc0:.3f} [{lo0:.3f}, {hi0:.3f}]   "
             f"gain {gain[0]:+.3f} [{gain[1]:+.3f}, {gain[2]:+.3f}]")
L.append("")

L.append("median predicted detectability P(detect) by PE tier:")
pe_names = list(PE_MAP.keys())
for k in range(1, 6):
    m = pe["pe"] == k
    if m.sum():
        L.append(f"  PE{k} {pe_names[k-1]:28s} "
                 f"n={int(m.sum()):5,}  median P {pe.loc[m,'p_seq'].median():.3f}  "
                 f"median fragments {pe.loc[m,'n_tryptic_7_30'].median():.0f}")
L.append("")

pe1 = pe[pe["pe"] == 1]
miss = pe[pe["pe"].isin([2, 3, 4])]
q25 = pe1["p_seq"].quantile(0.25)
hard = (miss["p_seq"] < q25).mean()
u, pu = mannwhitneyu(miss["p_seq"], pe1["p_seq"], alternative="less")
L += ["missing proteome (PE2-4; PE5 quarantined separately):",
      f"  n PE2-4 = {len(miss):,}",
      f"  median P(detect):  PE2-4 {miss['p_seq'].median():.3f}  vs  PE1 {pe1['p_seq'].median():.3f}",
      f"  fraction of PE2-4 below PE1's 25th-pctile detectability "
      f"(structurally hard): {hard:.1%}",
      f"  Mann-Whitney PE2-4 < PE1 (one-sided): p={pu:.2e}",
      f"  PE5 (uncertain, quarantined): n={int((pe['pe']==5).sum()):,}  "
      f"median P {pe.loc[pe['pe']==5,'p_seq'].median():.3f}"]

if have_pep:
    L.append("")
    L.append("inference-dark partition (no unique tryptic peptide in the 7-30 aa window):")
    for lab, sub in [("PE1", pe1), ("PE2-4", miss), ("PE5", pe[pe["pe"] == 5])]:
        z = sub["pep_n_unique"] == 0
        sh = sub["pep_frac_shared"] >= 0.5
        L.append(f"  {lab:6s} n={len(sub):6,}  zero unique peptides {z.mean():5.1%}   "
                 f">=50% peptides shared {sh.mean():5.1%}   "
                 f"median unique peptides {sub['pep_n_unique'].median():.0f}")

both = pe[pe["pe"].isin([1, 2, 3, 4])].copy()
both["missing"] = both["pe"].isin([2, 3, 4])
both["lenband"] = pd.qcut(both["length"], 5, labels=[f"L{i}" for i in range(1, 6)])
L.append("")
L.append("size-confound check: median P(detect), PE1 vs PE2-4 within length bands:")
for lb in [f"L{i}" for i in range(1, 6)]:
    b = both[both["lenband"] == lb]
    p1 = b[~b["missing"]]; pm = b[b["missing"]]
    lo_len, hi_len = int(b["length"].min()), int(b["length"].max())
    if len(pm) >= 10:
        L.append(f"  {lb} ({lo_len:5,}-{hi_len:5,} aa): PE1 {p1['p_seq'].median():.3f}  "
                 f"PE2-4 {pm['p_seq'].median():.3f}  gap {p1['p_seq'].median()-pm['p_seq'].median():+.3f}  "
                 f"(n PE2-4={len(pm)})")
    else:
        L.append(f"  {lb} ({lo_len:5,}-{hi_len:5,} aa): PE2-4 n={len(pm)} too few")

if PE_FILE_OLD.exists() and PE_FILE_OLD != PE_FILE:
    old = pd.read_csv(PE_FILE_OLD, sep="\t", dtype=str, keep_default_na=False)
    old.columns = [c.strip() for c in old.columns]
    oc = [c for c in old.columns if "existence" in c.lower()]
    ec = [c for c in old.columns if c.lower() == "entry"]
    if oc and ec:
        old = old.rename(columns={ec[0]: "uniprot", oc[0]: "pe_text_old"})
        old["pe_old"] = old["pe_text_old"].map(PE_MAP)
        j = pe.merge(old[["uniprot", "pe_old"]], on="uniprot", how="inner")
        was_missing = j[j["pe_old"].isin([2, 3, 4])]
        promoted = was_missing[was_missing["pe"] == 1]
        stayed = was_missing[was_missing["pe"].isin([2, 3, 4])]
        L.append("")
        L.append("prospective check: PE2-4 in the previous export, PE1 now")
        L.append(f"previously missing {len(was_missing):,}: promoted to PE1 {len(promoted):,}, still missing {len(stayed):,}")
        if len(promoted) >= 5 and len(stayed) >= 5:
            pv = mannwhitneyu(promoted["p_seq"], stayed["p_seq"], alternative="greater").pvalue
            L.append(f" median P(detect): promoted {promoted['p_seq'].median():.3f}  still missing {stayed['p_seq'].median():.3f}  "
                     f"(one-sided Mann-Whitney promoted > still missing: p={pv:.2e})")
            L.append(f"zero unique peptides: promoted {(promoted['pep_n_unique'] == 0).mean():.1%}  "
                     f"still missing {(stayed['pep_n_unique'] == 0).mean():.1%}")
            L.append("the scorer sees sequence only, so this is a forward test at the resolution")
            L.append("one UniProt release cycle allows. Promotions can come from non-MS evidence.")

text = "\n".join(L)
(TAB_DIR / "proteome_census.txt").write_text(text)
print(text)
