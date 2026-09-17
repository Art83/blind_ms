import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from config import DATA_DIR, TAB_DIR
import statsmodels.api as sm


PEP_COLS = ["pep_n_unique", "pep_frac_shared", "pep_max_shared_genes", "pep_n_clean"]
FEAT_COLS = ["uniprot", "tryptic_per_kda", "n_tryptic_7_30", "mol_weight_kDa", "n_tm",
             "is_secreted", "has_signal", "is_membrane", "disorder_fraction_proxy"]

MIN_CLEAN = 3
ORDER = ["confirmable", "chemistry_limited", "inference_limited", "inference_locked", "unscored"]


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
        L.append(f"{label}: too few to fit (n={len(dd)}, pos={int(dd[y].sum()) if len(dd) else 0})")
        return None
    X = sm.add_constant(pd.DataFrame({p: z(dd[p]) for p in preds}))
    try:
        m = sm.Logit(dd[y].values, X).fit(disp=0, maxiter=200)
        if not m.mle_retvals.get("converged", True):
            m = sm.Logit(dd[y].values, X).fit(disp=0, maxiter=500, method="bfgs")
    except Exception as e:
        L.append(f"{label}: fit failed ({e.__class__.__name__})")
        return None
    bad = (not m.mle_retvals.get("converged", True)) or bool(np.isnan(m.pvalues[preds]).any())
    flag = "didnt convert or separated" if bad else ""
    parts = "  ".join(f"{p} b={m.params[p]:+.3f} p={m.pvalues[p]:.1e}" for p in preds)
    L.append(f"{label} (n={len(dd):,}, pos={int(dd[y].sum()):,}): {parts}{flag}")
    return m


def build_resolver():
    fp = (pd.read_csv(TAB_DIR / "features_protein.tsv", sep="\t", usecols=["ensg", "uniprot"])
          .dropna().drop_duplicates("uniprot"))
    uni2ensg = dict(zip(fp["uniprot"], fp["ensg"]))
    gd = (pd.read_csv(TAB_DIR / "gene_dict.tsv", sep="\t", dtype=str)
          .dropna(subset=["symbol", "ensg"]).drop_duplicates("symbol"))
    sym2ensg = dict(zip(gd["symbol"], gd["ensg"]))
    us = pd.read_csv(DATA_DIR / "uniprot_human.tsv", sep="\t",
                     usecols=["Entry", "Gene Names (primary)", "Ensembl"])
    acc2sym = dict(zip(us["Entry"], us["Gene Names (primary)"]))
    acc2enst = {}
    for acc, ens in zip(us["Entry"], us["Ensembl"].fillna("")):
        acc2enst[acc] = [t.strip().split(" ")[0].split(".")[0] for t in ens.split(";") if t.strip().startswith("ENST")]
    gi = pd.read_csv(DATA_DIR / "Gene_info.txt", sep="\t", usecols=["Transcript ID", "Gene ID"], dtype=str)
    enst2ensg = dict(zip(gi["Transcript ID"].str.split(".").str[0], gi["Gene ID"].str.split(".").str[0]))

    def resolve(uniprot=None, gene=None):
        if isinstance(uniprot, str) and uniprot:
            for tok in uniprot.replace("|", ";").replace(",", ";").split(";"):
                acc = tok.split("-")[0].strip()
                if acc in uni2ensg:
                    return uni2ensg[acc]
                for t in acc2enst.get(acc, ()):
                    if t in enst2ensg:
                        return enst2ensg[t]
                s = acc2sym.get(acc)
                if isinstance(s, str) and s in sym2ensg:
                    return sym2ensg[s]
        if isinstance(gene, str) and gene:
            g = gene.split(";")[0].strip()
            if g in sym2ensg:
                return sym2ensg[g]
        return None
    return resolve


def read_adat(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [l.rstrip("\r\n") for l in fh]
    tb = next(i for i, l in enumerate(lines) if l.startswith("^TABLE_BEGIN"))
    rows = [l.split("\t") for l in lines[tb + 1:] if l != ""]
    col_meta = {}
    i = 0
    while rows[i][0] == "":
        r = rows[i]
        k = next(j for j, c in enumerate(r) if c != "")
        col_meta[r[k]] = r[k + 1:]
        i += 1
    n_rowmeta = k
    header = rows[i][:n_rowmeta]
    data = rows[i + 1:]
    row_meta = pd.DataFrame([r[:n_rowmeta] for r in data], columns=header)
    vals = pd.DataFrame([r[n_rowmeta + 1:] for r in data]).apply(pd.to_numeric, errors="coerce")
    vals.columns = col_meta["SeqId"]
    col_meta = pd.DataFrame(col_meta)
    return vals, row_meta, col_meta