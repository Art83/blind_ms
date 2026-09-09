"""
Another utils file but only with ML features related functions. Utils gets a bit crowded.
"""
import re
from utils import symbol_to_ensg
import uniprot_annot as UA
from config import ISOFORM_PICK

def _ensg_from_xref(x):
    if not isinstance(x, str):
        return []
    return re.findall(r"ENSG\d+", x)


# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# pK set for pI
PK_POS = {"Nterm": 8.6, "K": 10.8, "R": 12.5, "H": 6.5}
PK_NEG = {"Cterm": 3.6, "D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}

# Confidence from uniprot
PE_LEVEL = {"Evidence at protein level": 1, "Evidence at transcript level": 2,
            "Inferred from homology": 3, "Predicted": 4, "Uncertain": 5}


def _gravy(seq):
    vals = [KD[a] for a in seq if a in KD]
    return sum(vals) / len(vals) if vals else float("nan")


def _charge(pH, counts):
    pos = 1.0 / (1 + 10 ** (pH - PK_POS["Nterm"]))
    for aa, pk in (("K", PK_POS["K"]), ("R", PK_POS["R"]), ("H", PK_POS["H"])):
        pos += counts.get(aa, 0) / (1 + 10 ** (pH - pk))
    neg = 1.0 / (1 + 10 ** (PK_NEG["Cterm"] - pH))
    for aa, pk in (("D", PK_NEG["D"]), ("E", PK_NEG["E"]), ("C", PK_NEG["C"]), ("Y", PK_NEG["Y"])):
        neg += counts.get(aa, 0) / (1 + 10 ** (pk - pH))
    return pos - neg


def _pI(seq):
    if not seq:
        return float("nan")
    counts = {}
    for a in seq:
        counts[a] = counts.get(a, 0) + 1
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _charge(mid, counts) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def _tryptic_7_30(seq, start=1, end=None):
    if not seq:
        return 0
    end = end or len(seq)
    pos, n = 0, 0
    for frag in re.split(r"(?<=[KR])(?!P)", seq):
        if not frag:
            continue
        a, b = pos + 1, pos + len(frag)
        pos += len(frag)
        if 7 <= len(frag) <= 30 and a >= start and b <= end:
            n += 1
    return n


def build_features(path_uniprot, path_gene_dict, path_gene_features):
    import pandas as pd
    df = pd.read_csv(path_uniprot, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    gene_dict = pd.read_csv(path_gene_dict, sep="\t", dtype=str)
    # Columns for the job
    # Annotation , ensg is a nightmare, need to parse
    c_acc = "Entry"
    c_gene = "Gene Names (primary)"
    c_ens = "Ensembl"

    # features
    c_seq = "Sequence"
    c_len = "Length"
    c_mass = "Mass"
    c_tm = "Transmembrane"
    c_sig = "Signal peptide"
    c_pe = "Protein existence"
    c_cc = "Subcellular location [CC]"
    c_topo = "Topological domain"
    c_go = "Gene Ontology (cellular component)"

    sym2ensg = symbol_to_ensg(gene_dict, tag="uniprot")

    have_chain = all(c in list(df.columns) for c in ("Chain", "Propeptide"))
    have_ptm = all(c in list(df.columns) for c in ("Glycosylation", "Modified residue", "Disulfide bond", "Lipidation"))

    rows = []
    n_mapped = 0
    for _, r in df.iterrows():
        seq = (r[c_seq] or "").strip().upper()
        if not seq:
            continue
        sym = (r[c_gene] or "").strip().split()[0] if r[c_gene] else None
        ensg = sym2ensg.get(sym) if sym else None
        if ensg is None and c_ens:
            xr = _ensg_from_xref(r[c_ens])
            ensg = xr[0] if xr else None
        if ensg is None:
            continue
        n_mapped += 1
        mw = pd.to_numeric(str(r[c_mass]).replace(",", ""), errors="coerce") if c_mass else float("nan")
        length = int(pd.to_numeric(r[c_len], errors="coerce")) if c_len and r[c_len] else len(seq)
        rd = r.to_dict()
        m_start, m_end = UA.mature_range(rd, len(seq)) if have_chain else (1, len(seq))
        mature = seq[m_start - 1:m_end]
        ptm = (lambda col, key, ev: UA.count(rd.get(col, ""), key, ev)) if have_ptm else (lambda col, key, ev: 0)
        feat = dict(
            ensg=ensg,
            uniprot=r[c_acc] if c_acc else "",
            length=length,
            mw_da=mw,
            gravy=round(_gravy(mature), 4),
            pI=_pI(mature),
            n_tm=len(re.findall(r"TRANSMEM", r[c_tm])) if c_tm else 0,
            has_signal=int(bool((r[c_sig] or "").strip())) if c_sig else 0,
            n_tryptic_7_30=_tryptic_7_30(seq, m_start, m_end),
            mature_start=m_start,
            mature_end=m_end,
            mature_length=m_end - m_start + 1,
            n_glyco_sites=ptm("Glycosylation", "CARBOHYD", "predicted"),
            n_mod_res=ptm("Modified residue", "MOD_RES", "predicted"),
            n_disulfide=ptm("Disulfide bond", "DISULFID", "predicted"),
            n_lipid_sites=ptm("Lipidation", "LIPID", "predicted"),
            n_glyco_sites_observed=ptm("Glycosylation", "CARBOHYD", "observed"),
            n_mod_res_observed=ptm("Modified residue", "MOD_RES", "observed"),
            n_disulfide_observed=ptm("Disulfide bond", "DISULFID", "observed"),
            n_lipid_sites_observed=ptm("Lipidation", "LIPID", "observed"),
            has_propeptide=int(UA.count(rd.get("Propeptide", ""), "PROPEP") > 0) if have_chain else 0,
            pe_level=PE_LEVEL.get((r[c_pe] or "").strip(), float("nan")) if c_pe else float("nan"),
        )
        feat.update(UA.localisation(r[c_cc], r[c_topo] if c_topo else "", r[c_go] if c_go else ""))
        kda = (feat["mw_da"] / 1000.0) if pd.notna(feat["mw_da"]) and feat["mw_da"] else (feat["length"] * 0.11)
        feat["tryptic_per_kda"] = round(feat["n_tryptic_7_30"] / kda, 4) if kda else float("nan")
        rows.append(feat)

    feats = pd.DataFrame(rows)
    multi = feats.groupby("ensg").size()
    n_multi = int((multi > 1).sum())
    spread = feats.groupby("ensg")["n_tryptic_7_30"].agg(lambda s: s.max() - s.min())
    feats = (feats.sort_values("length", ascending=(ISOFORM_PICK == "shortest"))
             .drop_duplicates("ensg")
             .reset_index(drop=True))
    print(f"uniprot rows mapped to ensg (via symbol): {n_mapped:,} / {len(df):,}")
    print(f"ensg with >1 UniProt entry: {n_multi:,}; median tryptic spread "
          f"across isoforms {spread.median():.0f} (isoform pick = {ISOFORM_PICK})")

    gf_path = path_gene_features
    if gf_path.exists():
        gf = pd.read_csv(gf_path).rename(columns={"Gene stable ID": "ensg"})
        n_dup = int(gf["ensg"].duplicated().sum())
        if n_dup:
            flag_cols = [c for c in gf.columns if c.startswith("is_")] + \
                        [c for c in ("n_compartments",) if c in gf.columns]
            agg = {c: ("max" if c in flag_cols else "first") for c in gf.columns if c != "ensg"}
            gf = gf.groupby("ensg", as_index=False).agg(agg)
            print(f"gene_features: collapsed {n_dup} duplicate ensg rows "
                  f"(localisation flags unioned, other columns first row)")
        overlap = [c for c in gf.columns if c != "ensg" and c in feats.columns]
        if overlap:
            cmp = feats[["ensg"] + overlap].merge(gf[["ensg"] + overlap], on="ensg", suffixes=("", "_gf"))
            diffs = {c: (int(((cmp[c] == 1) & (cmp[c + "_gf"] == 0)).sum()),
                         int(((cmp[c] == 0) & (cmp[c + "_gf"] == 1)).sum()))
                     for c in overlap if c not in ("n_compartments",)}
            print(f"gene_features columns superseded by the uniprot export ({len(overlap)}), "
                  f"gained/lost vs the old flags: "
                  + ", ".join(f"{c} +{g:,}/-{l:,}" for c, (g, l) in diffs.items())
                  + (f", n_compartments differs for {int((cmp['n_compartments'] != cmp['n_compartments_gf']).sum()):,}"
                     if "n_compartments" in overlap else ""))
            gf = gf.drop(columns=overlap)
        if "loc_source" in feats.columns:
            src = feats["loc_source"].value_counts()
            print(f"localisation source: CC line {int(src.get('cc', 0)):,}, GO fallback {int(src.get('go', 0)):,}, "
                  f"neither {int(src.get('none', 0)):,}")
        feats = feats.merge(gf, on="ensg", how="left")
        cov = feats["hydrophobicity"].notna().mean() if "hydrophobicity" in feats else float("nan")
        print(f"gene_features merged: {gf.shape[1] - 1} extra cols, "
              f"coverage over UniProt genes {cov:.1%}")
    else:
        print("gene_features.csv not found -> tryptic + UniProt features only")
    return feats


# Transcript level features
HUMAN_CODON_W = {
    # Phe
    'TTT': 0.452, 'TTC': 1.000,
    # Leu
    'TTA': 0.128, 'TTG': 0.206, 'CTT': 0.206, 'CTC': 0.317,
    'CTA': 0.115, 'CTG': 1.000,
    # Ile
    'ATT': 0.490, 'ATC': 1.000, 'ATA': 0.243,
    # Met
    'ATG': 1.000,
    # Val
    'GTT': 0.261, 'GTC': 0.419, 'GTA': 0.165, 'GTG': 1.000,
    # Ser
    'TCT': 0.621, 'TCC': 0.724, 'TCA': 0.483, 'TCG': 0.172,
    'AGT': 0.483, 'AGC': 1.000,
    # Pro
    'CCT': 0.586, 'CCC': 1.000, 'CCA': 0.552, 'CCG': 0.207,
    # Thr
    'ACT': 0.480, 'ACC': 1.000, 'ACA': 0.560, 'ACG': 0.240,
    # Ala
    'GCT': 0.519, 'GCC': 1.000, 'GCA': 0.444, 'GCG': 0.185,
    # Tyr
    'TAT': 0.434, 'TAC': 1.000,
    # Stop codons (excluded from CAI)
    'TAA': 0.0, 'TAG': 0.0, 'TGA': 0.0,
    # His
    'CAT': 0.424, 'CAC': 1.000,
    # Gln
    'CAA': 0.346, 'CAG': 1.000,
    # Asn
    'AAT': 0.465, 'AAC': 1.000,
    # Lys
    'AAA': 0.432, 'AAG': 1.000,
    # Asp
    'GAT': 0.464, 'GAC': 1.000,
    # Glu
    'GAA': 0.427, 'GAG': 1.000,
    # Cys
    'TGT': 0.449, 'TGC': 1.000,
    # Trp
    'TGG': 1.000,
    # Arg
    'CGT': 0.159, 'CGC': 0.376, 'CGA': 0.200, 'CGG': 0.376,
    'AGA': 0.753, 'AGG': 1.000,
    # Gly
    'GGT': 0.282, 'GGC': 0.595, 'GGA': 0.416, 'GGG': 1.000,
}
STOP = {"TAA", "TAG", "TGA"}


def cai(cds):
    import math
    log_sum, n = 0.0, 0
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i:i + 3]
        if len(codon) < 3 or codon in STOP:
            break
        w = HUMAN_CODON_W.get(codon)
        if w is not None and w > 0:
            log_sum += math.log(w)
            n += 1
    return math.exp(log_sum / n) if n else 0.0


def kozak_score(seq, atg):
    checks = [(-3, ("A", "G"), 3.0), (3, ("G",), 2.0), (-1, ("C",), 1.0), (-2, ("C",), 1.0),
              (-4, ("A", "G"), 0.5), (-5, ("A", "G"), 0.5)]
    score = total = 0.0
    for off, ok, w in checks:
        total += w
        p = atg + off
        if 0 <= p < len(seq) and seq[p] in ok:
            score += w
    return score / total if total else 0.0


def find_cds(seq):
    best = None
    for m in re.finditer("ATG", seq):
        start = m.start()
        for j in range(start, len(seq) - 2, 3):
            if seq[j:j + 3] in STOP:
                if j - start >= 30 and (best is None or j - start > best[1] - best[0]):
                    best = (start, j)
                break
    return best


def cdna_record(ensg, seq):
    L = len(seq)
    if L == 0:
        return None
    rec = {"ensg": ensg, "manual_gc_content": (seq.count("G") + seq.count("C")) / L * 100,
           "cai": float("nan"), "kozak_score": float("nan"), "utr5_length": float("nan"),
           "utr3_length": float("nan"), "cds_length": float("nan"), "utr5_gc": float("nan"),
           "utr3_gc": float("nan"), "utr3_to_cds_ratio": float("nan")}
    orf = find_cds(seq)
    if orf is None:
        return rec
    atg, stop = orf
    cds = seq[atg:stop]
    utr3_start = stop + 3
    utr5, utr3 = seq[:atg], seq[utr3_start:]
    rec.update(cai=cai(cds), kozak_score=kozak_score(seq, atg), utr5_length=atg,
               utr3_length=max(0, L - utr3_start), cds_length=stop - atg,
               utr5_gc=((utr5.count("G") + utr5.count("C")) / len(utr5) * 100) if utr5 else float("nan"),
               utr3_gc=((utr3.count("G") + utr3.count("C")) / len(utr3) * 100) if utr3 else float("nan"),
               utr3_to_cds_ratio=max(0, L - utr3_start) / (stop - atg))
    return rec


def parse_cdna(path):
    import pandas as pd
    rows, cur, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    r = cdna_record(cur, "".join(buf).upper())
                    if r:
                        rows.append(r)
                cur, buf = line[1:].split("|")[0].split(".")[0].strip(), []
            else:
                buf.append(line)
    if cur:
        r = cdna_record(cur, "".join(buf).upper())
        if r:
            rows.append(r)
    return pd.DataFrame(rows).drop_duplicates("ensg")
