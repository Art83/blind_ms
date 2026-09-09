"""
Another utils file but only with ML features related functions. Utils gets a bit crowded.
"""
import re

import numpy as np
import pandas as pd

import uniprot_annot as UA
from config import ISOFORM_PICK, TAB_DIR
from utils import symbol_to_ensg


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


_PEST_POSITIVE = set("RKH")
_PEST_RESIDUES = set("PESTD")


def _frac(seq, aa_set):
    return sum(1 for a in seq if a in aa_set) / len(seq) if seq else 0.0


def _n_glyco_motifs(seq):
    return len(re.findall(r"(?=N[^P][ST])", seq))


def _low_complexity_fraction(seq, win=20, entropy_thresh=2.2):
    """Fraction of 20-residue windows with Shannon entropy below 2.2 bits."""
    import math
    if len(seq) < win:
        return 0.0
    n = low = 0
    for i in range(len(seq) - win + 1):
        w = seq[i:i + win]
        counts = {}
        for a in w:
            counts[a] = counts.get(a, 0) + 1
        ent = -sum((c / win) * math.log(c / win + 1e-12, 2) for c in counts.values())
        n += 1
        low += ent < entropy_thresh
    return low / max(n, 1)


def _disorder_fraction_proxy(seq, win=25):
    """Uversky-style proxy: fraction of 25-residue windows with charged
    fraction >= 0.32 and mean Kyte-Doolittle <= 0.2."""
    if len(seq) < win:
        return 0.0
    h = [KD.get(a, 0.0) for a in seq]
    n = dis = 0
    for i in range(len(seq) - win + 1):
        w = seq[i:i + win]
        if _frac(w, set("KRDE")) >= 0.32 and sum(h[i:i + win]) / win <= 0.2:
            dis += 1
        n += 1
    return dis / max(n, 1)


def _pest_features(seq):
    bounds = [-1] + [i for i, a in enumerate(seq) if a in _PEST_POSITIVE] + [len(seq)]
    scores, covered = [], 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        st, en = a + 1, b
        if en - st < 12:
            continue
        region = seq[st:en]
        sc = 0.22 * _frac(region, _PEST_RESIDUES) - 0.005 * len(region)
        if sc >= 0:
            scores.append(sc)
            covered += en - st
    if not scores:
        return dict(pest_score_max=0.0, pest_score_sum=0.0, n_pest_regions=0, pest_fraction=0.0)
    return dict(pest_score_max=max(scores), pest_score_sum=sum(scores), n_pest_regions=len(scores),
                pest_fraction=covered / len(seq) if seq else 0.0)


def sequence_composition(mature):
    """Composition and motif features on the mature chain."""
    return dict(
        cysteine_fraction=_frac(mature, {"C"}),
        charged_fraction=_frac(mature, set("DEKR")),
        proline_fraction=_frac(mature, {"P"}),
        glycine_fraction=_frac(mature, {"G"}),
        aromatic_fraction=_frac(mature, set("FYW")),
        n_glyco_motif_count=_n_glyco_motifs(mature),
        low_complexity_fraction=_low_complexity_fraction(mature),
        disorder_fraction_proxy=_disorder_fraction_proxy(mature),
        **_pest_features(mature),
    )


def build_features(path_uniprot, path_gene_dict, path_gene_features):
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
        feat.update(sequence_composition(mature))
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

    # merge the transcript block (feature_transcript.py) and the annotation
    # block (feature_annotation.py) by ENSG.
    for name, path in (("transcript", TAB_DIR / "features_transcript.tsv"),
                       ("annotation", TAB_DIR / "features_annotation.tsv")):
        if not path.exists():
            print(f"{name} {path.name} not found, block absent")
            continue
        t = pd.read_csv(path, sep="\t").drop_duplicates("ensg")
        clash = [c for c in t.columns if c != "ensg" and c in feats.columns]
        if clash:
            t = t.drop(columns=clash)
        feats = feats.merge(t, on="ensg", how="left")
        print(f"{name} {t.shape[1] - 1} columns, genes with a row in the block "
              f"{feats[t.columns[1]].notna().mean():.1%}")
    for c in ("is_in_complex", "n_complexes", "in_large_complex", "n_conserved_mirna_sites",
              "n_mirna_families", "uorf_count", "has_uorf", "rbp_n_binding_rbps",
              "rbp_hur_binding", "rbp_hnrnp_binding", "rbp_pumilio_binding", "rbp_destab_binding"):
        if c in feats.columns:
            feats[c] = feats[c].fillna(0)
    if {"n_conserved_mirna_sites", "utr3_length"} <= set(feats.columns):
        kb = feats["utr3_length"] / 1000.0
        feats["mirna_site_density"] = np.where(kb > 0, feats["n_conserved_mirna_sites"] / kb, np.nan)
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


def corum(sym2ensg, path_to_corum):
    if not path_to_corum.exists():
        print(f"CORUM{path_to_corum.name} missing")
        return None
    df = pd.read_csv(path_to_corum, sep="\t", encoding="utf-8", on_bad_lines="skip")
    sub_col = next((c for c in df.columns if "subunit" in c.lower() and "gene" in c.lower()), None)
    org_col = next((c for c in ("Organism", "organism", "Species") if c in df.columns), None)
    if sub_col is None:
        raise ValueError(f"CORUM subunit gene-name column not found in {list(df.columns)}")
    if org_col:
        df = df[df[org_col].astype(str).str.contains("Human|Homo sapiens", case=False, na=False)]
    sizes = {}
    for s in df[sub_col].dropna().astype(str):
        subs = {x.strip() for x in s.split(";") if x.strip()}
        for g in subs:
            sizes.setdefault(g, []).append(len(subs))
    rows = [dict(symbol=g, is_in_complex=1, n_complexes=len(v), complex_size_mean=float(np.mean(v)),
                 complex_size_max=int(max(v)), in_large_complex=int(max(v) >= 10)) for g, v in sizes.items()]
    r = pd.DataFrame(rows)
    r["ensg"] = r["symbol"].map(sym2ensg)
    n = r["ensg"].notna().sum()
    print(f"CORUM {len(df):,} human complexes, {len(r):,} subunit symbols, {n:,} mapped to ENSG ({n / len(r):.1%})")
    return r.dropna(subset=["ensg"]).drop_duplicates("ensg").drop(columns="symbol")


def targetscan(sym2ensg, ts_summary_path, ts_gene_info_path, human_id):
    if not ts_summary_path.exists():
        print(f"TargetScan {ts_summary_path.name} missing")
        return None
    df = pd.read_csv(ts_summary_path, sep="\t", on_bad_lines="skip")
    n_raw = len(df)
    df = df[pd.to_numeric(df["Species ID"], errors="coerce") == human_id].copy()
    df["Total num conserved sites"] = pd.to_numeric(df["Total num conserved sites"], errors="coerce").fillna(0).astype(int)
    df["_tx"] = df["Transcript ID"].astype(str).str.split(".").str[0]
    via_tx = 0
    if ts_gene_info_path.exists():
        gi = pd.read_csv(ts_gene_info_path, sep="\t", on_bad_lines="skip")
        gi = gi[pd.to_numeric(gi["Species ID"], errors="coerce") == human_id]
        gi = pd.DataFrame({"_tx": gi["Transcript ID"].astype(str).str.split(".").str[0],
                           "ensg": gi["Gene ID"].astype(str).str.split(".").str[0]}).drop_duplicates("_tx")
        df = df.merge(gi, on="_tx", how="left")
        via_tx = int(df["ensg"].notna().sum())
    else:
        df["ensg"] = None
    df["ensg"] = df["ensg"].where(df["ensg"].notna(), df["Gene Symbol"].map(sym2ensg))
    n_unmapped = int(df["ensg"].isna().sum())
    df = df.dropna(subset=["ensg"])
    agg = df.groupby("ensg").agg(n_conserved_mirna_sites=("Total num conserved sites", "sum"),
                                 n_mirna_families=("miRNA family", "nunique")).reset_index()
    print(f"TargetScan {n_raw:,} rows, human rows {len(df) + n_unmapped:,}, mapped via Gene_info {via_tx:,}, "
          f"via symbol {len(df) - via_tx:,}, unmapped {n_unmapped:,}; {len(agg):,} genes, "
          f"{int((agg['n_conserved_mirna_sites'] > 0).sum()):,} with at least one conserved site")
    return agg


def uorfs(sorf_path):
    if not sorf_path.exists():
        print(f"sORFs {sorf_path.name} missing")
        return None
    s = pd.read_csv(sorf_path, low_memory=False)
    s.columns = [c.lower().strip() for c in s.columns]
    if "gene_id" not in s.columns:
        raise ValueError(f"sORFs export has no gene_id column: {list(s.columns[:10])}")
    s["ensg"] = s["gene_id"].astype(str).str.split(".").str[0]
    g = s.groupby("ensg")
    agg = pd.DataFrame({"uorf_count": g.size(), "uorf_max_length_aa": g["sorf_length"].max(),
                        "uorf_mean_length_aa": g["sorf_length"].mean(),
                        "uorf_mean_phastcon": g["phastcon"].mean()}).reset_index()
    agg["has_uorf"] = 1
    print(f"sORFs.org {len(s):,} sORFs, {len(agg):,} genes with a 5'UTR sORF")
    return agg


def eclip(eclip_path, gene_coords, refresh=False):
    import gzip
    cache = TAB_DIR / "eclip_binding.tsv"
    if cache.exists() and not refresh:
        b = pd.read_csv(cache, sep="\t", dtype=str, keep_default_na=False)
        binding = {e: set(r.split(";")) - {""} for e, r in zip(b["ensg"], b["rbps"])}
        print(f"eCLIP cached {cache.name}: {len(binding):,} genes with a peak")
    else:
        beds = sorted(list(eclip_path.glob("*.bed.gz")) + list(eclip_path.glob("*.bed"))) if eclip_path.exists() else []
        if not beds or not gene_coords.exists():
            print(f"  [eCLIP] {len(beds)} bed files, coordinates {'present' if gene_coords.exists() else 'missing'}, skipped")
            return None
        gc = pd.read_csv(gene_coords, sep="\t", low_memory=False)
        gc.columns = ["ensg", "chrom", "start", "end"]
        gc["ensg"] = gc["ensg"].astype(str).str.split(".").str[0]
        gc["chrom"] = gc["chrom"].astype(str)
        gc = gc[gc["chrom"].str.match(r"^(\d+|X|Y|MT)$")]
        genes_by_chrom = {c: (g["ensg"].values, g["start"].values.astype(int), g["end"].values.astype(int))
                          for c, g in gc.groupby("chrom")}
        binding = {}
        for i, bed in enumerate(beds, 1):
            rbp = bed.name.split("_")[0].upper()
            opener = gzip.open if bed.name.endswith(".gz") else open
            peaks = []
            with opener(bed, "rt") as fh:
                for line in fh:
                    if line.startswith(("#", "track", "browser")):
                        continue
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 3:
                        try:
                            peaks.append((p[0].replace("chr", "", 1), int(p[1]), int(p[2])))
                        except ValueError:
                            pass
            if not peaks:
                continue
            pk = pd.DataFrame(peaks, columns=["chrom", "ps", "pe"])
            for chrom, cp in pk.groupby("chrom"):
                if chrom not in genes_by_chrom:
                    continue
                ens, gs, ge = genes_by_chrom[chrom]
                order = np.argsort(cp["ps"].values)
                ps = cp["ps"].values[order]
                pe_cummax = np.maximum.accumulate(cp["pe"].values[order])
                idx = np.searchsorted(ps, ge, side="left")
                hit = (idx > 0) & (pe_cummax[np.maximum(idx - 1, 0)] > gs)
                for e in ens[hit]:
                    binding.setdefault(e, set()).add(rbp)
            if i % 100 == 0:
                print(f"...{i} of {len(beds)} bed files")
        pd.DataFrame({"ensg": list(binding), "rbps": [";".join(sorted(r)) for r in binding.values()]}) \
            .to_csv(cache, sep="\t", index=False)
        print(f"eCLIP {len(beds)} bed files, {len(binding):,} genes with at least one overlapping peak, cached to {cache.name}")
    rows = [dict(ensg=e, rbp_n_binding_rbps=len(r),
                 rbp_hur_binding=int(any(x in {"ELAVL1", "HUR"} for x in r)),
                 rbp_hnrnp_binding=int(any(x.startswith("HNRNP") for x in r)),
                 rbp_pumilio_binding=int(any(x in {"PUM1", "PUM2"} for x in r)),
                 rbp_destab_binding=int(any(x in {"ZC3H12A", "TTP", "ZFP36"} for x in r)))
            for e, r in binding.items() if r]
    return pd.DataFrame(rows)


# AlphaFold
def plddt_from_cif(text):
    fields, out, in_loop = [], {}, False
    for line in text.splitlines():
        if line.startswith("_atom_site."):
            fields.append(line.strip().split(".", 1)[1])
            in_loop = True
            continue
        if in_loop and (line.startswith("ATOM") or line.startswith("HETATM")):
            p = line.split()
            if len(p) < len(fields):
                continue
            row = dict(zip(fields, p))
            if row.get("label_atom_id") != "CA":
                continue
            try:
                out[int(row["label_seq_id"])] = float(row["B_iso_or_equiv"])
            except (KeyError, ValueError):
                pass
        elif in_loop and line.startswith("#") and out:
            break
    return out


def read_alphafold(tar_path, comp):
    import tarfile, gzip
    per_acc = {}
    n_members = 0
    with tarfile.open(tar_path, "r|*") as tf:
        for m in tf:
            mm = comp.search(m.name)
            if not mm:
                continue
            acc, frag = mm.group(1), int(mm.group(2))
            f = tf.extractfile(m)
            if f is None:
                continue
            text = gzip.decompress(f.read()).decode("utf-8", errors="replace")
            pl = plddt_from_cif(text)
            if not pl:
                continue
            n_members += 1
            offset = (frag - 1) * 200          # AF fragments start every 200 residues
            d = per_acc.setdefault(acc, {})
            for pos, v in pl.items():
                d.setdefault(pos + offset, []).append(v)
            if n_members % 2000 == 0:
                print(f"... {n_members:,} models read")
    return {acc: {p: float(np.mean(v)) for p, v in d.items()} for acc, d in per_acc.items()}, n_members


def af_summary(pl, m_start, m_end):
    vals = np.array([pl[p] for p in range(m_start, m_end + 1) if p in pl], dtype=float)
    n_mature = m_end - m_start + 1
    if len(vals) == 0:
        return dict(af_plddt_mean=np.nan, af_frac_lt50=np.nan, af_frac_lt70=np.nan, af_frac_gt90=np.nan,
                    af_longest_lt50_run=np.nan, af_coverage=0.0)
    lt50 = vals < 50
    longest = run = 0
    for b in lt50:
        run = run + 1 if b else 0
        longest = max(longest, run)
    return dict(af_plddt_mean=float(vals.mean()), af_frac_lt50=float(lt50.mean()),
                af_frac_lt70=float((vals < 70).mean()), af_frac_gt90=float((vals > 90).mean()),
                af_longest_lt50_run=int(longest), af_coverage=float(len(vals) / n_mature))


def read_half_life(sym2ensg, path_to_HL, human_types, qual):
    if not path_to_HL.exists():
        print(f"half-life {path_to_HL.name} not found")
        return None
    h = pd.read_csv(path_to_HL, na_values=["#N/A"])
    val_cols = [c for c in h.columns if c.endswith("half_life") and any(c.startswith(t) for t in human_types)]
    rows = []
    for _, r in h.iterrows():
        vals = []
        for c in val_cols:
            q = r.get(c.replace("half_life", "dataQual"))
            v = r[c]
            if pd.notna(v) and isinstance(q, str) and q in qual:
                vals.append(float(v))
        if vals:
            rows.append(dict(symbol=r["gene_name"], hl_log2_hours=float(np.log2(np.median(vals))),
                             hl_n_values=len(vals)))
    d = pd.DataFrame(rows)
    d["ensg"] = d["symbol"].map(sym2ensg)
    n = d["ensg"].notna().sum()
    print(f"half-life {len(h):,} symbols in file, {len(d):,} with a usable human value, {n:,} mapped to ENSG")
    return d.dropna(subset=["ensg"]).drop_duplicates("ensg").drop(columns="symbol")