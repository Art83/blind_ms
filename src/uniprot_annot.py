"""
Helper to deal with uniprot download.

Parsing the uniprot feature-table strings that the 2026 export carries into positions and counts.

"""
from __future__ import annotations
import re

_KEYS = ("SIGNAL", "PROPEP", "CHAIN", "CARBOHYD", "MOD_RES", "DISULFID", "LIPID", "TRANSMEM", "TOPO_DOM")
_RANGE = re.compile(r"(%s)\s+([<>?]?\d+|\?)(?:\.\.([<>?]?\d+|\?))?" % "|".join(_KEYS))
OBSERVED_ECO = {"ECO:0000269", "ECO:0007744"}


def _int(x):
    # unknown positions ("?") return None and the range is dropped by ranges()
    try:
        return int(x.lstrip("<>?"))
    except (ValueError, AttributeError):
        return None


def _split_features(text, key):
    text = text or ""
    starts = [m.start() for m in re.finditer(r"(?:^|;\s*)(%s)\s" % "|".join(_KEYS), text)]
    chunks = []
    for i, st in enumerate(starts):
        en = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[st:en].lstrip("; ")
        if chunk.startswith(key + " "):
            chunks.append(chunk)
    return chunks


def ranges(text, key, evidence=None):
    out = []
    for chunk in _split_features(text, key):
        m = _RANGE.match(chunk)
        if not m:
            continue
        if evidence is not None:
            obs = bool(OBSERVED_ECO & set(re.findall(r"ECO:\d{7}", chunk)))
            if (evidence == "observed") != obs:
                continue
        a, b = _int(m.group(2)), _int(m.group(3)) if m.group(3) else _int(m.group(2))
        if a is not None and b is not None:
            out.append((a, b))
    return out


def count(text, key, evidence=None):
    return len(ranges(text, key, evidence))


def site_positions(row, evidence="predicted"):
    glyco = {a for a, _ in ranges(row.get("Glycosylation", ""), "CARBOHYD", evidence)}
    mod = {a for a, _ in ranges(row.get("Modified residue", ""), "MOD_RES", evidence)}
    lipid = {a for a, _ in ranges(row.get("Lipidation", ""), "LIPID", evidence)}
    ss = set()
    for a, b in ranges(row.get("Disulfide bond", ""), "DISULFID", evidence):
        ss.add(a)
        ss.add(b)
    return dict(glyco=glyco, mod=mod, lipid=lipid, disulfide=ss)


def mature_range(row, length):
    # span of all fully specified chains: a protein processed into several chains has all of them in the sample
    chains = [(a, b) for a, b in ranges(row.get("Chain", ""), "CHAIN") if b >= a]
    if chains:
        return max(1, min(a for a, _ in chains)), min(length, max(b for _, b in chains))
    start, end = 1, length
    sig = ranges(row.get("Signal peptide", ""), "SIGNAL")
    if sig:
        start = sig[0][1] + 1
    for a, b in ranges(row.get("Propeptide", ""), "PROPEP"):
        if a == start:
            start = b + 1
        elif b >= length:
            end = min(end, a - 1)
    return start, max(start, end)


# --- localisation from uniprot ---
# Keyword matches on the UniProt subcellular location text (custom rule). The first eight flags use the
# same keywords as hpa_to_tabula_rev/src/external_features.py and define n_compartments; the rest are
# extra flags for this project.
SHARED_LOC_FLAGS = {
    "is_secreted": ("secreted", "extracellular"),
    "is_membrane": ("cell membrane", "single-pass", "multi-pass"),
    "is_nuclear": ("nucleus", "nucleoplasm", "nucleolus", "nuclei"),
    "is_cytoplasmic": ("cytoplasm", "cytosol"),
    "is_mitochondrial": ("mitochondri",),
    "is_er_golgi": ("endoplasmic reticulum", "golgi"),
    "is_cytoskeleton": ("cytoskeleton", "microtubule"),
    "is_projection": ("cell projection", "cilium", "axon", "growth cone", "ranvier", "dendrite", "synaptic",
                      "synapse", "neurite", "dendritic spine", "postsynaptic", "presynaptic", "nerve terminal",
                      "neuronal projection"),
}
EXTRA_LOC_FLAGS = {
    "is_ecm": ("extracellular matrix", "basement membrane", "collagen"),
    "is_chromatin": ("chromosome", "chromatin", "centromere", "kinetochore", "telomere"),
    "is_lysosome": ("lysosom",),
    "is_peroxisome": ("peroxisom",),
    "is_endosome": ("endosom",),
    "is_vesicle": ("cytoplasmic vesicle", "secretory vesicle", "synaptic vesicle", "vesicle membrane",
                   "secretory granule", "zymogen granule", "melanosome", "lysosome-related"),
    "is_lipid_droplet": ("lipid droplet",),
}
LOC_FLAGS = {**SHARED_LOC_FLAGS, **EXTRA_LOC_FLAGS}
TOPOLOGY = {   # phrase in the CC text -> class
    "multi-pass membrane protein": "multipass",
    "single-pass type i membrane protein": "single_type1",
    "single-pass type ii membrane protein": "single_type2",
    "single-pass type iii membrane protein": "single_other",
    "single-pass type iv membrane protein": "single_other",
    "single-pass membrane protein": "single_other",
    "peripheral membrane protein": "peripheral",
    "lipid-anchor": "lipid_anchor",
    "gpi-anchor": "lipid_anchor",
}
TOPO_CLASSES = ["multipass", "single_type1", "single_type2", "single_other", "peripheral", "lipid_anchor"]
_ISO = re.compile(r"\[Isoform[^\]]*\]:")
_CLAUSE = re.compile(r"[.;]\s+")


def _cc_clauses(cc):
    cc = (cc or "").strip()
    if not cc:
        return []
    cc = _ISO.split(cc)[0]
    cc = re.sub(r"^SUBCELLULAR LOCATION:\s*", "", cc)
    out = []
    for cl in _CLAUSE.split(cc):
        cl = cl.strip()
        if not cl or cl.lower().startswith("note="):
            continue
        codes = set(re.findall(r"ECO:\d{7}", cl))
        obs = bool(codes & OBSERVED_ECO)
        out.append((re.sub(r"\{[^}]*\}", "", cl).strip().lower(), obs))
    return out


def _go_terms(go):
    return [t.split("[GO:")[0].strip().lower() for t in (go or "").split(";") if t.strip()]


def localisation(cc, topo_dom, go):
    """Compartment flags, topology class, membrane-domain fractions, evidence counts."""
    clauses = _cc_clauses(cc)
    go_terms = _go_terms(go)
    cc_full = re.sub(r"\{[^}]*\}", "", (cc or "")).lower()
    texts = [cc_full] if cc_full.strip() else go_terms
    f = {}
    for flag, keys in LOC_FLAGS.items():
        f[flag] = int(any(any(k in t for k in keys) for t in texts))
    cc_text = " ".join(c for c, _ in clauses)
    f["n_compartments"] = int(sum(f[k] for k in SHARED_LOC_FLAGS))
    f["is_multilocal"] = int(f["n_compartments"] > 1)
    topo = None
    for phrase, cls in TOPOLOGY.items():
        if phrase in cc_text:
            topo = cls if topo is None else topo # first match wins
            if cls == "multipass":
                topo = "multipass"
    for cls in TOPO_CLASSES:
        f[f"topo_{cls}"] = int(topo == cls)
    ext = cyt = 0
    for chunk in _split_features(topo_dom, "TOPO_DOM"):
        m = _RANGE.match(chunk)
        if not m:
            continue
        a, b = _int(m.group(2)), _int(m.group(3)) if m.group(3) else _int(m.group(2))
        if a is None or b is None:
            continue
        note = chunk.lower()
        if "extracellular" in note or "lumenal" in note or "luminal" in note:
            ext += b - a + 1
        elif "cytoplasmic" in note:
            cyt += b - a + 1
    f["topo_extracellular_aa"] = ext
    f["topo_cytoplasmic_aa"] = cyt
    f["loc_n_clauses"] = len(clauses)
    f["loc_n_observed"] = int(sum(o for _, o in clauses))
    f["loc_go_n_terms"] = len(go_terms)
    f["loc_source"] = "cc" if cc_full.strip() else ("go" if go_terms else "none")
    return f


