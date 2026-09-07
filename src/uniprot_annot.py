"""
Helper to deal with uniprot download.

Parsing the uniprot feature-table strings that the 2026 export carries into positions and counts.

"""
from __future__ import annotations
import re

_RANGE = re.compile(r"(SIGNAL|PROPEP|CHAIN|CARBOHYD|MOD_RES|DISULFID|LIPID|TRANSMEM)\s+([<>?]?\d+)(?:\.\.([<>?]?\d+))?")
_KEYS = ("SIGNAL", "PROPEP", "CHAIN", "CARBOHYD", "MOD_RES", "DISULFID", "LIPID", "TRANSMEM")
OBSERVED_ECO = {"ECO:0000269", "ECO:0007744"}


def _int(x):
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
    chains = ranges(row.get("Chain", ""), "CHAIN")
    if chains:
        a, b = chains[0]
        return max(1, a), min(length, b)
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

