"""
Another utils file but only with ML features related functions. Utils gets a bit crowded.
"""
import re


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

