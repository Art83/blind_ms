"""
Merging the three harmonized sources into one analysis ready source, one row per
(ensg, tissue), with explicit status per source, the hpa vs gtex agreement
class, and the MS-dark candidate flag.

Concept decisions:
1. Core is hpa(ihc) vs gtex (because both yield true negatives).
2. paxdb is positive-only, so it never votes on absence. it contributes
abundance (paxdb_ppm, paxdb_ppm_global) and a corroborating MS positive.
3. gtex absence is split into measured_absent (in matrix, NA this tissue)
vs not_measured (not in the gtex matrix at all). These are different
flavours of MS-dark and are kept distinct.

Main output:
agreement_class (defined only where ihc_status in (present,absent) &
gtex_status in (measured,measured_absent):
both_present: ihc present & gtex measured
both_absent: ihc absent & gtexx measured_absent
IHC_only: ihc present & gtex measured_absent <- MS-dark candidate
GTEx_only: ihc absent   & gtex measured <- IHC false-neg candidate

"""
from __future__ import annotations
from config import TAB_DIR
from utils import _qc, build_grid


grid, tri = build_grid(TAB_DIR)
grid.to_csv(TAB_DIR / "grid.tsv", sep="\t", index=False)

counts = (grid.groupby(["tissue", "agreement_class"]).size().unstack(fill_value=0))
counts.to_csv(TAB_DIR / "grid_class_counts.tsv", sep="\t")
qc = _qc(grid)
(TAB_DIR / "grid_qc.txt").write_text(qc)
print(qc)
