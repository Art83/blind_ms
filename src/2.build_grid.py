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
types of MS-dark and are kept distinct.
"""
from config import TAB_DIR
from utils import _qc, build_grid


grid, tri = build_grid(TAB_DIR)
grid.to_csv(TAB_DIR / "grid.tsv", sep="\t", index=False)

counts = (grid.groupby(["tissue", "agreement_class"]).size().unstack(fill_value=0))
counts.to_csv(TAB_DIR / "grid_class_counts.tsv", sep="\t")
qc = _qc(grid, tri)
(TAB_DIR / "grid_qc.txt").write_text(qc)
print(qc)
