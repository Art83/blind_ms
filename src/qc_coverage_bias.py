"""
Bias check for the abundance-coverage drop.

The symbol-join drops ~24% of all paxdb proteins

Questions:
1. do those IHC-present proteins have a different MS-dark rate than the retained ones? (if not, the drop is inert)
2. are they enriched for MS-dark specifically? (if yes, the cross-atlas identifier merge is itself detectability structured)
"""
from __future__ import annotations
from utils import _report, _build
from config import TAB_DIR, QC_DIR


m = _build(TAB_DIR)
text = _report(m)
(QC_DIR / "coverage_bias.txt").write_text(text)
print(text)
