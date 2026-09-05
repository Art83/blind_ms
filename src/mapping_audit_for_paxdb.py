"""
Checking mapping for the PaxDb symbol->ENSG join.

Note for motivation:
~16% of PaxDb proteins have a symbol that does not resolve to an ENSG in the HPA
gene dictionary and are silently dropped before modelling. Ambiguous symbols are
negligible, so the question that matters for bias is whether the
unmapped proteins differ from the mapped ones on abundance (main contributor to detection).
If unmapped proteins are systematically lower-abundance, dropping
them makes the blind-spot estimate conservative.
if they look the same the loss is ignorable. Any case this add clarity on what the missing 16% are.
"""
from __future__ import annotations
from utils import compare, audit
from config import TAB_DIR, QC_DIR

wb = audit(TAB_DIR)
wb[["symbol", "ensg", "mapped", "log_ppm"]].to_csv(QC_DIR / "mapping_audit.tsv", sep="\t", index=False)
text = compare(wb)
(QC_DIR / "mapping_audit.txt").write_text(text)
print(text)
