from spillety.evidence.evidence import build_evidence, evidence_hash, verify_evidence
from spillety.evidence.merkle import (
    merkle_proof,
    merkle_root,
    ots_anchor,
    sign_root,
    verify_proof,
    verify_root,
)
from spillety.evidence.sar import approve_sar, from_evidence, submit_sar

# ponytail: legacy `spillety.evidence.worm` path removed with worm.py; S4 repoints
# tests/test_pipeline.py to this module and drops the HMAC-based verify call.
__all__ = [
    "approve_sar",
    "build_evidence",
    "evidence_hash",
    "from_evidence",
    "merkle_proof",
    "merkle_root",
    "ots_anchor",
    "sign_root",
    "submit_sar",
    "verify_evidence",
    "verify_proof",
    "verify_root",
]
