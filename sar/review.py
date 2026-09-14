"""Human-review gate: a SAR cannot finalize without analyst review (G7.4)."""
import copy
import datetime


class ReviewState:
    DRAFT = "DRAFT"
    REVIEWED = "REVIEWED"
    FINALIZED = "FINALIZED"


def finalize_sar(sar_fields: dict, reviewed: bool = False, reviewer: str = "") -> dict:
    """Finalize a SAR filing; requires explicit human review.

    Raises PermissionError when reviewed is False or reviewer is empty.
    """
    if not reviewed or not reviewer.strip():
        raise PermissionError("SAR finalization requires human review (reviewer + flag)")
    finalized = copy.deepcopy(sar_fields)
    finalized["state"] = ReviewState.FINALIZED
    finalized["reviewer"] = reviewer.strip()
    finalized["finalized_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return finalized
