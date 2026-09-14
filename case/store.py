"""Local file-backed case store: create cases from alert JSON, track status, escalate P0."""
import copy
import datetime
import uuid

P0_SCORE_THRESHOLD = 0.8


class CaseStatus:
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"

    ALL = (OPEN, IN_REVIEW, ESCALATED, CLOSED)


class LocalCaseStore:
    """In-memory case storage with alert-JSON intake and P0 escalation.

    Pragmatic local stand-in for ServiceNow; the ServiceNow adapter
    (servicenow.py) mirrors these cases outbound.
    """

    def __init__(self) -> None:
        self._cases: dict[str, dict] = {}

    def create_case(self, alert: dict) -> dict:
        case_id = str(uuid.uuid4())
        priority = "P0" if self._is_p0(alert) else "P2"
        case = {
            "case_id": case_id,
            "alert_id": alert.get("alert_id"),
            "status": CaseStatus.ESCALATED if priority == "P0" else CaseStatus.OPEN,
            "priority": priority,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "alert": copy.deepcopy(alert),
            "history": ["created"],
        }
        if priority == "P0":
            case["history"].append("auto-escalated P0")
        self._cases[case_id] = case
        return copy.deepcopy(case)

    def get_case(self, case_id: str) -> dict | None:
        case = self._cases.get(case_id)
        return copy.deepcopy(case) if case is not None else None

    def update_status(self, case_id: str, status: str) -> dict:
        if status not in CaseStatus.ALL:
            raise ValueError(f"unknown status: {status}")
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"unknown case: {case_id}")
        # Closed cases are terminal: reopening needs a new case for audit clarity.
        if case["status"] == CaseStatus.CLOSED:
            raise ValueError("closed case cannot change status")
        case["status"] = status
        case["history"].append(f"status->{status}")
        return copy.deepcopy(case)

    def escalate(self, case_id: str) -> dict:
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"unknown case: {case_id}")
        case["priority"] = "P0"
        case["status"] = CaseStatus.ESCALATED
        case["history"].append("manual-escalated P0")
        return copy.deepcopy(case)

    def list_cases(self, status: str | None = None) -> list[dict]:
        cases = list(self._cases.values())
        if status is not None:
            cases = [c for c in cases if c["status"] == status]
        return copy.deepcopy(cases)

    @staticmethod
    def _is_p0(alert: dict) -> bool:
        if alert.get("decision") == "BLOCK":
            return True
        score = alert.get("risk_score", 0.0)
        return isinstance(score, (int, float)) and score >= P0_SCORE_THRESHOLD
