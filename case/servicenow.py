"""ServiceNow adapter: interface + offline stub (G7.3)."""
from typing import Protocol


class ServiceNowClient(Protocol):
    """Outbound case-sync interface; real impl needs credentials + network."""

    def push_case(self, case: dict) -> str:
        ...

    def fetch_status(self, external_id: str) -> str:
        ...


class StubServiceNowAdapter:
    """Local stub: records pushes in memory, no network calls.

    # ponytail: real adapter needs ServiceNow instance URL + OAuth credentials
    and network access, none of which exist in this offline environment.
    """

    def __init__(self) -> None:
        self.pushed: dict[str, dict] = {}

    def push_case(self, case: dict) -> str:
        external_id = f"SNOW-{case['case_id'][:8]}"
        self.pushed[external_id] = dict(case)
        return external_id

    def fetch_status(self, external_id: str) -> str:
        if external_id not in self.pushed:
            raise KeyError(f"unknown ticket: {external_id}")
        return str(self.pushed[external_id].get("status", "OPEN"))
