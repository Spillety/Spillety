import datetime
import uuid
from typing import Any, Protocol

from .bfs_subgraph import SubgraphExtractor
from .subgraph_cache import SubgraphCache


class TchgtForward(Protocol):
    """Stub interface for the TCH-GT forward pass (wave B provides kernels)."""

    def forward(self, subgraph: dict) -> dict:
        ...


class StubForward:
    """Minimal forward stub for wiring tests; returns a fixed risk score."""

    def __init__(self, risk_score: float = 0.87):
        self._risk = risk_score

    def forward(self, subgraph: dict) -> dict:
        return {"risk_score": float(self._risk)}


class TCHGTOrchestrator:
    """Assembles lazy inference pipeline; kernels stay behind TchgtForward."""

    def __init__(
        self,
        graph: Any,
        extractor: SubgraphExtractor | None = None,
        cache: SubgraphCache | None = None,
        forward_model: TchgtForward | None = None,
        block_threshold: float = 0.8,
        review_threshold: float = 0.5,
    ):
        self._graph = graph
        self._extractor = extractor or SubgraphExtractor(depth=2)
        self._cache = cache or SubgraphCache()
        self._forward = forward_model or StubForward()
        self._block_threshold = block_threshold
        self._review_threshold = review_threshold

    def process_transaction(self, tx: dict) -> dict:
        subgraph = self._get_subgraph(tx)
        raw_score = float(self._forward.forward(subgraph)["risk_score"])
        risk_score = max(0.0, min(1.0, raw_score))
        decision = self._decide(risk_score)
        alert = {
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "transaction": {
                "tx_hash": str(tx["tx_hash"]),
                "from": str(tx["from"]),
                "to": str(tx["to"]),
                "amount": float(tx["amount"]),
                "asset": tx["asset"],
            },
            "risk_score": risk_score,
            "decision": decision,
            "explanation": self._explain_stub(tx, risk_score),
        }
        from model_core.explainability.schema.validator import AMLValidator

        AMLValidator().validate(alert)
        return alert

    def _get_subgraph(self, tx: dict) -> dict:
        key = tx["tx_hash"]
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        subgraph = self._extractor.extract(tx["from"], tx["to"], self._graph)
        self._cache.put(key, subgraph)
        return subgraph

    def _decide(self, risk_score: float) -> str:
        if risk_score >= self._block_threshold:
            return "BLOCK"
        if risk_score >= self._review_threshold:
            return "REVIEW"
        return "ALLOW"

    def _explain_stub(self, tx: dict, risk_score: float) -> dict:
        edge = f"{tx['from']}→{tx['to']}"
        score_without = max(0.0, min(1.0, risk_score * 0.5))
        return {
            "counterfactual": {
                "removed_edge": edge,
                "score_without_edge": score_without,
                "delta": score_without - risk_score,
                "interpretation": "Edge is causally necessary for high risk",
            },
            "causal_path": [{"edge": edge, "causal_effect": risk_score}],
            "hyperbolic_distance": {"nearest_scam_cluster": "unknown", "distance": 0.0, "percentile": 0.0},
            "hawkes_intensity": {"lambda_t": 0.0, "threshold": 2.0, "trigger": "no_burst"},
            "regulatory_references": ["FATF Recommendation 16 (Travel Rule)"],
        }
