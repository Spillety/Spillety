import numpy as np
import torch
import networkx as nx

from model_core.inference.bfs_subgraph import SubgraphExtractor
from model_core.inference.micro_batch import MicroBatchBuilder
from model_core.inference.gpu_forward import GPUInference
from model_core.inference.subgraph_cache import SubgraphCache


def test_bfs_subgraph_extractor():
    G = nx.karate_club_graph()
    extractor = SubgraphExtractor(depth=2)
    result = extractor.extract(0, G)
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)
    assert len(result["nodes"]) > 0


def test_bfs_subgraph_union_from_to():
    G = nx.path_graph(6)
    extractor = SubgraphExtractor(depth=2)
    single = extractor.extract(0, G)
    union = extractor.extract(0, 5, G)
    assert set(single["nodes"]).issubset(set(union["nodes"]))
    assert 0 in union["nodes"] and 5 in union["nodes"]
    # depth 2 from both ends of path 0-1-2-3-4-5 covers all nodes
    assert set(union["nodes"]) == {0, 1, 2, 3, 4, 5}


def test_micro_batch_builder():
    builder = MicroBatchBuilder(min_size=32, max_size=64)
    events = [{"data": np.random.randn(10)} for _ in range(40)]
    batches = builder.build(events)
    assert len(batches) == 1
    assert len(batches[0]) == 40
    assert all(e != {} for e in batches[0])


def test_micro_batch_flush():
    builder = MicroBatchBuilder(min_size=2, max_size=4)
    events = [{"data": np.array([1.0])} for _ in range(3)]
    batches = builder.build(events)
    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert all(e != {} for e in batches[0])


def test_gpu_inference():
    model = torch.nn.Linear(10, 5)
    inference = GPUInference(model, device="cpu")
    batch = torch.randn(32, 10)
    output = inference.forward(batch)
    assert output.shape == (32, 5)
    large_batch = torch.randn(128, 10)
    out_large = inference.forward(large_batch)
    assert out_large.shape == (128, 5)


def test_subgraph_cache():
    cache = SubgraphCache(capacity=3, ttl_hours=1.0)
    cache.put("addr1", {"emb": [1.0, 2.0]})
    cache.put("addr2", {"emb": [3.0, 4.0]})
    result = cache.get("addr1")
    assert result is not None
    assert result["emb"] == [1.0, 2.0]
    cache.put("addr3", {"emb": [5.0, 6.0]})
    cache.put("addr4", {"emb": [7.0, 8.0]})
    assert len(cache) == 3
    assert cache.get("addr2") is None


def test_subgraph_cache_ttl_expiry():
    cache = SubgraphCache(capacity=10, ttl_hours=0.0)
    cache.put("addr1", {"emb": [1.0]})
    import time
    time.sleep(0.01)
    result = cache.get("addr1")
    assert result is None


def test_cache_hit_rate():
    cache = SubgraphCache(capacity=100, ttl_hours=1.0)
    for i in range(50):
        cache.put(f"addr_{i}", {"emb": [float(i)]})
    hits = sum(1 for i in range(50) if cache.get(f"addr_{i}") is not None)
    assert hits / 50 > 0.6


def test_orchestrator_returns_valid_v2():
    from model_core.inference.orchestrator import TCHGTOrchestrator, StubForward
    from model_core.explainability.schema.validator import AMLValidator
    G = nx.path_graph(6)
    orch = TCHGTOrchestrator(G, forward_model=StubForward(risk_score=0.87))
    tx = {"tx_hash": "0xabc", "from": 0, "to": 5, "amount": 125000.0, "asset": "USDT"}
    alert = orch.process_transaction(tx)
    assert alert["decision"] == "BLOCK"
    assert alert["risk_score"] == 0.87
    assert AMLValidator().validate(alert) is True
    # second call hits cache, same shape
    alert2 = orch.process_transaction(tx)
    assert alert2["transaction"] == alert["transaction"]


def test_orchestrator_decision_thresholds():
    from model_core.inference.orchestrator import TCHGTOrchestrator, StubForward
    G = nx.path_graph(6)
    base = {"tx_hash": "0xt", "from": 0, "to": 1, "amount": 1.0, "asset": "USDT"}
    assert TCHGTOrchestrator(G, forward_model=StubForward(0.9)).process_transaction(base)["decision"] == "BLOCK"
    assert TCHGTOrchestrator(G, forward_model=StubForward(0.6)).process_transaction({**base, "tx_hash": "0xt2"})["decision"] == "REVIEW"
    assert TCHGTOrchestrator(G, forward_model=StubForward(0.1)).process_transaction({**base, "tx_hash": "0xt3"})["decision"] == "ALLOW"


if __name__ == "__main__":
    test_bfs_subgraph_extractor()
    test_micro_batch_builder()
    test_micro_batch_flush()
    test_gpu_inference()
    test_subgraph_cache()
    test_subgraph_cache_ttl_expiry()
    test_cache_hit_rate()
    test_orchestrator_returns_valid_v2()
    test_orchestrator_decision_thresholds()
    print("All inference tests passed")
