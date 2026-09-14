import networkx as nx
import yaml

from entity_resolution.src.analyst_queue import AnalystQueue
from entity_resolution.src.anomaly_detection import OverMergingDetector
from entity_resolution.src.exchange_filter import ExchangeFilter
from entity_resolution.src.metrics import pairwise_precision_recall
from entity_resolution.src.ml_pipeline import MLPipeline
from entity_resolution.src.pipeline import EntityResolutionPipeline
from entity_resolution.src.union_find import UnionFind


def _graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edge("a", "b", weight=2.0, timestamp=100.0)
    g.add_edge("b", "c", weight=1.0, timestamp=200.0)
    g.add_edge("hub", "a", weight=5.0, timestamp=300.0)
    g.add_edge("hub", "b", weight=5.0, timestamp=310.0)
    g.add_edge("hub", "c", weight=5.0, timestamp=320.0)
    return g


def test_timing_merges_close_spends_only():
    p = EntityResolutionPipeline()
    p.process_timed_spends(
        [("t1", 0.0), ("t2", 100.0), ("far", 99999.0)], window_sec=3600.0
    )
    assert p.uf.is_connected("t1", "t2")
    assert not p.uf.is_connected("t1", "far")


def test_fee_merges_similar_fees_only():
    p = EntityResolutionPipeline()
    p.process_fee_cluster(
        {"f1": 0.001, "f2": 0.00102, "whale": 5.0}, tolerance=0.05
    )
    assert p.uf.is_connected("f1", "f2")
    assert not p.uf.is_connected("f1", "whale")


def test_heuristics_skip_exchange_addresses():
    ef = ExchangeFilter()
    ef.load_from_labels({"ex"})
    p = EntityResolutionPipeline(exchange_filter=ef)
    assert p.process_transaction(["ex", "victim"]) is None
    assert p.process_timed_spends([("ex", 0.0), ("v", 10.0)]) == []
    assert p.process_fee_cluster({"ex": 0.001, "v": 0.001}) == []
    assert not p.uf.is_connected("ex", "v")


def test_anomaly_detector_threshold_intact():
    det = OverMergingDetector()
    assert not det.check_cluster(1000)
    assert det.check_cluster(1001)


def test_analyst_queue_routing_intact():
    q = AnalystQueue()
    assert q.enqueue({"size": 60}, 0.1) == "P0"
    assert q.enqueue({"size": 3}, 0.5) == "P1"
    assert q.enqueue({"size": 3}, 0.9) == "P2"


def test_embeddings_deterministic_and_structural():
    ml = MLPipeline(model_registry_path="dummy", embedding_dim=32)
    first = ml.generate_embeddings(_graph())
    second = ml.generate_embeddings(_graph())
    assert set(first) == {"a", "b", "c", "hub"}
    assert all(first[n].shape == (32,) for n in first)
    assert all((first[n] == second[n]).all() for n in first)
    assert not (first["hub"] == first["c"]).all()
    assert ml.generate_embeddings(nx.Graph()) == {}


def test_fusion_weights_match_pipeline_yaml():
    with open("entity_resolution/config/pipeline.yaml") as fh:
        cfg = yaml.safe_load(fh)["pipeline"]["merge"]
    assert EntityResolutionPipeline.fused_confidence(0.5, 0.9) == (
        cfg["heuristic_weight"] * 0.5 + cfg["ml_weight"] * 0.9
    ) / (cfg["heuristic_weight"] + cfg["ml_weight"])
    p = EntityResolutionPipeline()
    assert (p.heuristic_weight, p.ml_weight) == (
        cfg["heuristic_weight"], cfg["ml_weight"],
    )
    assert p._fuse(0.5, None) == 0.5


def test_thresholds_config_has_g4_gates():
    with open("entity_resolution/config/thresholds.yaml") as fh:
        cfg = yaml.safe_load(fh)["thresholds"]
    assert cfg["ml"]["precision_min"] == 0.95
    assert cfg["ml"]["recall_min"] == 0.80
    assert cfg["heuristics"]["timing_window_sec"] == 3600
    assert cfg["heuristics"]["fee_tolerance"] == 0.05


def test_precision_recall_gates_on_labeled_data():
    with open("entity_resolution/config/thresholds.yaml") as fh:
        gates = yaml.safe_load(fh)["thresholds"]["ml"]
    p = EntityResolutionPipeline()
    p.process_transaction(["a1", "a2"])
    p.process_transaction(["a2", "a3", "a4"])
    p.process_transaction(["b1", "b2", "b3"])
    p.process_timed_spends([("b3", 0.0), ("b4", 60.0), ("stranger", 10**6)])
    p.process_fee_cluster({"c1": 0.001, "c2": 0.00101, "other": 2.0})
    truth = {
        "a1": "E1", "a2": "E1", "a3": "E1", "a4": "E1",
        "b1": "E2", "b2": "E2", "b3": "E2", "b4": "E2",
        "c1": "E3", "c2": "E3",
        "stranger": "E4", "other": "E5",
    }
    predicted = {addr: p.uf.find(addr) for addr in truth}
    scores = pairwise_precision_recall(predicted, truth)
    assert scores["precision"] >= gates["precision_min"]
    assert scores["recall"] >= gates["recall_min"]


def test_union_find_co_spend_untouched():
    uf = UnionFind()
    ef = ExchangeFilter()
    from entity_resolution.src.union_find import process_co_spend

    assert process_co_spend(uf, ["x", "y"], ef) == uf.find("x")
    assert uf.is_connected("x", "y")
