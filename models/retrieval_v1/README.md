# Retrieval v1 — HNSW on real PCA32 embeddings (J3, §3.5)

Anchors: all 46564 labeled txs → PCA32 (fit on train steps≤30 only, seed 72).
HNSW M=24 ef_c=128 ef_s=100.

- recall@10 vs brute on 1000 queries (seed 72): 0.9950
- query latency: p50 0.038 ms, p99 0.097 ms
- footprint: raw 5.96 MB, index 15.86 MB (`data/derived/hnsw_v1.bin`, 15856000 bytes)

## PQ (m=16, sklearn-KMeans codebooks)

- recall@10 after compression: 0.8513 (agreement top-10 = same metric)
- footprint ratio float32→PQ: 8.0x (745024 bytes codes only)
- Assumption: ΔPR-AUC assumed 0.0 — honest measurement needs GBDT retraining on PQ vectors (out of J3 scope). Conditional recommendation via `should_use_pq(0, 0, recall)`: **keep float32**.
