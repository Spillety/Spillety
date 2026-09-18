# ER v1 — CIOH + fusion on Elliptic real data

Assumption: edgelist edge (txId1->txId2) is treated as one co-spending transaction [txId1, txId2], i.e. address≈txId; clusters = transitive closure (connected components) of this CIOH proxy. Clustering is unsupervised, so no temporal leakage into cluster precision; fusion uses a temporal split (train max(step)<= 35, valid above).

Clusters: 49 over 203769 nodes (0 singletons), size median 4291 / mean 4158.6 / max 7880.

Cluster precision (≥2 labeled): 0.0000 unweighted, 0.0000 labeled-size-weighted (0/49 pure).

Fusion PR-AUC on 22953 temporal-valid pairs: logistic=0.8870, clayton=0.8668, gumbel=0.8660, frank=0.8668. Selected: logistic.

Clusters parquet: data/derived/er_v1_clusters.parquet (1.3 MB).
