import numpy as np
from sklearn.neighbors import NearestNeighbors


def sample_hard_negatives(features: np.ndarray, labels: np.ndarray, k: int = 10, threshold: float = 0.85) -> list[int]:
    legit_mask = labels == 5
    scam_mask = labels == 0
    if not legit_mask.any() or not scam_mask.any():
        return []
    nn = NearestNeighbors(n_neighbors=min(k, legit_mask.sum()), metric="cosine").fit(features[legit_mask])
    distances, _ = nn.kneighbors(features[scam_mask])
    hard_idx = np.where(legit_mask)[0][np.where((1 - distances) > threshold)[0]]
    easy_indices = np.where(legit_mask)[0]
    if len(hard_idx) == 0:
        return [int(i) for i in easy_indices[:k]]
    hard_list = list(hard_idx)
    easy_list = [int(i) for i in easy_indices if int(i) not in hard_list][:len(hard_list)]
    return hard_list + easy_list

