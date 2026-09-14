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
    legit_indices = np.where(legit_mask)[0]
    easy_indices = np.where(legit_mask)[0]
    if len(hard_idx) == 0:
        return [int(i) for i in easy_indices[:k]]
    hard_list = list(hard_idx)
    easy_list = [int(i) for i in easy_indices if int(i) not in hard_list][:len(hard_list)]
    return hard_list + easy_list


class HardNegativeSampler:
    def __init__(self, k: int = 10, threshold: float = 0.85):
        self.k = k
        self.threshold = threshold

    def sample(self, features: np.ndarray, labels: np.ndarray) -> list[int]:
        return sample_hard_negatives(features, labels, self.k, self.threshold)


def demo() -> None:
    np.random.seed(42)
    features = np.random.rand(100, 128)
    labels = np.array([0] * 10 + [5] * 90)
    idx = sample_hard_negatives(features, labels, k=5, threshold=0.5)
    assert len(idx) > 0
    sampler = HardNegativeSampler(k=5, threshold=0.5)
    idx2 = sampler.sample(features, labels)
    assert len(idx2) > 0
    print("negatives demo passed")


if __name__ == "__main__":
    demo()
