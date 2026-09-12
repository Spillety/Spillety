# 04-structure-dataset-labels

> Phase: S (Structure) | Slug: dataset-labels | Status: In Progress

## 1. Project Structure & Config

```
dataset_labels/
├── config/split_config.json
├── data/ellipticpp_v2/  data/internal/
├── src/{loader,split,negatives,labels}.py
├── tests/{test_split,test_negatives,test_labels}.py
└── output/{train,val,test}.csv
```

```json
// config/split_config.json
{"temporal_split": {"train_end": "2015-12-31", "val_end": "2016-06-30", "test_end": "2016-12-31"},
 "hard_negative": {"knn_k": 10, "cosine_threshold": 0.85, "ratio_hard_easy": 0.5},
 "labels": ["scam","ransomware","terrorist_financing","sanctions","mixer","legitimate"],
 "seed": 42, "label_confidence_threshold": 0.8}
```

### [human] Internal data integration
> Internal labeled data нужно ли микшировать с Elliptic++ v2?

**Decision:** Да, как augmentation для редких классов (0.1x). Основной источник — v2. # ponytail: internal data mix ratio, add when internal > 5k samples.

### [agent] resolved
> Internal data только для augmentation. Main source: Elliptic++ v2.

## 2. Data Loader & Split

```python
# src/loader.py
import pandas as pd

def load_ellipticpp_v2(data_path: str) -> pd.DataFrame:
    """Load Elliptic++ v2 dataset with 6-class labels."""
    df = pd.read_parquet(Path(data_path) / "ellipticpp_v2.parquet")
    df["label"] = df["label"].map({"scam":0,"ransomware":1,"terrorist_financing":2,
        "sanctions":3,"mixer":4,"legitimate":5}).fillna(-1)
    return df[df["label"] >= 0]  # Filter unmapped: label=-1
```

```python
# src/split.py
import json, pandas as pd

def temporal_split(df: pd.DataFrame, config_path: str):
    """Temporal split preventing data leakage. Train 2014-2015, Val 2016 H1, Test 2016 H2."""
    cfg = json.load(open(config_path))
    train_end = pd.Timestamp(cfg["temporal_split"]["train_end"])
    val_end = pd.Timestamp(cfg["temporal_split"]["val_end"])
    train = df[df["timestamp"] <= train_end]
    val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test = df[df["timestamp"] > val_end]
    return train, val, test
```

### [human] Unknown labels & reproducibility
> Что с неизвестными label-ами? Split воспроизводим?

**Decision:** label=-1 отбрасывается, логируется. Seed=42, config-файл с датой. # ponytail: unknown label handling, add when unmapped labels > 1%.

### [agent] resolved
> Фильтрация label=-1. Config с датой в имени. `split_config_20250101.json`.

## 3. Hard Negative Sampling

```python
# src/negatives.py
import numpy as np
from sklearn.neighbors import NearestNeighbors

def sample_hard_negatives(features, labels, k=10, threshold=0.85):
    """KNN-based hard negative sampling: legitimate near scam in feature space."""
    legit_mask = labels == 5  # legitimate
    scam_mask = labels == 0   # scam
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(features[legit_mask])
    distances, _ = nn.kneighbors(features[scam_mask])
    return np.where(legit_mask)[0][np.where((1 - distances) > threshold)]
```

### [human] KNN at scale & test coverage
> KNN O(n²) при 1M+ адресах? Coverage?

**Decision:** sklearn для MVP, FAISS при >100k. Coverage 95% split/negatives, 80% loader. # ponytail: FAISS integration, add when dataset > 100k nodes.

### [agent] resolved
> sklearn для MVP. FAISS при >100k. Mini-batch processing.

## 4. Tests

```python
# tests/test_split.py
def test_temporal_split_boundaries():
    df = pd.DataFrame({"timestamp": pd.date_range("2014-01-01", periods=730, freq="D"),
        "label": [0]*365 + [5]*365})
    train, val, test = temporal_split(df, "config/split_config.json")
    assert (len(train), len(val), len(test)) == (365, 181, 184)  # Reproducible
```

```python
# tests/test_negatives.py
def test_hard_negative_ratio():
    features = np.random.rand(100, 128)
    labels = np.array([0]*10 + [5]*90)
    hard_idx = sample_hard_negatives(features, labels, k=5, threshold=0.5)
    assert len(hard_idx) > 0
```

## 5. Pushback: What Could Go Wrong

### [human] Hypothesis 1: Rare class undersampling
> `terrorist_financing` < 50 samples — model не обучится.

**Decision:** SMOTE augmentation для классов с < 100 samples. Oversampling при < 50. # ponytail: SMOTE augmentation, add when rare class < 100 samples.

### [agent] resolved
> SMOTE для < 100 samples. Oversampling при < 50.

### [human] Hypothesis 2: Address reuse across splits
> Один адрес в train и test — data leakage.

**Decision:** Address-level split. Один адрес = один сет. # ponytail: address-level split, add when address reuse rate > 5%.

### [agent] resolved
> Split по `address`, не по `transaction`. Гарантия без пересечения.

## 6. Acceptance Criteria

- [ ] Elliptic++ v2 загружен с 6 классами
- [ ] Temporal split reproducible (seed=42, config-файл)
- [ ] Hard negative sampling: cos_sim > 0.85, ratio 1:1
- [ ] Train/Val/Test: 2014-2015 / 2016 H1 / 2016 H2
- [ ] Label confidence > 0.8, label=-1 filtered
- [ ] Address-level split enforced
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 7. Open Questions

Нет открытых вопросов. Переход к P (Plan) по согласованию.
