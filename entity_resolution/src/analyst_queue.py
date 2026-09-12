class AnalystQueue:
    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold
        self.p0: list[dict] = []
        self.p1: list[dict] = []
        self.p2: list[dict] = []

    def enqueue(self, cluster: dict, confidence: float) -> str:
        entry = {"cluster": cluster, "confidence": confidence}
        if confidence < 0.3 and cluster.get("size", 0) > 50:
            self.p0.append(entry)
            return "P0"
        elif confidence < self.confidence_threshold:
            self.p1.append(entry)
            return "P1"
        else:
            self.p2.append(entry)
            return "P2"

    def get_daily_batch(self) -> list[dict]:
        batch = self.p1.copy()
        self.p1.clear()
        return batch

    def get_immediate(self) -> list[dict]:
        immediate = self.p0.copy()
        self.p0.clear()
        return immediate
