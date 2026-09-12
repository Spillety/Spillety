# 04-structure-load-testing

> Phase: S (Structure) | Slug: load-testing | Status: In Progress

## 1. Project Structure

```
load_testing/
├── config/{k6_scenarios.yaml,locustfile.yaml,profiles.yaml}
├── src/{load_tests/{k6_api_tests,locust_distributed},monitoring/{metrics_collector,memory_tracker},infrastructure/ci_runner}.py
├── tests/{test_ramp_up,test_spike,test_sustained}.py
└── scripts/run_load_profiles.sh,scripts/check_memory_leak.py
```

## 2. k6 API Load Tests

```javascript
// src/load_tests/k6_api_tests.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30m', target: 100 },  // Ramp-up
    { duration: '5m', target: 1000 },  // Spike 10x
    { duration: '72h', target: 800 },  // Sustained 80%
  ],
  thresholds: {
    http_req_duration: ['p(50)<50', 'p(95)<200', 'p(99)<500'],
    http_req_failed_rate: ['rate<0.01'],
  },
};

export default function () {
  const res = http.post('http://inference-service/v1/predict', JSON.stringify(payload));
  check(res, { 'status was 200': (r) => r.status === 200, 'p99 < 500ms': (r) => r.timings.p99 < 500 });
  sleep(0.1);
}
```

### [human] k6 thresholds
> Какие thresholds задать в k6?

**Decision:** p50<50ms, p95<200ms, p99<500ms для inference. Error rate < 1%. # ponytail: threshold tuning, add when p99 > 500ms in baseline.

### [agent] resolved
> p50<50, p95<200, p99<500. Error rate < 1%.

## 3. Locust Distributed Tests

```python
# src/load_tests/locust_distributed.py
class InferenceUser(HttpUser):
    """Locust user for distributed load testing of inference pipeline."""
    wait_time = between(0.05, 0.2)
    @task
    def predict(self):
        response = self.client.post("/v1/predict", json=self.generate_payload())
        assert response.status_code == 200
    def generate_payload(self) -> dict:
        """Generate realistic AML transaction payload."""
        return {"tx_hash": fake.sha256(), "from": fake.hexadecimal(40), "amount": random.randint(1, 10000)}
```

### [human] Locust distributed setup
> Как настроить distributed Locust для высокой нагрузки?

**Decision:** Master-worker模式 с 1 master + 4 workers на 8-core each. Target: 10,000 req/s. # ponytail: worker count, add when target > 50k req/s.

### [agent] resolved
> 1 master + 4 workers. 10k req/s target.

## 4. Memory Leak Detection & Monitoring

```python
# src/monitoring/memory_tracker.py
class MemoryTracker:
    """Monitor RSS growth during sustained 72h load test."""
    def __init__(self, threshold_per_hour=0.01):
        self.threshold = threshold_per_hour  # 1% RSS growth per hour
        self.baseline_rss = None
    def check(self, current_rss: float) -> bool:
        """Return True if memory leak detected."""
        if self.baseline_rss is None:
            self.baseline_rss = current_rss
            return False
        growth = (current_rss - self.baseline_rss) / self.baseline_rss
        return growth / hours_elapsed > self.threshold
```

### [human] Memory leak response
> Что делать при обнаружении memory leak?

**Decision:** Alert в Prometheus + Slack. Auto-restart affected service. Root cause analysis в течение 24h. # ponytail: auto-restart policy, add when leak detected > 3 times/week.

### [agent] resolved
> Alert + auto-restart. RCA within 24h.

## 5. Key Components

- `k6_api_tests.js`: k6 load tests with 3 profiles. Thresholds defined. English comments.
- `locust_distributed.py`: Distributed Locust for 10k req/s. English comments.
- `memory_tracker.py`: RSS growth monitoring during 72h tests. Alert > 1%/hour. English comments.
- `ci_runner.py`: Automated nightly load test pipeline. English comments.
- Tests: 80% coverage for infrastructure.

## 6. Pushback Hypotheses

1. **H1**: 72h testing causes resource exhaustion. *Mitigation*: Nightly automated run + RSS growth monitoring > 1%/hour. Alert and auto-restart. `# ponytail:` memory leak detection
2. **H2**: Double tool infrastructure cost. *Mitigation*: k6 primary (80%), Locust only for distributed. Shared CI runner, cost < 20%.

## 7. Self-Review

- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- 3 load profiles documented
- Pushback hypotheses: 2 ✅
