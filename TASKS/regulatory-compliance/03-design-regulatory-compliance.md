# 8.5 Regulatory Compliance — Design

## Context
Travel Rule IVMS101, sanctions screening (OFAC/EU/UN), config-driven jurisdiction rules (YAML), compliance audit (docs + tests + model cards).

### [human] Design Decisions
- **Travel Rule IVMS101**: Стандарт для VASP-to-VASP передачи данных; не кастомный формат
- **Sanctions screening**: Real-time проверка против OFAC, EU, UN списков; flag как node feature в TCH-GT
- **Config-driven jurisdiction rules**: YAML-файлы per jurisdiction; динамическая загрузка без деплоя
- **Compliance audit**: Documentation + test cases + model cards для каждой модели

### [agent] resolved
- IVMS101: JSON schema compliant with FATF standard; fields: `originator`, `beneficiary`, `vasp_info`
- Sanctions screening: Batch screening hourly + real-time on new entity; match against downloaded lists (updated daily)
- Jurisdiction config: `jurisdictions/{country}.yaml` loaded at runtime; fallback to default
- Model cards: MLflow-integrated; each model version has `model_card.yaml` with training data, bias, fairness metrics

## Architecture
```
Transaction → IVMS101 Message Builder → Sanctions Screening (OFAC/EU/UN)
  ├── Jurisdiction Config: YAML → rules engine
  ├── TCH-GT Integration: sanctions flag → node feature
  ├── Compliance Audit: docs + tests + model cards
  └── Output: IVMS101 JSON → regulator API (Phase 2)
```

**Screening sources**: OFAC SDN, EU Consolidated List, UN Security Council

## Pushback Hypotheses
1. **H1**: Real-time sanctions screening adds latency to inference pipeline. *Mitigation*: Pre-screen known addresses; cache results with TTL; async screening for new entities. `# ponytail:` screening latency
2. **H2**: YAML jurisdiction configs are prone to typos and misconfiguration. *Mitigation*: JSON Schema validation on load; CI tests verify all jurisdiction files. `# ponytail:` config validation

## Open Questions
- Как часто обновлять sanctions lists (в реальном времени или по расписанию)?
- Нужен ли human-in-the-loop для sanctions match escalation?

## [human] Acceptance
- IVMS101 message format passes FATF compliance test
- Sanctions screening latency < 100ms per transaction
- All jurisdiction rules validated by CI before deployment
- Model cards exist for 100% of production models
