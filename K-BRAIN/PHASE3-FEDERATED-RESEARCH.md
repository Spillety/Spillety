# Phase 3: Federated Learning — Research Document

> Статус: Future Research (не часть MVP или Phase 2)
> Создано на основе реорганизации K-BRAIN/TODO.md, 2026-09-11

## Scope

Консорциумный подход к обучению модели AML на данных нескольких организаций без обмена raw-данными.

## Key Technical Areas

- **Differential Privacy:** ε-DP guarantees для каждого участника.
- **Secure Multi-Party Computation (SMPC):** Для aggregation gradients без раскрытия отдельных вкладов.
- **Regulatory:** Antitrust review — консорциум VASPs может вызвать вопросы антимонопольного регулирования.

## Open Questions for Future

1. Какой protocol для SMPC (secret sharing vs garbled circuits)?
2. Как обеспечить convergence при различной качестве данных у участников?
3. Governance model для consortium — кто контролирует aggregation server?
4. Compliance с cross-border data regulations при федеративном обучении?

## Dependencies

- Phase 1 stable (EVM-only, on-chain data).
- Phase 2 stable (internal labeled dataset, causal + hyperbolic model).
- Infrastructure блок (K8s, feature store) готов к distributed setting.

## Timeline

12-18 months от начала Phase 2.

## Status

Deferred. Исследование начнётся после стабилизации Phase 2.
