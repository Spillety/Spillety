# 8.4 Data Encryption & Key Management — Design

## Context
HashiCorp Vault для key management, PII masking в logs, RBAC для MVP, immutable audit logging в Iceberg.

### [human] Design Decisions
- **HashiCorp Vault**: KMS для всех secrets; не AWS KMS (vendor lock-in)
- **PII masking в logs**: Mask `address`, `name`, `email` fields в application logs
- **RBAC для MVP**: Role-based access control; ABAC — Phase 2
- **Immutable audit logging**: Audit trail в Iceberg; immutable once written

### [agent] resolved
- Vault: Transit engine for encryption/decryption; KV v2 for secrets; PKI for TLS certs
- PII masking: Regex-based log filter; `address` → `0x***...***`; log masking in Python `logging` handler
- RBAC: K8s RBAC + Vault policies; roles: `analyst`, `admin`, `readonly`
- Iceberg audit: Append-only tables; `FORCE_UPDATE_MODE` disabled; Hive-style ACID transactions

## Architecture
```
Services → Vault (Transit/KV) → Secrets
  ├── Application Logs → PII Mask Handler → Sanitized Logs
  ├── K8s RBAC → Role Policies → Vault Access
  └── Audit Events → Iceberg Append-Only Table → Immutable Trail
```

**Encryption**: AES-256-GCM для data at rest; TLS 1.3 for data in transit

## Pushback Hypotheses
1. **H1**: Vault introduces single point of failure for all secrets. *Mitigation*: Vault cluster with HA mode; integrated caching; seal key backup via Shamir. `# ponytail:` Vault HA
2. **H2**: PII masking regex may miss edge cases in complex log formats. *Mitigation*: Structured logging (JSON); mask at serialization layer, not regex. `# ponytail:` masking strategy

## Open Questions
- Нужен ли Vault agent injector для K8s pods или manual init?
- Как часто ротировать encryption keys в Vault Transit?

## [human] Acceptance
- Zero PII in plaintext logs
- All secrets fetched from Vault (zero hardcoded credentials)
- RBAC covers 100% of API endpoints
- Audit log immutable and queryable in Iceberg
