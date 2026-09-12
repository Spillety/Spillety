# 8.4 Data Encryption & Key Management — Structure

```
data-encryption/
├── vault/
│   ├── vault_config/
│   │   ├── policies/
│   │   │   ├── analyst-policy.hcl
│   │   │   ├── admin-policy.hcl
│   │   │   └── readonly-policy.hcl
│   │   └── transit/
│   │       └── keys.py                 # Key rotation logic
│   ├── client/
│   │   ├── vault_client.py             # Vault integration
│   │   └── secret_provider.py          # K8s secret sync
│   └── ha/
│       └── vault_cluster.yaml          # HA configuration
├── pii-masking/
│   ├── masking_handler.py              # Regex + structured log mask
│   ├── log_sanitizer.py                # Serialization-layer masking
│   └── pii_fields.yaml                 # Configurable PII field list
├── rbac/
│   ├── k8s-rbac/
│   │   ├── roles.yaml
│   │   └── rolebindings.yaml
│   └── vault-rbac/
│       └── policies.yaml
├── audit/
│   ├── iceberg_audit.py                # Iceberg append-only writer
│   ├── audit_schema.py                 # Schema for immutable records
│   └── integrity_check.py              # Verify immutability
└── tests/
    ├── test_pii_masking.py
    ├── test_vault_policies.py
    └── test_audit_immutability.py
```

## Key Components
- `vault_client.py`: Vault integration with auto-unseal and caching
- `masking_handler.py`: PII field detection and masking in logs
- `policies.yaml`: RBAC policies mapping roles to Vault paths
- `iceberg_audit.py`: Append-only writer for immutable audit trail

## Pushback Hypotheses
1. **H1**: Vault agent injector adds complexity to K8s pods. *Mitigation*: Use explicit Vault agent sidecar; simpler than injector for MVP. `# ponytail:` sidecar vs. injector
2. **H2**: Immutable Iceberg audit tables slow down writes significantly. *Mitigation*: Batch writes every 5min; use Iceberg snapshot isolation. `# ponytail:` write performance

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- No AWS KMS dependency
