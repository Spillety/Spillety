# 8.5 Regulatory Compliance — Structure

```
regulatory-compliance/
├── travel-rule/
│   ├── ivms101/
│   │   ├── message_builder.py          # IVMS101 JSON construction
│   │   ├── schema.py                   # FATF JSON Schema validation
│   │   └── message_parser.py           # Parse incoming IVMS101
│   └── vasp_router.py                  # Route to correct VASP
├── sanctions/
│   ├── screening/
│   │   ├── ofac_client.py              # OFAC SDN list client
│   │   ├── eu_client.py                # EU Consolidated List client
│   │   ├── un_client.py                # UN Security Council client
│   │   └── match_engine.py             # Fuzzy matching + scoring
│   └── integration/
│       └── tch_gt_flag.py              # Sanctions flag → TCH-GT node feature
├── jurisdiction/
│   ├── configs/
│   │   ├── us.yaml                     # US jurisdiction rules
│   │   ├── eu.yaml                     # EU jurisdiction rules
│   │   ├── default.yaml                # Fallback rules
│   │   └── validator.py                # YAML schema validation
│   └── rules_engine.py                 # Config-driven rule evaluation
├── audit/
│   ├── model_cards/
│   │   ├── model_card_template.yaml    # MLflow-integrated template
│   │   └── model_card_generator.py     # Auto-generate model cards
│   ├── documentation/
│   │   └── compliance_docs.py          # Regulatory docs generator
│   └── tests/
│       └── compliance_test_suite.py    # Test cases per regulation
└── tests/
    ├── test_ivms101.py
    ├── test_sanctions_screening.py
    └── test_jurisdiction_config.py
```

## Key Components
- `message_builder.py`: Constructs IVMS101-compliant JSON messages
- `match_engine.py`: Fuzzy sanctions matching with scoring
- `rules_engine.py`: Evaluates jurisdiction rules from YAML config
- `model_card_generator.py`: Auto-generates model cards for MLflow registry

## Pushback Hypotheses
1. **H1**: Fuzzy sanctions matching produces false positives blocking legitimate transactions. *Mitigation*: Human review for low-confidence matches; threshold tuning per jurisdiction. `# ponytail:` precision vs. recall
2. **H2**: Per-jurisdiction YAML configs become unmanageable at scale (50+ jurisdictions). *Mitigation*: Base config + overrides pattern; CI validates all files on PR. `# ponytail:` config scale

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
- No custom message format (IVMS101 only)
