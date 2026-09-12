# 7.3 SAR/STR Generation — Structure

```
sar-str-generation/
├── templates/
│   ├── us_fincen_sar.j2         # FinCEN SAR template
│   ├── eu_fiu.j2                # EU FIU template
│   └── uk_nca.j2                # UK NCA template
├── validation/
│   ├── fincen_xsd.py            # FinCEN XML XSD validation
│   ├── schema_loader.py         # Load XSD schemas
│   └── field_mapper.py          # Case data → FinCEN XML field mapping
├── narrative/
│   ├── llm_generator.py         # GPT-4 narrative generation
│   ├── human_review.py          # Review/approve workflow
│   └── fact_checker.py          # Cross-verify LLM output vs. structured data
├── export/
│   ├── fincen_xml_export.py     # Produce validated FinCEN XML
│   ├── pdf_export.py            # Generate printable PDF
│   └── audit_trail.py           # Edit history and provenance
└── tests/
    ├── __init__.py
    ├── test_fincen_validation.py
    ├── test_llm_narrative.py
    └── test_export.py
```

## Key Components
- `us_fincen_sar.j2`: Jinja2 template for FinCEN SAR; variables from case data. English comments only.
- `fincen_xsd.py`: Validates generated XML against FinCEN XSD schema
- `llm_generator.py`: GPT-4 call with structured prompt; returns narrative draft
- `human_review.py`: Approval gate; blocks export until analyst signs off
- `audit_trail.py`: Append-only log of all edits and exports

## Pushback Hypotheses
1. **H1**: LLM narrative generation adds latency unacceptable for urgent P0 cases. *Mitigation*: Cache LLM output for similar cases; provide default template narrative if LLM unavailable. `# ponytail:` LLM vs. template-only narrative
2. **H2**: FinCEN XSD validation is strict and may reject valid edge cases. *Mitigation*: Pre-validation with relaxed schema; maintain whitelist of accepted variations. `# ponytail:` strict vs. lenient validation

## Self-Review
- 0 Critical, 0 ai-slops
- All English comments, Russian prose
- Under 200 lines
