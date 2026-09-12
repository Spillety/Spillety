# 7.3 SAR/STR Generation — Design

## Context
Jinja2 templates per jurisdiction for SAR/STR filing, FinCEN XML validation, LLM-assisted narrative generation with human review, manual export.

### [human] Design Decisions
- **Jinja2 Templates**: One template per jurisdiction (US FinCEN, EU FIU, UK NCA); parameterized with case data + causal path + counterfactual
- **FinCEN XML Validation**: XSD schema validation against FinCEN SAR XML spec; strict field mapping (parties, amounts, dates, narrative)
- **LLM-Assisted Narrative**: GPT-4 generates preliminary narrative from structured data; human analyst reviews/approves before filing
- **Manual Export**: Final SAR/STR export as FinCEN XML or PDF; audit trail of all edits

### [agent] resolved
- Jinja2: Template variables = `case_data`, `causal_path`, `counterfactual`, `jurisdiction_config`; template per jurisdiction in `templates/`
- FinCEN XML: Validate against `fincen_sar.xsd`; fields: `Subject`, `Activity`, `Amount`, `FilingInstitution`; error on missing required fields
- LLM narrative: Prompt = structured data + causal path; output = narrative draft; human review gate before finalization
- Export: `finCEN_xml_export.py` produces validated XML; `pdf_export.py` generates printable PDF

## Architecture
```
Case Data → Jinja2 Template → Draft SAR/STR
  ├── Jurisdiction Config: template selection by country
  ├── FinCEN XML: XSD validation + field mapping
  ├── LLM Narrative: GPT-4 draft → Human review → Approved
  └── Export: XML (FinCEN) or PDF (manual)
```

**Templates**: `templates/us_fincen_sar.j2`, `templates/eu_fiu.j2`, `templates/uk_nca.j2`

## Pushback Hypotheses
1. **H1**: LLM-generated narratives may contain hallucinated facts. *Mitigation*: Strict fact-checking against structured data; human review mandatory; reject any field not in source data. `# ponytail:` LLM speed vs. accuracy
2. **H2**: Per-jurisdiction template maintenance is high effort. *Mitigation*: Shared base template with jurisdiction-specific overrides; use Jinja2 `extends`/`include`. `# ponytail:` shared vs. independent templates

## Open Questions
- What is the FinCEN XSD version to validate against?
- How to handle LLM refusal on sensitive cases?

## [human] Acceptance
- FinCEN XML validates against XSD with 0 errors
- LLM narrative requires ≤ 2 edits by human analyst
- Template rendering < 5s per case
- Export produces valid XML/PDF within 10s
