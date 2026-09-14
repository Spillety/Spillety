class NarrativeGenerator:
    """Deterministic AML alert narratives from templates."""

    _TEMPLATES: dict[str, str] = {
        "ransomware": "ALERT: {jurisdiction} ransomware detected. Path: {path}. Immediate review required.",
        "sanctions": "ALERT: {jurisdiction} sanctions violation. Involved party: {path}.",
        "scam": "ALERT: {jurisdiction} scam pattern identified. Route: {path}.",
        "terrorist_financing": "ALERT: {jurisdiction} terrorist financing flagged. Chain: {path}.",
        "mixer": "ALERT: {jurisdiction} mixer usage detected. Flow: {path}.",
        "default": "ALERT: {jurisdiction} anomaly detected. Path: {path}.",
    }

    def generate(self, alert_type: str, causal_path: str, jurisdiction: str) -> str:
        template = self._TEMPLATES.get(alert_type)
        if template is None:
            return f"ALERT: {jurisdiction} anomaly detected ({alert_type}). Path: {causal_path}."
        return template.format(
            jurisdiction=jurisdiction, path=causal_path
        )
