from typing import Any


class NarrativeGenerator:
    """Template-based narrative generation for AML alerts.

    Uses Jinja2-like templates to produce deterministic narratives
    from alert type, causal path, and jurisdiction.
    """

    _TEMPLATES: dict[str, str] = {
        "ransomware": "ALERT: {jurisdiction} ransomware detected. Path: {path}. Immediate review required.",
        "sanctions": "ALERT: {jurisdiction} sanctions violation. Involved party: {path}.",
        "scam": "ALERT: {jurisdiction} scam pattern identified. Route: {path}.",
        "terrorist_financing": "ALERT: {jurisdiction} terrorist financing flagged. Chain: {path}.",
        "mixer": "ALERT: {jurisdiction} mixer usage detected. Flow: {path}.",
        "default": "ALERT: {jurisdiction} anomaly detected. Path: {path}.",
    }

    def generate(self, alert_type: str, causal_path: str, jurisdiction: str) -> str:
        """Generate a narrative from alert components.

        Args:
            alert_type: Type of AML alert.
            causal_path: Serialized causal path string.
            jurisdiction: Jurisdiction code (e.g., 'US', 'EU').

        Returns:
            Formatted narrative string.
        """
        template = self._TEMPLATES.get(alert_type, self._TEMPLATES["default"])
        return template.format(
            jurisdiction=jurisdiction, path=causal_path
        )


def demo() -> None:
    """Smoke test: verify NarrativeGenerator produces correct narratives."""
    gen = NarrativeGenerator()
    r1 = gen.generate("ransomware", "A->B->C", "US")
    assert "ransomware" in r1 and "US" in r1
    r2 = gen.generate("scam", "X->Y", "EU")
    assert "scam" in r2 and "EU" in r2
    r3 = gen.generate("unknown", "Z", "UK")
    assert "unknown" in r3
    print("narrative demo passed")


if __name__ == "__main__":
    demo()
