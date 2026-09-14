"""Render SAR filings from mapped fields with Jinja2 (G7.4)."""
import pathlib

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent / "templates"
_REGIME_TEMPLATE = {"FinCEN": "fincen.j2", "EU": "eu.j2", "UK": "uk.j2"}


def render_sar(sar_fields: dict) -> str:
    """Render mapped SAR fields to filing text for sar_fields['regime']."""
    regime = sar_fields.get("regime")
    template_name = _REGIME_TEMPLATE.get(regime)
    if template_name is None:
        raise ValueError(f"unsupported regime: {regime}")
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    return env.get_template(template_name).render(**sar_fields)
