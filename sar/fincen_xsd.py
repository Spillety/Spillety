"""FinCEN SAR XML builder + structural validation (G7.4)."""
import pathlib
import xml.etree.ElementTree as ET

REQUIRED_ELEMENTS = ("AlertID", "FiledAt", "Amount", "Asset", "Originator", "Beneficiary", "TxHash", "RiskScore", "Narrative")

XSD_PATH = pathlib.Path(__file__).resolve().parent / "fincen_sar.xsd"


def build_fincen_xml(sar_fields: dict) -> str:
    """Serialize mapped SAR fields to the minimal FinCEN SAR XML shape."""
    root = ET.Element("SAR", attrib={"regime": "FinCEN"})
    values = {
        "AlertID": str(sar_fields.get("alert_id", "")),
        "FiledAt": str(sar_fields.get("filed_at", "")),
        "Amount": str(sar_fields.get("amount", "")),
        "Asset": str(sar_fields.get("asset", "")),
        "Originator": str(sar_fields.get("originator", "")),
        "Beneficiary": str(sar_fields.get("beneficiary", "")),
        "TxHash": str(sar_fields.get("tx_hash", "")),
        "RiskScore": str(sar_fields.get("risk_score", "")),
        "Narrative": str(sar_fields.get("suspicious_activity", "")),
    }
    for name in REQUIRED_ELEMENTS:
        child = ET.SubElement(root, name)
        child.text = values[name]
    return ET.tostring(root, encoding="unicode")


def validate_fincen_xml(xml_text: str) -> bool:
    """Structurally validate SAR XML against the required element set.

    # ponytail: true XSD validation needs lxml/xmlschema + network access,
    both unavailable offline; this checks element presence/order and that
    the schema file exists, which catches emitter drift.
    """
    if not XSD_PATH.exists():
        raise FileNotFoundError(f"schema file missing: {XSD_PATH}")
    root = ET.fromstring(xml_text)
    if root.tag != "SAR" or root.attrib.get("regime") != "FinCEN":
        raise ValueError("root must be <SAR regime='FinCEN'>")
    children = [c.tag for c in root]
    if children != list(REQUIRED_ELEMENTS):
        raise ValueError(f"element mismatch: {children}")
    if any((c.text or "").strip() == "" for c in root):
        raise ValueError("empty required element")
    float(root.findtext("Amount"))
    score = float(root.findtext("RiskScore"))
    if not 0.0 <= score <= 1.0:
        raise ValueError("RiskScore out of range 0-1")
    return True
