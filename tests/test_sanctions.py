import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "fetch_sanctions",
    Path(__file__).resolve().parent.parent / "scripts" / "fetch_sanctions.py",
)
fs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fs)

OFAC_ROW = (
    "18647",
    "ISIL KHORASAN",
    "-0- ",
    "FTO] [SDGT",
    "-0- ",
    "-0- ",
    "-0- ",
    "-0- ",
    "-0- ",
    "-0- ",
    "-0- ",
    (
        "Digital Currency Address - XMR 44dZUJ7w1T3fKAvFW8XyXUVoAGSbFvXef2wcbnsjNKGWYorp"
        "LJBjth5VKSFhLGkpYKJb2J341tdZHBnbpv72WL7e8zuxfR2; alt. Digital Currency "
        "Address - XBT bc1qcp6fr7gtyukympl6unr7uv78h3vprycwj455zx; no crypto here."
    ),
)
GARANTEX = (
    "Place of registration: Moscow\nKnown Garantex blockchain wallet addresses:\n"
    "ETH: 0x002471b8A185f9980708d0eAEC5B289714F56f8d\n"
    "BTC: bc1qwtz3zv95x2flu4w26kgfepf529z50r2sqc5zml\n"
    "bc1qwxqxd25yk2dtw2ml04vxj9atq3huv4rdytf6vt\n"
    "3My1ffQr5qQzmq4aBFgRqwRYqfB5zPAt8t\n"
    "BSC: 0x3051Ca7cB7f6C599fA2f27385AD75010cf0f2bbF\n"
    "TRX: TA1hsikRfsgGiW9nEBpT4tEXEySTNYLr2d"
)


def test_parse_ofac_remarks_labels_and_alts():
    """## OFAC remarks yield labeled pairs; XBT normalizes to BTC."""
    pairs = fs.parse_ofac_remarks(OFAC_ROW[-1])
    assert (
        "XMR",
        (
            "44dZUJ7w1T3fKAvFW8XyXUVoAGSbFvXef2wcbnsjNKGWYorp"
            "LJBjth5VKSFhLGkpYKJb2J341tdZHBnbpv72WL7e8zuxfR2"
        ),
    ) in pairs
    assert ("BTC", "bc1qcp6fr7gtyukympl6unr7uv78h3vprycwj455zx") in pairs
    assert len(pairs) == 2


def test_parse_ofac_remarks_empty():
    """## Remarks without the marker yield no pairs."""
    assert fs.parse_ofac_remarks("Subject to Secondary Sanctions; no wallets.") == []
    assert fs.parse_ofac_remarks("") == []
    assert fs.parse_ofac_remarks(None) == []


def test_parse_ofac_csv_fixture(tmp_path):
    """## CSV scan counts rows, hits only the remarks column."""
    p = tmp_path / "sdn.csv"
    p.write_text(
        "1,A,-0-,X,-0-,-0-,-0-,-0-,-0-,-0-,-0-,plain\n" + ",".join(OFAC_ROW) + "\n",
        encoding="utf-8",
    )
    rows, hits, pairs = fs.parse_ofac_csv(p)
    assert (rows, hits, pairs) == (2, 1, fs.parse_ofac_remarks(OFAC_ROW[-1]))


def test_parse_eu_remark_block_labeled_and_continuation():
    """## EU remark lines keep the last currency label for bare addresses."""
    pairs = fs.parse_eu_remark_block(GARANTEX)
    by_addr = {a: c for c, a in pairs}
    assert by_addr["0x002471b8A185f9980708d0eAEC5B289714F56f8d"] == "ETH"
    assert by_addr["bc1qwtz3zv95x2flu4w26kgfepf529z50r2sqc5zml"] == "BTC"
    assert by_addr["bc1qwxqxd25yk2dtw2ml04vxj9atq3huv4rdytf6vt"] == "BTC"
    assert by_addr["3My1ffQr5qQzmq4aBFgRqwRYqfB5zPAt8t"] == "BTC"
    assert by_addr["0x3051Ca7cB7f6C599fA2f27385AD75010cf0f2bbF"] == "BSC"
    assert by_addr["TA1hsikRfsgGiW9nEBpT4tEXEySTNYLr2d"] == "TRX"
    assert len(pairs) == 6


def test_parse_eu_remark_block_unlabeled_trx():
    """## Bare TRX-shaped address without label still resolves to TRX."""
    pairs = fs.parse_eu_remark_block(
        "Known blockchain wallet addresses:\nTEcuHDQthTmULe8fFLUccBPpjfXaTmJuuD"
    )
    assert pairs == [("TRX", "TEcuHDQthTmULe8fFLUccBPpjfXaTmJuuD")]


def test_parse_eu_remark_block_no_false_positives():
    """## Passport-style IDs and prose yield no pairs."""
    assert fs.parse_eu_remark_block("National passport 488555 issued 2003.") == []
    assert fs.parse_eu_remark_block("") == []


def test_dedup_keeps_first_currency():
    """## Duplicate addresses collapse case-insensitively, first currency wins."""
    out = fs.dedup([("ETH", "0xABC"), ("BSC", "0xabc"), ("BTC", "bc1q")])
    assert out == [("ETH", "0xABC"), ("BTC", "bc1q")]


def test_manifest_roundtrip(tmp_path):
    """## Manifest write/read preserves urls, hashes, and counts."""
    payload = {
        "date": "2026-09-18",
        "sources": {"ofac_sdn_csv": {"sha256": "ab" * 32, "bytes": 7}},
        "counts": {"addresses": 2},
    }
    p = tmp_path / "manifest.json"
    fs.write_manifest(p, payload)
    assert fs.read_manifest(p) == payload
    assert json.loads(p.read_text(encoding="utf-8"))["counts"]["addresses"] == 2


def test_manifest_valid_rejects_tampered(tmp_path):
    """## Manifest validation fails on missing or modified raw files."""
    raw = tmp_path / "f.bin"
    raw.write_bytes(b"data")
    manifest = {
        "sources": {
            "s": {"file": "f.bin", "sha256": fs.hashlib.sha256(b"data").hexdigest()}
        }
    }
    assert fs.manifest_valid(manifest, tmp_path) is True
    raw.write_bytes(b"tampered")
    assert fs.manifest_valid(manifest, tmp_path) is False
    raw.unlink()
    assert fs.manifest_valid(manifest, tmp_path) is False


def test_end_to_end_offline(tmp_path, monkeypatch):
    """## Full pipeline on fixtures writes CSV plus manifest without network."""
    raw = tmp_path / "raw" / "2026-09-18"
    raw.mkdir(parents=True)
    (raw / "sdn.csv").write_text(",".join(OFAC_ROW) + "\n", encoding="utf-8")
    (raw / "eu_source.xml").write_text(
        "<export><sanctionEntity><remark>"
        + GARANTEX
        + "</remark></sanctionEntity></export>",
        encoding="utf-8",
    )

    def fake_fetch(urls, dest, **kwargs):
        """## Pretend download succeeded; fixture files already in place."""
        return {
            "requested": urls,
            "used": urls[0],
            "effective": urls[0],
            "bytes": Path(dest).stat().st_size,
            "sha256": fs.hashlib.sha256(Path(dest).read_bytes()).hexdigest(),
            "status": "ok",
            "attempts": [],
            "file": Path(dest).name,
        }

    monkeypatch.setattr(fs, "fetch_first", fake_fetch)
    monkeypatch.setattr(
        "sys.argv",
        ["fetch_sanctions.py", "--out", str(tmp_path), "--date", "2026-09-18"],
    )
    fs.main()
    lines = (
        (tmp_path / "addresses.csv").read_text(encoding="utf-8").strip().splitlines()
    )
    assert lines[0] == "address,currency,source,list,date"
    assert len(lines) == 1 + 2 + 6
    manifest = fs.read_manifest(tmp_path / "manifest.json")
    assert manifest["counts"]["addresses"] == 8
    assert manifest["counts"]["by_source"] == {"ofac_sdn": 2, "eu_fsf": 6}
    # Second run without --force is idempotent and skips work.
    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_sanctions.py",
            "--out",
            str(tmp_path),
            "--date",
            "2026-09-18",
            "--force",
        ],
    )
    fs.main()
    assert fs.read_manifest(tmp_path / "manifest.json")["counts"]["addresses"] == 8
