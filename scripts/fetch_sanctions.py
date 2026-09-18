"""## Fetch OFAC/EU sanctions sources and extract crypto anchor addresses.

Usage: python3 scripts/fetch_sanctions.py [--out data/sanctions] [--force]
"""

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Spillety-KYT/2.0"}

OFAC_SDN_URLS = [
    "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV",
    "https://www.treasury.gov/ofac/downloads/sdn.csv",
]
EU_XML_URLS = [
    # Official EU FSF endpoint first; token-gated (403 without token), kept to
    # record the failure honestly in the manifest before using the mirror.
    "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content",
    "https://data.opensanctions.org/datasets/latest/eu_fsf/source.xml",
]

OFAC_RE = re.compile(
    r"(?:alt\.\s*)?Digital Currency Address\s*-\s*([A-Za-z]{2,10})\s+([A-Za-z0-9]+)"
)
EU_LABEL_RE = re.compile(
    r"\b(BTC|XBT|ETH|BSC|TRX|XMR|LTC|BCH|DOGE|XRP|USDT|USDC|SOL|BNB|DASH|ZEC|ETC|ARB)\s*:\s*([A-Za-z0-9]+)"
)
EU_REMARK_RE = re.compile(r"<remark>(.*?)</remark>", re.DOTALL)
EU_ENTITY_RE = re.compile(r"<sanctionEntity\b")
BARE_PATS = (  # (compiled, currency, needs_label); loose BTC pattern needs a label
    (re.compile(r"0x[a-fA-F0-9]{40}"), "ETH", False),
    (re.compile(r"bc1[a-z0-9]{25,62}"), "BTC", False),
    (re.compile(r"\bT[A-Za-z1-9]{33}\b"), "TRX", False),
    (re.compile(r"\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b"), "XMR", False),
    (re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"), "BTC", True),
)
CURRENCY_FIX = {"XBT": "BTC"}


def normalize_currency(code):
    """## Map source tickers to canonical codes (XBT -> BTC), uppercased."""
    code = code.upper()
    return CURRENCY_FIX.get(code, code)


def download(url, dest, timeout=120):
    """## Stream URL to dest with UA header, return (bytes, sha256, final_url)."""
    req = urllib.request.Request(url, headers=UA)
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        final = resp.geturl()
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest(), final


def fetch_first(urls, dest, reuse_ok=True):
    """## Try URLs in order, return manifest entry; raise on total failure."""
    if reuse_ok and dest.exists() and dest.stat().st_size > 0:
        return {
            "requested": urls,
            "used": "vendored",
            "effective": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "status": "reused",
            "attempts": [],
        }
    attempts = []
    for url in urls:
        try:
            size, sha, final = download(url, dest)
            return {
                "requested": urls,
                "used": url,
                "effective": final,
                "bytes": size,
                "sha256": sha,
                "status": "ok",
                "attempts": attempts,
            }
        except Exception as exc:  # noqa: BLE001 - recorded into manifest
            attempts.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    raise RuntimeError(f"all sources failed for {dest}: {attempts}")


def parse_ofac_remarks(text):
    """## Extract (currency, address) pairs from one SDN remarks cell."""
    return [
        (normalize_currency(cur), addr) for cur, addr in OFAC_RE.findall(text or "")
    ]


def parse_ofac_csv(path):
    """## Scan SDN CSV remarks column, return (rows, hit_rows, pairs)."""
    pairs, rows, hits = [], 0, 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            rows += 1
            if not row:
                continue
            found = parse_ofac_remarks(row[-1])
            if found:
                hits += 1
                pairs.extend(found)
    return rows, hits, pairs


def extract_eu_remarks(xml_text):
    """## Return free-text remark blocks where EU hides wallet addresses."""
    return EU_REMARK_RE.findall(xml_text)


def parse_eu_remark_block(block):
    """## Labeled `CUR: addr` lines plus continuation bare-pattern matches."""
    pairs, current = [], None
    for line in block.splitlines():
        labels = EU_LABEL_RE.findall(line)
        if labels:
            current = normalize_currency(labels[-1][0])
            pairs.extend((normalize_currency(c), a) for c, a in labels)
            continue
        for pat, cur, needs_label in BARE_PATS:
            if needs_label and current != cur:
                continue
            for addr in pat.findall(line):
                pairs.append((current or cur, addr))
    return pairs


def parse_eu_xml(path):
    """## Count entities, mine remark blocks, return (entities, remarks, pairs)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    remarks = extract_eu_remarks(text)
    pairs = [p for block in remarks for p in parse_eu_remark_block(block)]
    return len(EU_ENTITY_RE.findall(text)), len(remarks), pairs


def dedup(pairs):
    """## Deduplicate on lowercased address, keep first-seen currency."""
    seen = {}
    for cur, addr in pairs:
        seen.setdefault(addr.lower(), (cur, addr))
    return list(seen.values())


def write_manifest(path, payload):
    """## Serialize fetch manifest (urls, hashes, counts) as JSON."""
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_manifest(path):
    """## Load fetch manifest back into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_valid(manifest, raw_dir):
    """## True when recorded raw files still exist with matching sha256."""
    try:
        for entry in manifest["sources"].values():
            p = Path(raw_dir) / entry["file"]
            if (
                not p.exists()
                or hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]
            ):
                return False
        return True
    except (KeyError, OSError):
        return False


def main():
    """## Fetch sources unless manifest matches, extract pairs, dump CSV+manifest."""
    args = argparse.ArgumentParser()
    args.add_argument("--out", default="data/sanctions")
    args.add_argument("--force", action="store_true")
    args.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    opt = args.parse_args()

    out = Path(opt.out)
    raw_dir = out / "raw" / opt.date
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    csv_path = out / "addresses.csv"

    if not opt.force and manifest_path.exists() and csv_path.exists():
        manifest = read_manifest(manifest_path)
        if manifest.get("date") == opt.date and manifest_valid(manifest, raw_dir):
            print(
                f"up to date ({csv_path}, {manifest['counts']['addresses']} addresses), skip; use --force"
            )
            return

    sdn_path, eu_path = raw_dir / "sdn.csv", raw_dir / "eu_source.xml"
    ofac_entry = fetch_first(OFAC_SDN_URLS, sdn_path, reuse_ok=not opt.force)
    eu_entry = fetch_first(EU_XML_URLS, eu_path, reuse_ok=not opt.force)
    ofac_entry["file"], eu_entry["file"] = sdn_path.name, eu_path.name

    ofac_rows, ofac_hits, ofac_pairs = parse_ofac_csv(sdn_path)
    eu_entities, eu_remarks, eu_pairs = parse_eu_xml(eu_path)

    rows = [(a, c, "ofac_sdn", "OFAC-SDN", opt.date) for c, a in dedup(ofac_pairs)]
    rows += [(a, c, "eu_fsf", "EU", opt.date) for c, a in dedup(eu_pairs)]
    by_cur, by_src = {}, {}
    for _, cur, src, _, _ in rows:
        by_cur[cur] = by_cur.get(cur, 0) + 1
        by_src[src] = by_src.get(src, 0) + 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["address", "currency", "source", "list", "date"])
        w.writerows(rows)

    manifest = {
        "date": opt.date,
        "sources": {"ofac_sdn_csv": ofac_entry, "eu_fsf_xml": eu_entry},
        "counts": {
            "ofac_rows": ofac_rows,
            "ofac_rows_with_crypto": ofac_hits,
            "ofac_pairs": len(ofac_pairs),
            "eu_entities": eu_entities,
            "eu_remarks": eu_remarks,
            "eu_pairs": len(eu_pairs),
            "addresses": len(rows),
            "by_currency": by_cur,
            "by_source": by_src,
        },
        "notes": (
            "EU direct FSF endpoint needs a token (403); using the "
            "OpenSanctions mirror of the genuine EU FSF XML. EU XML has no "
            "dedicated crypto field; wallets mined from free-text remarks."
        ),
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"saved -> {csv_path} ({len(rows)}), raw in {raw_dir}")


if __name__ == "__main__":
    main()
