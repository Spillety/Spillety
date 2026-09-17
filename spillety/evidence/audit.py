import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class WormLog:
    def __init__(self, genesis: str = "00" * 32):
        self._genesis = genesis
        self._entries: list[dict] = []
        self._prev = genesis

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    @property
    def worm_root(self) -> str:
        return self._entries[-1]["entry_hash"] if self._entries else self._genesis

    # alias head
    @property
    def head(self) -> str:
        return self.worm_root

    def append(self, event_type: str, alert_id: str, actor: str, details: dict | None = None) -> dict:
        entry = {
            "index": len(self._entries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "alert_id": alert_id,
            "actor": actor,
            "details": details or {},
            "prev_hash": self._prev,
        }
        canon = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        entry_hash = hashlib.sha256(canon.encode()).hexdigest()
        entry["entry_hash"] = entry_hash
        self._entries.append(entry)
        self._prev = entry_hash
        return entry

    def verify_chain(self) -> bool:
        prev = self._genesis
        for e in self._entries:
            # recompute hash without entry_hash field
            tmp = {k: v for k, v in e.items() if k != "entry_hash"}
            canon = json.dumps(tmp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            h = hashlib.sha256(canon.encode()).hexdigest()
            if h != e["entry_hash"] or e["prev_hash"] != prev:
                return False
            prev = e["entry_hash"]
        return True

    def anchor(self, path: str | Path, merkle_root: str, extra: dict | None = None) -> Path:
        payload = {
            "merkle_root": merkle_root,
            "root_b64": base64.b64encode(bytes.fromhex(merkle_root)).decode() if len(merkle_root) == 64 else merkle_root,
            "worm_head": self.worm_root,
            "n_entries": len(self._entries),
            "anchored_at": datetime.now(timezone.utc).isoformat(),
            "proxy": "OpenTimestamps -> Bitcoin (mock, OTS stamp of root)",
        }
        if extra:
            payload.update(extra)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return p

    def to_dicts(self) -> list[dict]:
        return self.entries


# functional helpers for notebook-compat
_default_log: WormLog | None = None


def get_default_log() -> WormLog:
    global _default_log
    if _default_log is None:
        _default_log = WormLog()
    return _default_log


def worm_append(event_type: str, alert_id: str, actor: str, details: dict | None = None, log: WormLog | None = None) -> dict:
    target = log if log is not None else get_default_log()
    return target.append(event_type, alert_id, actor, details)


def worm_root(log: WormLog | None = None) -> str:
    target = log if log is not None else get_default_log()
    return target.worm_root


def verify_worm(log: WormLog | None = None) -> bool:
    target = log if log is not None else get_default_log()
    return target.verify_chain()


if __name__ == "__main__":
    from spillety.evidence.worm import build_evidence, evidence_hash, merkle_root, merkle_proof, verify_proof, sign_evidence, verify_evidence

    DEMO_KEY = b"spillety-demo-key-32-bytes!!1234"
    log = WormLog()
    # create 5 alerts lifecycle
    evs = [
        build_evidence(
            f"ALT-{i:08d}-000",
            0.9 - i * 0.05,
            {"distance": float(i), "source": "edgelist"},
            {"edge": f"{i}->{i+1}", "effect": 0.5},
            {"feat_1": 0.1},
            {"model_version": "v1", "date": "2026-09-16", "source": "elliptic_raw"},
        )
        for i in range(5)
    ]
    for ev in evs:
        log.append("create", ev["alert_id"], "model@spillety", {"risk_score": ev["risk_score"]})
        log.append("view", ev["alert_id"], "analyst@spillety", {"distance": ev["anchors"]["distance"]})
    assert log.verify_chain()
    assert len(log.entries) == 10
    # tamper detection
    log._entries[3]["details"] = {"tampered": True}
    assert not log.verify_chain(), "tamper must break chain"
    # restore
    log = WormLog()
    for ev in evs:
        log.append("create", ev["alert_id"], "model@spillety", {})
    assert log.verify_chain()

    # worm_root anchoring with merkle
    leaves = [evidence_hash(e) for e in evs]
    root = merkle_root(leaves + leaves[:95]) if len(leaves) < 100 else merkle_root(leaves)
    # test N=100 path
    full_leaves = [evidence_hash(build_evidence(f"A-{i}", 0.6, {"distance": 1.0, "source": "edgelist"}, {"edge": "a->b", "effect": 0.5}, {}, {"model_version": "v1", "date": "2026-09-16", "source": "elliptic_raw"})) for i in range(100)]
    import math

    from spillety.evidence.worm import build_merkle

    levels, r100 = build_merkle(full_leaves)
    p7 = merkle_proof(levels, 0)
    assert len(p7) == 7 and verify_proof(full_leaves[0], p7, r100)
    print(f"ok: worm chain {len(log.entries)} entries head={log.worm_root[:12]}... proof N=100 size 7 verified")
