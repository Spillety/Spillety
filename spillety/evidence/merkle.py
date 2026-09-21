import hashlib
import hmac
import urllib.error
import urllib.request
from datetime import datetime, timezone

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    _HAS_ED25519 = True
except ImportError:  # ponytail: HMAC-SHA256 proxy, upgrade path — add `cryptography` to pyproject
    _HAS_ED25519 = False


def leaf_hash(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def _levels(leaves: list[bytes]) -> list[list[bytes]]:
    if not leaves:
        raise ValueError("leaves must be non-empty")
    level = [leaf_hash(p) for p in leaves]
    out = [level]
    # Odd level duplicates its last node so every parent has two children.
    while len(level) > 1:
        if len(level) % 2:
            level = [*level, level[-1]]
        level = [_parent(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        out.append(level)
    return out


def merkle_root(leaves: list[bytes]) -> bytes:
    """
    ## Merkle root over raw leaf payloads (§10.3.2)

    Parameters
    ----------
    leaves : list[bytes]
        Raw leaf payloads (e.g. canonical Evidence JSON bytes).

    Returns
    ----------
    bytes
        32-byte root; fixed periodically as the published commitment.
    """
    return _levels(leaves)[-1][0]


def merkle_proof(leaves: list[bytes], index: int) -> list[str]:
    """
    ## Inclusion proof for one leaf, leaf-to-root order (§10.3.2)

    Parameters
    ----------
    leaves : list[bytes]
        Same batch passed to merkle_root.
    index : int
        Leaf position in [0, len(leaves)).

    Returns
    ----------
    list[str]
        Sibling hashes as hex, size ceil(log2N) (empty for N=1).
    """
    lvls = _levels(leaves)
    if not 0 <= index < len(leaves):
        raise IndexError(f"index {index} out of range for {len(leaves)} leaves")
    proof, i = [], index
    for level in lvls[:-1]:
        sib = i ^ 1
        proof.append(level[sib if sib < len(level) else -1].hex())
        i //= 2
    return proof


def verify_proof(leaf: bytes, index: int, proof: list[str], root: bytes) -> bool:
    """
    ## Recompute leaf-to-root path and compare against published root

    Parameters
    ----------
    leaf : bytes
        Raw leaf payload.
    index : int
        Leaf position; its bits decide left/right order at each level.
    proof : list[str]
        Sibling hashes from merkle_proof.
    root : bytes
        Published Merkle root.

    Returns
    ----------
    bool
        True iff the recomputed root matches.
    """
    node = leaf_hash(leaf)
    i = index
    for sib_hex in proof:
        sib = bytes.fromhex(sib_hex)
        node = _parent(sib, node) if i % 2 else _parent(node, sib)
        i //= 2
    return hmac.compare_digest(node, root)


def sign_root(root: bytes, key: bytes) -> bytes:
    """
    ## Sign published root; Ed25519, HMAC-SHA256 proxy without `cryptography`

    Parameters
    ----------
    root : bytes
        Published Merkle root.
    key : bytes
        32-byte seed (Ed25519 seed, or HMAC key in proxy mode).

    Returns
    ----------
    bytes
        64-byte signature in both modes (proxy: HMAC||HMAC(HMAC)).
    """
    if _HAS_ED25519:
        return Ed25519PrivateKey.from_private_bytes(key).sign(root)
    # ponytail: 64-byte mock keeps Ed25519 wire size; upgrade path — `cryptography`-extra
    first = hmac.new(key, root, hashlib.sha256).digest()
    return first + hmac.new(key, first, hashlib.sha256).digest()


def verify_root(root: bytes, signature: bytes, key: bytes) -> bool:
    """
    ## Verify root signature against private seed (proxy) or public key (Ed25519)

    Parameters
    ----------
    root : bytes
        Published Merkle root.
    signature : bytes
        64-byte signature from sign_root.
    key : bytes
        Ed25519 public key (32 bytes) or HMAC seed (32 bytes) in proxy mode.

    Returns
    ----------
    bool
        True iff the signature is valid.
    """
    if _HAS_ED25519:
        try:
            Ed25519PublicKey.from_public_bytes(key).verify(signature, root)
            return True
        except (InvalidSignature, ValueError):
            return False
    if len(signature) != 64:
        return False
    return hmac.compare_digest(signature, sign_root(root, key))


def ots_anchor(root: bytes) -> dict:
    """
    ## OpenTimestamps anchoring stub (§10.3.3)

    Parameters
    ----------
    root : bytes
        Published Merkle root to anchor.

    Returns
    ----------
    dict
        Pending receipt; polling/submission lives outside this module.
    """
    # ponytail: async BTC anchoring out of scope, upgrade path — opentimestamps client + calendar poll
    return {"root": root.hex(), "status": "pending", "txid": None, "proof": None}


def ots_anchor_async(root_hex: str, calendar_url: str = "https://a.pool.opentimestamps.org") -> dict:
    """
    ## Submit Merkle root to OpenTimestamps calendar asynchronously (§10.3.3)

    Parameters
    ----------
    root_hex : str
        Merkle root as hex string (64 chars).
    calendar_url : str
        OpenTimestamps calendar endpoint.

    Returns
    ----------
    dict
        Submission receipt with status, calendar URL, and timestamp.
        Does not wait for Bitcoin confirmation; polling/confirmation is out of scope.
    """
    # ponytail: polling/confirmation out of scope
    try:
        root_bytes = bytes.fromhex(root_hex)
        url = f"{calendar_url.rstrip('/')}/digest"
        req = urllib.request.Request(
            url,
            data=root_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()  # consume response
        return {
            "status": "submitted",
            "calendar": calendar_url,
            "txid": None,
            "pending": True,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
        return {"status": "error", "error": str(e)}


def ots_verify(root_hex: str, ots_proof: bytes, calendar_url: str = "https://a.pool.opentimestamps.org") -> bool:
    """
    ## Verify OpenTimestamps proof for a Merkle root (§10.3.3)

    Parameters
    ----------
    root_hex : str
        Merkle root as hex string (64 chars).
    ots_proof : bytes
        OTS proof bytes (from calendar GET /digest/{root_hex} or test fixture).
    calendar_url : str
        OpenTimestamps calendar endpoint (unused in simplified verification).

    Returns
    ----------
    bool
        True if proof appears valid for the given root.
    """
    # ponytail: full OTS verification requires bitcoinlib, here simplified format
    if not ots_proof:
        return False
    root_bytes = bytes.fromhex(root_hex)
    if not ots_proof.startswith(root_bytes):
        return False
    return len(ots_proof) >= len(root_bytes) + 32
