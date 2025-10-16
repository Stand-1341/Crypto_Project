# skeleton.py
import os, json, base64, numpy as np
from typing import Dict
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-norm embedding")
    return v / n

def quantize_embedding(v, q: int) -> bytes:

    import numpy as np
    if q < 0 or q > 6:
        raise ValueError("q out of range (0..6)")
    v = np.asarray(v, dtype=np.float32)
    return np.round(v, decimals=q).astype(np.float32, copy=False).tobytes()

def derive_key_from_embedding(embedding, salt: bytes, params: dict) -> bytes:

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    name = str(params.get("name", ""))
    if name != "HKDF-SHA256":
        raise ValueError("Wrong params for derive_key_from_embedding")

    length = int(params.get("length", 32))
    if length not in (16, 32):
        raise ValueError("HKDF length must be 16 or 32")

    info = str(params.get("info", "face-key-v1")).encode()
    q = int(params.get("q", 3))

    # Use your existing normalize_embedding()
    v = normalize_embedding(embedding)
    ikm = quantize_embedding(v, q)

    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    return hkdf.derive(ikm)
def derive_key_from_password(password: str, salt: bytes, params: dict) -> bytes:

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    name = str(params.get("name", ""))
    if name != "PBKDF2-HMAC-SHA256":
        raise ValueError("Wrong params for derive_key_from_password")

    length = int(params.get("length", 32))
    if length not in (16, 32):
        raise ValueError("PBKDF2 length must be 16 or 32")

    iters = int(params.get("iters", 150_000))
    if iters < 50_000:
        raise ValueError("PBKDF2 iterations too low; set >= 50k")

    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=length, salt=salt, iterations=iters)
    return kdf.derive(password.encode("utf-8"))


def aes_gcm_encrypt(key: bytes, plaintext: bytes) -> Dict[str, bytes]:
    if len(key) not in (16, 32):
        raise ValueError("Key must be 16 or 32 bytes for AES-128/256")
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext, associated_data=None)
    return {"nonce": nonce, "ct": ct}

def aes_gcm_decrypt(key: bytes, nonce: bytes, ct: bytes) -> bytes:
    aes = AESGCM(key)
    return aes.decrypt(nonce, ct, associated_data=None)

def pack_qr_payload(obj: Dict[str, bytes | str | Dict]) -> str:

    enc = {}
    for k, v in obj.items():
        if isinstance(v, (bytes, bytearray)):
            enc[k] = base64.b64encode(v).decode()
        else:
            enc[k] = v
    blob = json.dumps(enc).encode()
    return base64.b64encode(blob).decode()

def unpack_qr_payload(b64s: str) -> Dict[str, bytes | str | Dict]:
    data = json.loads(base64.b64decode(b64s).decode())
    out = {}
    for k, v in data.items():
        if isinstance(v, str) and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in v):
            try:
                out[k] = base64.b64decode(v)
                continue
            except Exception:
                pass
        out[k] = v
    return out
