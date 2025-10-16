# make_qr.py
import os
import qrcode, numpy as np
from skeleton import (derive_key_from_embedding, derive_key_from_password,
                      aes_gcm_encrypt, pack_qr_payload)


def compute_embedding_from_image(path: str):
    import numpy as np, face_recognition
    FACE_MODEL = "hog"
    UPSAMPLE   = 1
    JITTERS    = 1

    img = face_recognition.load_image_file(path)
    locs = face_recognition.face_locations(img, number_of_times_to_upsample=UPSAMPLE, model=FACE_MODEL)
    if not locs:
        raise ValueError("No face found in enrollment image")

    def area(box):
        t, r, b, l = box
        return max(0, b - t) * max(0, r - l)

    loc = max(locs, key=area)
    encs = face_recognition.face_encodings(img, known_face_locations=[loc], num_jitters=JITTERS, model="small")
    if not encs:
        raise ValueError("Failed to compute face embedding")

    return np.array(encs[0], dtype=np.float32)
def build_face_kdf_params(q: int = 3) -> dict:
    return {
        "name": "HKDF-SHA256",
        "length": 32,
        "info": "face-key-v1",
        "q": int(q),
        "model": "hog",
        "upsample": 1,
        "jitters": 1,
    }
def compute_embedding_from_webcam(samples: int = 3):

    import cv2, time, numpy as np, face_recognition
    FACE_MODEL, UPSAMPLE, JITTERS = "hog", 1, 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")
    t0 = time.time()
    while time.time() - t0 < 0.5:
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture frame from camera")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=UPSAMPLE, model=FACE_MODEL)
    if not locs:
        raise ValueError("No face detected in camera frame")

    def area(box): t, r, b, l = box; return max(0, b - t) * max(0, r - l)
    loc = max(locs, key=area)

    encs = []
    for _ in range(samples):
        e = face_recognition.face_encodings(rgb, known_face_locations=[loc], num_jitters=JITTERS, model="small")
        if e: encs.append(e[0])
    if not encs:
        raise ValueError("Failed to compute face embedding from webcam")

    emb = np.array(np.mean(encs, axis=0), dtype=np.float32)
    np.save("enrolled_face_embedding.npy", emb, allow_pickle=False)
    print("Saved enrolled embedding to enrolled_face_embedding.npy")
    return emb


def build_pw_kdf_params(iters: int = 150_000) -> dict:

    return {
        "name": "PBKDF2-HMAC-SHA256",
        "length": 32,
        "iters": int(iters),
    }


if __name__ == "__main__":
    import sys
    from getpass import getpass
    import os
    from getpass import getpass
    plaintext = getpass("Secret to protect (UTF-8): ").encode("utf-8")
    salt = os.urandom(16)

    mode = sys.argv[1] if len(sys.argv) > 1 else "face"

    if mode == "face":
        img_path = sys.argv[2] if len(sys.argv) > 2 else "reference_face.jpg"
        emb = compute_embedding_from_image(img_path)
        kdf_params = build_face_kdf_params(q=3)
        key = derive_key_from_embedding(emb, salt, kdf_params)
        out_png = "certificate_face_qr.png"

    elif mode == "face_cam":
        emb = compute_embedding_from_webcam()
        kdf_params = build_face_kdf_params(q=3)
        key = derive_key_from_embedding(emb, salt, kdf_params)
        out_png = "certificate_face_qr.png"


    elif mode == "password":
        pw1 = getpass("New password: ")
        pw2 = getpass("Repeat password: ")
        if pw1 != pw2:
            raise SystemExit("Passwords do not match.")
        kdf_params = build_pw_kdf_params(iters=150_000)
        key = derive_key_from_password(pw1, salt, kdf_params)
        out_png = "certificate_pw_qr.png"

    else:
        raise SystemExit("Usage: python make_qr.py face <image> | face_cam | password")

    enc = aes_gcm_encrypt(key, plaintext)
    payload = {
        "salt": salt,
        "kdf_params": kdf_params,
        "nonce": enc["nonce"],
        "ciphertext": enc["ct"],
        "meta": {"alg": "AES-GCM", "ver": 1},
    }

    b64blob = pack_qr_payload(payload)

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_Q)
    qr.add_data(b64blob)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(out_png)
    print(f"Saved QR to {out_png}")
