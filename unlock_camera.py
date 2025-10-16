# unlock_camera.py
import cv2, numpy as np
from pyzbar import pyzbar
from skeleton import (unpack_qr_payload, derive_key_from_embedding,
                      derive_key_from_password, aes_gcm_decrypt)
import os

def cos_sim(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
def capture_live_embedding(samples: int = 5):
    import cv2, time
    encs = []
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    t0 = time.time()
    while time.time() - t0 < 0.5:
        cap.read()

    while len(encs) < samples:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            e = compute_embedding_from_frame(frame)
            encs.append(e.astype(np.float32))
        except Exception:
            continue

    cap.release()
    if not encs:
        raise RuntimeError("Failed to compute live embedding from camera")
    return np.mean(np.stack(encs, axis=0), axis=0).astype(np.float32)

def compute_embedding_from_frame(frame):
    import numpy as np, cv2, face_recognition
    FACE_MODEL = "hog"   # must match enrollment
    UPSAMPLE   = 1
    JITTERS    = 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=UPSAMPLE, model=FACE_MODEL)
    if not locs:
        raise ValueError("No face detected in camera frame")

    def area(box):
        t, r, b, l = box
        return max(0, b - t) * max(0, r - l)

    loc = max(locs, key=area)
    encs = face_recognition.face_encodings(rgb, known_face_locations=[loc], num_jitters=JITTERS, model="small")
    if not encs:
        raise ValueError("Failed to compute face embedding from frame")

    return np.array(encs[0], dtype=np.float32)


def scan_qr_with_webcam():
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    qr_b64 = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cv2.imshow("Scan QR (press q to cancel)", frame)

            maybe = read_qr_from_frame(frame)
            if maybe:
                qr_b64 = maybe
                break

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return qr_b64

def read_qr_from_frame(frame):
    for b in pyzbar.decode(frame):
        s = b.data.decode("utf-8")
        if s:
            return s
    return None

if __name__ == "__main__":
    qr_b64 = scan_qr_with_webcam()
    if not qr_b64:
        print("No QR read. Exiting.")
        raise SystemExit(0)

    payload = unpack_qr_payload(qr_b64)
    name = str(payload["kdf_params"].get("name", ""))

    if name == "PBKDF2-HMAC-SHA256":
        from getpass import getpass
        pw = getpass("Password: ")
        key = derive_key_from_password(pw, payload["salt"], payload["kdf_params"])


    elif name == "HKDF-SHA256":

        enroll_path = "enrolled_face_embedding.npy"

        if not os.path.exists(enroll_path):
            raise SystemExit("Missing enrolled_face_embedding.npy on this machine.")

        enrolled = np.load(enroll_path).astype(np.float32)


        live = capture_live_embedding(samples=5)

        if cos_sim(live, enrolled) < 0.95:
            print("Face check failed.")

            raise SystemExit(2)
        key = derive_key_from_embedding(enrolled, payload["salt"], payload["kdf_params"])


    else:
        raise SystemExit(f"Unsupported KDF mode: {name}")

    pt = aes_gcm_decrypt(key, payload["nonce"], payload["ciphertext"])
    print("UNLOCK SUCCESS. Plaintext:", pt.decode(errors="ignore"))
