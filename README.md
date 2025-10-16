# Crypto project 2025

## Install

1. Python 3.11+ and dependencies (They are located in `depend.cmd`).
2. Confirm you have a working camera.

## Enroll → make a QR

* **Password mode**

  ```
  python make_qr.py password
  ```

  Prompts twice, saves **certificate_pw_qr.png**. 
* **Face via webcam (recommended)**

  ```
  python make_qr.py face_cam
  ```

  Saves **enrolled_face_embedding.npy** and **certificate_face_qr.png**. 
* **Face from image**

  ```
  python make_qr.py face path\to\face.jpg
  ```

  Saves **certificate_face_qr.png**. 

> The QR stores only `{salt, kdf_params, nonce, ciphertext, meta}`—no plaintext or keys. 

## Unlock

```
python unlock_camera.py
```

* Show the QR to the webcam.
* If it’s a **password** QR → you’ll be prompted; correct password decrypts. 
* If it’s a **face** QR → requires **enrolled_face_embedding.npy** on this machine, checks similarity (default **0.95**) and then decrypts. 

## Notes / Troubleshooting

* “Missing enrolled_face_embedding.npy” → run `python make_qr.py face_cam` on this machine. 
* Adjust similarity threshold in `unlock_camera.py` if needed (line with `cos_sim(... ) < 0.95`). 
