@echo off
setlocal
REM ==== Adjust this path if your Python 3.11 is elsewhere ====
set "PY=C:\Users\snnyv\AppData\Local\Programs\Python\Python311\python.exe"

echo [1/5] Upgrading pip tooling...
"%PY%" -m pip install -i https://pypi.org/simple --upgrade pip wheel setuptools

echo [2/5] Core deps (binary wheels only)...
"%PY%" -m pip install -i https://pypi.org/simple --only-binary=:all: ^
 numpy pillow opencv-python qrcode cryptography pyzbar click colorama

echo [3/5] Windows dlib (prebuilt)...
"%PY%" -m pip install -i https://pypi.org/simple --only-binary=:all: dlib-bin==19.24.6

echo [4/5] Face model pack (no binary needed)...
"%PY%" -m pip install -i https://pypi.org/simple face_recognition_models==0.3.0

echo [5/5] face-recognition (stop it from re-solving dlib/models)...
"%PY%" -m pip install -i https://pypi.org/simple face-recognition==1.3.0 --no-deps

echo.
"%PY%" -c "import dlib,face_recognition,face_recognition_models,cv2,pyzbar,qrcode,cryptography,numpy; print('ok')"

endlocal
