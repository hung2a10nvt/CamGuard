# Real-Time Face Recognition & Instant Intruder Alerts 

## Compilation & Build

```bash
git clone [https://github.com/hung2a10nvt/CamGuard.git]
cd CamGuard
```

Download and put [`face_detection_yunet_2022mar.onnx`](https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx) and [`face_recognition_sface_2021dec.onnx`](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) models into your build execution directory.

```bash
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build
./CamGuard.exe
```

## Instruction

Users can enroll themselves or others into a local whitelist via a Live Scan module, instructing the system to bypass recognized faces and focus only on potential threats.

## Testing

If an unauthorized individual (stranger) is detected in the frame, the system automatically captures the event and immediately pushes a photo notification to the owner via Telegram.

<img width="397" height="71" alt="Screenshot 2026-06-01 073623" src="https://github.com/user-attachments/assets/12e8d9c4-c737-4f2b-b9e3-54e4d1f690f8" />
