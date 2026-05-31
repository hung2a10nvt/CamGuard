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
