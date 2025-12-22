# Real-time Edge Overlay (CPU vs CUDA)

## Features
- Input: webcam, video, single image, or synthetic frame (no dataset needed)
- Pipeline: BGR→Gray → Blur → Sobel edges → red overlay
- Outputs: mean latency, p95 latency, FPS
- Optional: save processed frames to disk

## Build

### Requirements
- CMake 3.20+
- OpenCV (C++ dev)
- Optional: CUDA Toolkit (to enable CUDA backend)

### Linux / macOS (CPU only)
```bash
cmake -S . -B build -DENABLE_CUDA=OFF
cmake --build build -j
./build/app --backend cpu --input none
