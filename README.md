<div align="center">

# VeilFrame Android

**High-Performance On-Device Neural Super-Resolution, Computer Vision & Privacy Engine**

[![Android CI](https://github.com/sahir247/veilframe_android/actions/workflows/android-ci.yml/badge.svg)](https://github.com/sahir247/veilframe_android/actions/workflows/android-ci.yml)
[![Platform](https://img.shields.io/badge/Platform-Android%208.0%2B%20(API%2026%2B)-brightgreen.svg)](https://android.com)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3%20Compute-red.svg)](https://www.vulkan.org)
[![NCNN](https://img.shields.io/badge/NCNN-20241226-blue.svg)](https://github.com/Tencent/ncnn)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.14.0%20Custom-orange.svg)](https://opencv.org)
[![Page Size](https://img.shields.io/badge/Memory-16%20KB%20Page%20Aligned-purple.svg)](https://developer.android.com/guide/practices/page-sizes)

</div>

---

## Overview

**VeilFrame Android** is a production-grade, privacy-first Android application designed for high-throughput, on-device neural media processing. Operating completely offline without cloud dependencies, VeilFrame delivers desktop-class AI super-resolution, optical flow analysis, face detection, and mathematical vector QR generation directly on mobile silicon.

---

## Core Subsystems

### 1. Neural Super-Resolution Engine (NCNN + Vulkan)
- **Multi-Model Neural Pipeline**:
  - `Real-ESRGAN AnimeVideoV3 2×` (Compact SRVGGNet, line-art & UI)
  - `Real-ESRGAN AnimeVideoV3 4×` (Compact SRVGGNet, 4× line-art & graphics)
  - `Real-ESRGAN x4plus` (Deep 23-block RRDBNet, photo & natural textures)
  - Algorithmic fallbacks: Lanczos-3, Bicubic Spline, Nearest Neighbor
- **Hardware Acceleration**: Low-latency Vulkan 1.3 compute pipelines optimized for Qualcomm Adreno 800-series (Snapdragon 8 Elite / Adreno 829) and ARM Mali GPUs.
- **Streaming Tile Inference**: Dynamic memory-aware scanline processing that decomposes massive multi-megapixel images into overlapped tiles with zero visible boundary seams.
- **16 KB Page Size Alignment**: Fully compliant with Android 15+ 16 KB memory page boundaries (`ps16k`).
- **Foreground Service Architecture**: Continuous, cancelable background processing via `UpscaleForegroundService` with dynamic notification progress, wakelock safety, and immediate tempfile unlinking on cancellation.

### 2. Computer Vision Subsystem (OpenCV 4.14.0 Custom)
- In-tree, custom-compiled OpenCV 4.14.0 Android SDK with WeChatQRCode integration.
- Accelerated face detection, morphological contour extraction, and optical flow motion estimation.
- Strict native governance ratchets to prevent memory leaks and track native handle lifecycles.

### 3. QR Code Subsystem (EFQRCode Parity)
- Bit-for-bit mathematical parity with upstream **EFQRCode 7.0.3**.
- Canonical Geometry Intermediate Representation (IR) shared across Android Canvas and SVG backends.
- Full support for multi-directional linear gradients, custom finder patterns, embedded logos, and animated SVG generation.

### 4. Modern Material 3 Expressive UI
- Crafted with Android Jetpack Material 3 Expressive components.
- Responsive layout adapting from compact portrait phones to foldable devices and tablets.
- Real-time processing analytics, throughput telemetry (MP/s), and memory profiles (PSS).

---

## Repository Structure

```text
veilframe_android/
├── android/
│   ├── app/                      # VeilFrame main application module
│   │   ├── src/main/cpp/         # Native C++ NCNN inference engine & JNI bridge
│   │   ├── src/main/java/        # Kotlin application, UI, services & domain logic
│   │   ├── src/main/res/         # Material 3 Expressive resources & layouts
│   │   ├── src/test/             # Fast JVM unit test suite (1,129 tests)
│   │   └── src/androidTest/      # On-device Vulkan & NCNN validation matrices
│   ├── opencv-sdk/               # In-tree OpenCV 4.14.0 Android library
│   ├── scripts/                  # OpenCV build & CV governance verification scripts
│   ├── build.gradle.kts          # Top-level Gradle build configuration
│   ├── settings.gradle.kts       # Module declarations
│   └── gradlew / gradlew.bat     # Gradle wrapper scripts
├── docs/                         # Technical documentation & engineering audits
│   ├── ANDROID_NCNN_UPSCALER_DEVICE_VALIDATION.md   # Hardware & emulator benchmark gate
│   ├── AI_UPSCALE_MODEL_CATALOG.md                  # Model weights & parameter catalog
│   ├── M3_EXPRESSIVE_DESIGN_SYSTEM.md               # UI design system specification
│   ├── realsr_vs_matuncnn_deep_architecture_audit.md# Deep architecture audit
│   └── device_validation_artifacts/                 # Benchmark output sample assets
├── .github/workflows/            # Continuous Integration & release automation
│   ├── android-ci.yml            # Automated build, unit tests & 16 KB alignment check
│   └── build-opencv-sdk.yml      # Custom OpenCV SDK compilation workflow
├── AGENTS.md                     # Engineering instructions, parity & testing protocols
├── gradlew / gradlew.bat         # Root invocation wrappers
└── README.md                     # Project overview & documentation
```

---

## Building & Testing

### Prerequisites
- **JDK**: Java Development Kit 17 (Eclipse Temurin or OpenJDK recommended)
- **Android SDK**: API Level 35 (compileSdk 35, minSdk 26, targetSdk 35)
- **Android NDK**: Version `26.1.10909125` or `27.2.12479018`
- **CMake**: Version `3.22.1` or higher

### Building the Project
From the repository root:
```powershell
# Windows
.\gradlew.bat :app:assembleDebug

# Linux / macOS
./gradlew :app:assembleDebug
```

Or from within the `android/` subfolder:
```bash
cd android
./gradlew :app:assembleDebug
```

The resulting debug APK is generated at:
`android/app/build/outputs/apk/debug/app-debug.apk`

### Running the Unit Test Suite
```powershell
.\gradlew.bat :app:testDebugUnitTest
```
*Current test suite: **1,129 tests passed (0 failures, 0 errors, 0 skipped)**.*

### Running Instrumented On-Device Validation
With an ADB device or emulator connected:
```powershell
# Run the 18-case comprehensive upscaler test matrix
adb shell am instrument -w -r -e class com.veilframe.app.upscale.NcnnExtendedEmulatorMatrixTest com.veilframe.app.debug.test/androidx.test.runner.AndroidJUnitRunner
```

---

## Benchmark & Hardware Gate Highlights

| Platform / SoC | Accelerator | Latency (1280×960 -> 2560×1920) | Throughput | Stability |
|---|---|---|---|---|
| **Qualcomm Snapdragon 8 Elite** | Adreno 829 Vulkan 1.3 | **1.74 s** | **2.82 MP/s** | 100% Solid |
| **Android 17 Emulator (x86_64)**| GFXStream Vulkan Bridge | **5.19 s** | **0.95 MP/s** | 100% Solid |

*For full benchmark details, visual inspection audits, and numerical stability matrices, see [`docs/ANDROID_NCNN_UPSCALER_DEVICE_VALIDATION.md`](docs/ANDROID_NCNN_UPSCALER_DEVICE_VALIDATION.md).*

---

## Documentation Index

- [Android NCNN Device Validation Gate](docs/ANDROID_NCNN_UPSCALER_DEVICE_VALIDATION.md)
- [AI Upscale Model Catalog](docs/AI_UPSCALE_MODEL_CATALOG.md)
- [RealSR vs MatUNCNN Deep Architecture Audit](docs/realsr_vs_matuncnn_deep_architecture_audit.md)
- [Material 3 Expressive Migration Plan](docs/M3_EXPRESSIVE_MIGRATION_PLAN.md)
- [Engineering Instructions & Parity Rules (AGENTS.md)](AGENTS.md)

---

## License

This project is licensed under the Apache License 2.0. Third-party components (NCNN, OpenCV, Real-ESRGAN models) are subject to their respective open-source licenses.
