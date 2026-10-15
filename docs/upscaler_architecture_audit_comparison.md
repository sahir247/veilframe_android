# VeilFrame AI Image Upscaler: In-Depth Competitive Architecture Audit & Modernization Blueprint

**Audit Target Repositories:**
1. [Parasaran-Python/pixelclarity-android](https://github.com/Parasaran-Python/pixelclarity-android) (Kotlin · ONNX Runtime · Tiled Inference)
2. [Celesth/MatUnCNN](https://github.com/Celesth/MatUnCNN) (Kotlin · NCNN · Vulkan · Jetpack Compose)
3. [tumuyan/RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) (Java/C++ · NCNN · Vulkan · Multi-engine CLI/GUI)

**Target Ecosystem Context:**
- **OS Baseline**: Android 8.0 (API 26) through Android 15 / 16 (API 35+ with mandatory 16 KB page-size ELF alignment).
- **Critical Architectural Shift**: **Android 15 has officially shut down / deprecated NNAPI (Neural Networks API)**. Hardware acceleration via NNAPI delegates is no longer viable for new OS releases.
- **Current VeilFrame State**: VeilFrame's existing upscaler suffers from ONNX Runtime CPU/native memory spikes, NNAPI fallback overhead, and complex lifecycle races.

---

## 1. Deep Codebase Inspection of the 3 Repositories

### Repository 1: PixelClarity Android (`Parasaran-Python/pixelclarity-android`)
* **Stack**: Kotlin, ONNX Runtime Mobile (`com.microsoft.onnxruntime:onnxruntime-android:1.30.0`), AndroidX, ViewBinding.
* **Engine Type**: In-process JNI via Microsoft's official ONNX Runtime AAR.
* **Model Formats**: Dynamic ONNX (`.onnx`) hosted on GitHub Releases (`v1.0.0-models`), loaded on-demand via HTTP with SHA-256 integrity verification.

#### Architectural Breakdown:
- **Session Lifecycle**: Handled in `RealESRGANUpscalerEngine.kt`. Initializes `OrtEnvironment` and `OrtSession`. It configures intra-op threads up to 4 and allows selecting delegates (`NPU_NNAPI`, `AUTO`, `CPU`).
- **Tiling Implementation**: 
  - Iterates through the input bitmap using fixed tile dimensions (`128x128` or `256x256`) with a `10px` overlap pad (`tilePad`).
  - Slices RGB channels into an NCHW `FloatBuffer` (`1 x 3 x H x W`), feeds `OnnxTensor` into the session, reads output float values, crops away the padded border, and sets pixels into an in-memory `outputBitmap` via `Bitmap.setPixels()`.
- **Memory Safety & Scalability**:
  - **Fatal Defect**: All processing occurs strictly in **Java Heap RAM**. It allocates `outputBitmap = createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888)`.
  - For a standard 12 MP camera photo upscaled 4× $\to$ **192 MP output**:
    - An uncompressed 192 MP ARGB_8888 bitmap requires **~768 MB of contiguous RAM**.
    - Android app heap limits typically range from **192 MB to 512 MB**.
    - Result: **Instant `OutOfMemoryError` crash** on any photo larger than ~3–4 MP.
- **Android 15+ Compatibility**:
  - Relies heavily on `options.addNnapi()` for NPU acceleration. On Android 15+, NNAPI is deprecated and stripped from active driver pipelines. ONNX Runtime either refuses the delegate or silently falls back to unaccelerated CPU partitioning with severe overhead.

---

### Repository 2: MatUnCNN (`Celesth/MatUnCNN`)
* **Stack**: Kotlin, Jetpack Compose (Material 3), C++ JNI (`matuncnn_launcher.cpp`, `vulkan_hook.cpp`), NCNN (Vulkan), ImageMagick.
* **Engine Type**: **Subprocess Shell Launcher (`ProcessBuilder`)**.
* **Model Formats**: NCNN binary pair (`.param` + `.bin`), downloaded dynamically as a pre-bundled zip (`realsr-assets-v1.0.0-beta.zip`).

#### Architectural Breakdown:
- **Execution Mechanism (`ExecHelper.kt`)**:
  - MatUnCNN is **not** an in-process native library! It extracts precompiled Linux/Android ELF binaries (`realsr-ncnn`, `magick`) to internal storage and executes them via `ProcessBuilder`.
  - **The `noexec` Catastrophe**: Starting in Android 10 (API 29) and enforced strictly in Android 14/15, app internal storage (`/data/data/<package>/`) is mounted with the `noexec` flag. Directly running binaries triggers `execve() failed: EACCES (Permission denied)`.
  - To circumvent Android OS security, MatUnCNN attempts hacks:
    1. Direct execution (fails on Android 10+).
    2. `/proc/self/fd/<fd>` descriptor execution (opening the file and executing its file descriptor).
    3. `/system/bin/linker64 <binary>` invocation.
  - **Security & Play Store Violation**: Google Play Store policy and modern SELinux policies (`untrusted_app` domain) strictly forbid running unbundled downloaded binaries or bypassing `noexec`. Any app with this pattern will be rejected or banned.
- **GPU Acceleration**:
  - Uses Tencent's NCNN Vulkan compute shaders.
  - Bypasses NNAPI entirely, providing genuine GPU acceleration on Adreno and Mali chips.
  - Features an `LD_PRELOAD` hook library (`vulkan_hook.cpp`) to log `vkQueueSubmit failed -4` errors and spoof Vulkan API versions to avoid Adreno black-screen bugs (`bug_buffer_image_load_zero`).

---

### Repository 3: RealSR-NCNN-Android (`tumuyan/RealSR-NCNN-Android`)
* **Stack**: Java, C++ NDK (CMake), OpenCV, LibWebP, Tencent NCNN (Vulkan).
* **Engine Type**: Dual project:
  1. `RealSR-NCNN-Android-CLI`: Compiles standalone command-line executables (`realsr-ncnn`, `waifu2x-ncnn`, `realcugan-ncnn`, `srmd-ncnn`, `Anime4k`, `resize-ncnn`).
  2. `RealSR-NCNN-Android-GUI`: Java wrapper app.
* **Model Formats**: NCNN `.param` / `.bin` and Alibaba MNN (`.mnn`).

#### Architectural Breakdown:
- **Execution Mechanism (`ImageProcessor.java`)**:
  - Exactly like MatUnCNN, the GUI is merely a shell frontend.
  - It runs `ProcessBuilder("sh")`, writes shell commands to standard input (`cd <workingDir>; chmod +x *ncnn; export LD_LIBRARY_PATH=...; ./realsr-ncnn -i in.png -o out.png; exit`), and parses stdout lines for percentage updates.
  - Requires writing input images to disk as PNG/JPG files, waiting for the subprocess to finish writing the output image, and then reloading the output into memory.
  - Heavy disk I/O, flash wear, latency, and identical `noexec` security failures on modern Android versions.
- **The Core C++ Engine (`realsr.cpp`)**:
  - While the GUI architecture is unusable for a production app, its **underlying C++ code (`realsr.cpp` & shaders)** is gold.
  - Implements Nihui's optimized Vulkan pipelines: FP16 packed/storage math, Winograd convolutions, SGEMM matrix multiplication, and spatial tiling with seam-blending overlap.

---

## 2. Reliability & Architecture Scorecard

| Dimension | PixelClarity Android | MatUnCNN | RealSR-NCNN-Android | VeilFrame (Current) |
|---|:---:|:---:|:---:|:---:|
| **Overall Score** | **6.5 / 10** | **3.5 / 10** | **4.0 / 10** | **6.0 / 10** |
| **Architectural Purity** | **8.5 / 10** (In-process JNI) | **2.0 / 10** (Subprocess hack) | **2.5 / 10** (Subprocess shell) | **8.0 / 10** (In-process JNI + Hybrid) |
| **Android 15+ Future-Proofing** | **3.0 / 10** (NNAPI dependent) | **1.0 / 10** (SELinux `noexec` blocks it) | **1.0 / 10** (`noexec` blocks it) | **4.0 / 10** (NNAPI in ladder, XNNPACK CPU fallback) |
| **Memory / OOM Safety** | **3.0 / 10** (In-heap bitmaps only) | **7.0 / 10** (Disk-streamed in CLI) | **6.5 / 10** (Disk-streamed in CLI) | **8.5 / 10** (Streamed PNG + Band Writer) |
| **Google Play Compliance** | **10 / 10** (100% compliant) | **0 / 10** (Violates dynamic exec policy) | **0 / 10** (Violates dynamic exec policy) | **10 / 10** (100% compliant) |
| **Inference Speed / GPU** | **4.0 / 10** (CPU/NNAPI partition) | **9.0 / 10** (Native Vulkan GPU) | **9.0 / 10** (Native Vulkan GPU) | **4.5 / 10** (CPU/NNAPI partition) |
| **Model Ecosystem** | **6.0 / 10** (Real-ESRGAN ONNX only) | **8.5 / 10** (NCNN Real-ESRGAN/RealSR) | **9.5 / 10** (RealSR, RealCUGAN, Waifu2x, MNN) | **8.0 / 10** (Real-ESRGAN, SCUNet, FBCNN) |

---

## 3. Comprehensive Feature & Technical Comparison Matrix

| Criteria | PixelClarity Android | MatUnCNN | RealSR-NCNN-Android | VeilFrame (Current State) | VeilFrame (Proposed NCNN Target) |
|---|---|---|---|---|---|
| **Runtime Engine** | ONNX Runtime Mobile (`v1.30.0`) | Tencent NCNN (`2024xxxx`) via CLI binary | Tencent NCNN via CLI binary | ONNX Runtime Mobile (`v1.20.x`) | Tencent NCNN via **In-Process JNI NDK** |
| **Execution Boundary** | In-process JNI | External ELF via `ProcessBuilder` & `/proc/self/fd` | External ELF via `ProcessBuilder("sh")` | In-process JNI | **In-process JNI (Direct C++ Call)** |
| **Hardware Delegates** | NNAPI, CPU (XNNPACK) | Vulkan Compute (GPU), CPU | Vulkan Compute (GPU), CPU | NNAPI, XNNPACK, CPU | **Vulkan Compute (Adreno/Mali/Tensor), CPU (NEON)** |
| **Android 15 Status** | NNAPI disabled; CPU fallback | Blocked by W^X / `noexec` | Blocked by W^X / `noexec` | NNAPI disabled; CPU fallback | **100% Compliant (Vulkan is first-class on Android 15+)** |
| **16 KB Page Alignment** | Configured in Gradle | Unknown / Stale CLI binaries | Stale binaries (4 KB aligned) | Fully verified 16 KB compliant | **16 KB ELF-aligned native `.so`** |
| **Tiling Engine** | In-Memory Kotlin Bitmap slicing | C++ NCNN spatial tiling in CLI binary | C++ NCNN spatial tiling in CLI binary | Kotlin `AiProcessor` + `TiledBitmapPipeline` | **C++ Native Tiling with Streaming Output** |
| **Memory Footprint** | Massive (Heap-bound: $>500$ MB for 12 MP) | Low JVM heap; high process swap | Low JVM heap; high process swap | Low JVM heap; uses `StreamingPngWriter` | **Minimal JVM heap; zero CLI overhead; bounded native RAM** |
| **Model Packaging** | GitHub Releases on-demand (mmap) | GitHub Releases zip on-demand | Bundled in assets or manual SD card | On-demand downloader (mmap + SHA-256) | **On-demand downloader (mmap + SHA-256)** |
| **Adreno GPU Stability** | N/A (NNAPI/CPU) | Known black output (`vkQueueSubmit -4`) | Known black output on Adreno | N/A (runs on CPU/NNAPI) | **Shader timeout watchdog clamp + FP16 safety switches** |
| **APK Size Overhead** | $+18$ MB (ONNX Runtime AAR) | $+45$ MB (Bundled CLI + Assets) | $+65$ MB (Bundled CLI + Models) | $+20$ MB (ONNX Runtime AAR) | **$+6$ MB (`libncnn.so` + JNI wrapper per ABI)** |

---

## 4. The Critical Android 15+ Reality: Why NNAPI is Dead

Google's Android 15 developer documentation explicitly states:
> *"The Neural Networks API (NNAPI) is deprecated starting in Android 15. App developers should migrate to LiteRT (formerly TensorFlow Lite) or vendor-specific acceleration APIs."*

### What this means for ONNX Runtime:
1. In ONNX Runtime Mobile, `options.addNnapi()` is an abstraction over Android's `libneuralnetworks.so`.
2. On Android 15+ devices, NNAPI HAL implementations are no longer updated or certified by Google.
3. When an ONNX model encounters unsupported operations under NNAPI, it silently partitions the computation graph back to CPU. The overhead of repeatedly copying tensors between NNAPI buffers and CPU memory destroys performance.
4. **Conclusion**: Keeping ONNX Runtime + NNAPI as VeilFrame's primary acceleration strategy guarantees that VeilFrame will run **100% on CPU** on all Android 15+ devices, causing high latency, device heating, and user churn.

### Why Vulkan (via NCNN) is the Future on Android:
1. **Vulkan is mandatory on all modern Android devices** (Android 10+ requires Vulkan 1.1; Android 14/15 requires Vulkan 1.3).
2. Vulkan compute shaders run directly on the device's GPU (Qualcomm Adreno, ARM Mali, Immortalis, Samsung Xclipse, PowerVR).
3. Vulkan does **not** depend on OS neural APIs; it talks directly to the GPU driver.
4. **NCNN is the most mature Vulkan neural engine in existence**, optimized by Tencent specifically for mobile architectures with zero-copy FP16 math and custom SPIR-V shaders.

---

## 5. Implementation Blueprint: Adopting NCNN Vulkan in VeilFrame

### What Must Change in VeilFrame

VeilFrame should **NOT** adopt MatUnCNN or RealSR-NCNN-Android's flawed `ProcessBuilder` architecture. Instead, VeilFrame must implement an **In-Process JNI NCNN Engine**.

```
VeilFrame Architecture Evolution:

[Current ONNX Path]
ImageUri -> MediaDecoder -> AiProcessor -> ONNX Runtime (CPU/NNAPI) -> StreamingPngWriter -> Result
                                                  ↓ (Slow on Android 15, native RAM spikes)

[New Unified Dual-Engine Architecture]
ImageUri -> MediaDecoder -> UpscaleInferenceEngine
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
       [Vulkan NCNN JNI]                 [Algorithmic Fallback]
  Tencent libncnn.so (Vulkan)              Lanczos3 / Bicubic
        (GPU Accelerated)                  (100% Offline Math)
                │
                ▼
     StreamingPngWriter -> Scoped Storage (Pictures/VeilFrame)
```

### Detailed Package & Component Impact

#### 1. Native Build Layer (`CMakeLists.txt` & NDK)
* **New Dependencies**:
  - `ncnn-android-vulkan` prebuilt release (static or shared libraries for `arm64-v8a`, `armeabi-v7a`, `x86_64`).
  - Vulkan headers (part of Android NDK `r26`+).
* **CMake Configuration** (`android/app/src/main/cpp/CMakeLists.txt`):
  ```cmake
  cmake_minimum_required(VERSION 3.22.1)
  project(veilframe_ncnn)

  set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
  find_package(ncnn REQUIRED)

  add_library(veilframe_upscale_native SHARED
      veilframe_ncnn_jni.cpp
      realsr_vulkan.cpp
  )

  target_link_libraries(veilframe_upscale_native
      ncnn
      android
      jnigraphics
      log
  )
  ```

#### 2. Native C++ JNI Bridge (`veilframe_ncnn_jni.cpp`)
* Implements direct in-memory JNI methods passing raw bitmap buffers or tile arrays:
  ```cpp
  extern "C" JNIEXPORT jint JNICALL
  Java_com_veilframe_app_upscale_inference_NcnnVulkanEngine_nativeUpscaleTile(
      JNIEnv* env, jobject thiz,
      jlong handle,
      jbyteArray inRgb, jint tileW, jint tileH,
      jbyteArray outRgb, jint scale,
      jint padLeft, jint padTop, jint cropW, jint cropH
  );
  ```
* Bypasses all `ProcessBuilder`, `noexec`, and `/proc/self/fd` issues.

#### 3. Overcoming the Adreno GPU Black-Screen Defect (`VK_ERROR_DEVICE_LOST`)
* **Root Cause in MatUnCNN/RealSR**: When a tile is too large (e.g. $512\times 512$ with 23 RRDB blocks), the Vulkan compute dispatch takes $>2000\text{ ms}$. The Qualcomm Adreno GPU kernel watchdog detects a frozen GPU thread and kills the context (`-4`), returning a black bitmap.
* **VeilFrame Solution**:
  1. **Dynamic Tile Size Clamping**: On Adreno GPUs (detected via `VkPhysicalDeviceProperties.vendorID == 0x5143`), automatically clamp max tile size to `128` or `192` (instead of `256` or `384`).
  2. **Sub-dispatch Flushes**: Call `vkQueueSubmit` with smaller compute batches so the GPU driver stays responsive.
  3. **Seamless CPU Fallback**: If Vulkan returns `VK_ERROR_DEVICE_LOST` or fails initialization, fail gracefully to NCNN multi-threaded ARM NEON CPU mode or VeilFrame's mathematical Lanczos 3 engine.

#### 4. Model Storage & Migration
* **Model Format**: Replace bulky `.onnx` models (68 MB) with NCNN `.bin` + `.param` pairs (often 40% smaller due to native FP16 weight quantization, ~18–35 MB).
* **Download Pipeline**: Re-use VeilFrame's existing `ModelDownloadManager` with SHA-256 checksum verification.

---

## 6. Implementation Action Plan

1. **Phase 1: Dual-Backend Abstraction**
   - Refactor `UpscaleInferenceEngine.kt` to define an interface `SuperResolutionBackend` (`OnnxBackend` vs `NcnnVulkanBackend`).
   - Retain ONNX as a legacy fallback while introducing the native NCNN JNI bridge.

2. **Phase 2: Add NCNN Vulkan Native Library**
   - Integrate `ncnn-android-vulkan` prebuilts with 16 KB ELF alignment into `android/app/src/main/jniLibs/`.
   - Implement `NcnnVulkanEngine.kt` and `realsr_vulkan.cpp` with tile seam blending.

3. **Phase 3: Connect to `StreamingPngWriter`**
   - Connect NCNN tile outputs directly to VeilFrame's `StreamingPngWriter`, preserving VeilFrame's zero-heap-RAM advantage over PixelClarity.

4. **Phase 4: Deprecate NNAPI**
   - Remove NNAPI probes on Android 15+ devices; route directly to Vulkan GPU $\to$ NCNN CPU $\to$ Lanczos 3.

---

*Report prepared for VeilFrame Architecture & Mobile Machine Learning Subsystem.*
