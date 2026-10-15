# Hardened Architectural Audit: RealSR-NCNN-Android vs. MatUnCNN & Implementation Blueprint

**Target System**: VeilFrame Image Upscaler Subsystem  
**Distribution Target**: GitHub Releases (Direct Sideloaded APK / Non-Google Play)  
**Evaluated Codebases**:
* **Repo A**: [`tumuyan/RealSR-NCNN-Android`](https://github.com/tumuyan/RealSR-NCNN-Android) (Branch `master`)
* **Repo B**: [`Celesth/MatUnCNN`](https://github.com/Celesth/MatUnCNN) (Branch `main`)
* **Host Engine**: [`VeilFrame`](https://github.com/sahir247/VeilFrame) (`StreamingPngWriter.kt`, `UpscaleInferenceEngine.kt`)
**Date of Audit**: October 14, 2026  
**Auditor**: VeilFrame Core Engineering  

---

## 1. Ground-Truth Claim-by-Claim Audit

| Claim Evaluated | Verification Verdict | Source-Level Reality & Code Proof |
| :--- | :---: | :--- |
| **RealSR has no universal 160 px Adreno limit** | **VERIFIED TRUE** | In `RealSR main.cpp:973-998`, tile selection branches by model family. `models-Real-ESRGANv` (v2/v3) uses a pure VRAM heap-budget ladder ($400, 200, 100, 64, 32$) without any Adreno-specific 160 px cap. The 160 px cap applies **only** to legacy non-v models when $\text{heap\_budget} > 2800\text{ MB}$. |
| **Adding a row callback makes RealSR stream output** | **INCOMPLETE AS PROPOSED** | `realsr.cpp:process()` allocates an on-GPU horizontal band `out_gpu` across the full image width, but copies it into a caller-allocated full `outimage` buffer in host memory. Adding a row callback eliminates full-image output allocation **only if** the caller-provided `outimage` is removed and GPU band memory is reclaimed per strip. |
| **B1 uses approximately 15–25 MB of RAM** | **UNSUPPORTED / FALSE** | Fails to account for full `inimage` memory, native model parameters, Vulkan compute staging allocators, NCNN intermediate blobs, and full-width output bands. Actual memory footprint scales with input dimensions, output width, scale factor, and GPU staging buffers. |
| **B1 guarantees memory usage independent of resolution** | **FALSE AS PROPOSED** | RealSR's `process()` receives the entire input image as an un-tiled `ncnn::Mat inimage`. Furthermore, `out_gpu.create(w * scale, ...)` scales linearly with the full upscaled width. True resolution-independent memory requires **both input region streaming and output band streaming**. |
| **B2 allocates the entire output image in memory** | **VERIFIED TRUE** | In CLI execution (`main.cpp`), the process allocates `v.outimage = ncnn::Mat(w * scale, h * scale, c)` in host RAM before writing to disk with `stb_image` / `webp_image`. For 4K ($4\times$) this is $\approx 380\text{ MiB}$; for 8K ($4\times$) this is $\approx 1.48\text{ GiB}$ in raw RGB bytes alone. |
| **B2 provides complete isolation from GPU crashes** | **OVERSTATED** | Standard process termination (e.g. `SIGSEGV`) is isolated from the host app process, but GPU hardware watchdog timeouts (`VK_ERROR_DEVICE_LOST`), kernel driver panics, and OS-wide LowMemoryKiller (LMK) invocations impact the entire device and foreground app. |
| **MatUnCNN's Adreno CPU fallback should be removed** | **REASONABLE, WITH TESTING** | MatUnCNN unconditionally injects `-g -1` (CPU mode) on Snapdragon devices (`MainViewModel.kt:121, 271`). Adreno should be tested with model-aware tile sizing on Vulkan GPU before deciding whether any hardware-specific CPU fallback is required. |
| **VeilFrame should replace its ONNX engine with NCNN** | **REJECT AS BLANKET REPLACEMENT** | VeilFrame's `UpscaleInferenceEngine` orchestrates algorithmic models (Lanczos/Bicubic), face restoration (CodeFormer), and ONNX models. NCNN should initially be introduced as a **coexisting, dedicated backend** for supported NCNN models (Real-ESRGAN, RealSR, RealCUGAN), preserving ONNX for other workloads. |

---

## 2. Deep Dive: Memory Mathematics & The Real Working Set

The original RealSR CLI allocates full input and output buffers in host memory:
```cpp
// RealSR main.cpp:
v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);
```

### Calculated Raw Buffer Sizes for $4\times$ Upscaling:

| Input Resolution | Dimensions | Raw Input (3-ch RGB) | Upscaled Dimensions ($4\times$) | Raw Output (3-ch RGB) | Total Host I/O Buffers Alone |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **FHD (1080p)** | $1920 \times 1080$ | **$5.93\text{ MiB}$** | $7680 \times 4320$ (33.1 MP) | **$94.9\text{ MiB}$** | **$\approx 100.8\text{ MiB}$** |
| **4K (UHD)** | $3840 \times 2160$ | **$23.7\text{ MiB}$** | $15360 \times 8640$ (132.7 MP) | **$379.7\text{ MiB}$** | **$\approx 403.4\text{ MiB}$** |
| **8K (UHD-2)** | $7680 \times 4320$ | **$94.9\text{ MiB}$** | $30720 \times 17280$ (530.8 MP) | **$1,518.7\text{ MiB}$ ($1.48\text{ GiB}$)** | **$1,613.6\text{ MiB}$ ($\approx 1.58\text{ GiB}$)** |

*Note: In monolithic CLI execution, allocating $>1.6\text{ GB}$ of uncompressed host buffers creates severe memory pressure that risks Low Memory Killer (LMKD) termination, though actual survival depends on active heap limits, foreground status, and device zRAM swap compression.*

### Accurate Native Memory & Allocation Branches in `realsr.cpp`:
1. **Input is Not Streamed in Existing RealSR**: `RealSR::process()` receives the entire input image in `inimage`. For an 8K image, the input host buffer alone is $\approx 94.9\text{ MiB}$.
2. **GPU Band Allocation Branches**: In `realsr.cpp:297-306`:
   ```cpp
   int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
   int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

   ncnn::VkMat out_gpu;
   if ((opt.use_fp16_storage || opt.use_fp16_packed) && opt.use_int8_storage)
   {
       out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, blob_vkallocator);
   }
   else
   {
       out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, blob_vkallocator);
   }
   ```
   `TILE_SIZE_Y` (e.g. 160) is in **input image pixels**. Output band height is $(out\_tile\_y1 - out\_tile\_y0) \times scale = 160 \times 4 = \mathbf{640\text{ rows}}$.
   For an 8K input upscaled $4\times$, output width is **$30,720\text{ pixels}$**. Memory depends strictly on the active branch:

   | Execution Branch | Condition | Element Size | 3-Channel RGB Band | 4-Channel RGBA Band |
   | :--- | :--- | :---: | :---: | :---: |
   | **Branch 1 (Packed 8-bit)** | `(fp16_storage \|\| fp16_packed) && int8_storage` | **1 byte** | **$56.25\text{ MiB}$** ($30,720 \times 640 \times 3 \times 1$) | **$75.0\text{ MiB}$** |
   | **Branch 2 (Float32 fallback)** | Int8 storage disabled or unsupported | **4 bytes** | **$225.0\text{ MiB}$** ($30,720 \times 640 \times 3 \times 4$) | **$300.0\text{ MiB}$** |

   *Note: Branch 1 requires both FP16 storage/packing and Int8 storage flags to be enabled in `ncnn::Option` and supported by the device's Vulkan driver (`VK_KHR_8bit_storage`). If Int8 storage is unavailable, the GPU buffer is $225\text{ MiB}$, not $56.25\text{ MiB}$.*
   *Furthermore, do not assume an identical staging RAM allocation: in Branch 1, `realsr.cpp:591` clones directly into the preallocated destination buffer (`outimage.data + offset`); in Branch 2, a temporary host `Mat out` is instantiated prior to pixel conversion.*

3. **Scanline-Order vs. Horizontal-Tile Streaming Reality**:
   Standard PNG encoding consumes scanlines sequentially across the full width ($0 \dots W-1$). Because RealSR iterates through all horizontal tiles for a given vertical band before downloading, the completed band can be streamed scanline-by-scanline into the PNG writer. However, working memory is **independent of image height, but scales with image width**:
   $$\text{Peak Working RAM} \propto (\text{Image Width} \times \text{Scale}) \times (\text{Tile Height} \times \text{Scale}) + \text{Model Tensors} + \text{Staging}.$$
4. **Architectural Guarantee**:
   > **Native streaming eliminates the massive full-image output buffer ($380\text{ MB}\text{–}1.5\text{ GB}$). However, achieving a strictly bounded working set requires refactoring both the input loading path (region streaming) and the output processing path (band streaming). Peak memory remains a dynamic function of input width, output scale, tile size, model architecture, and GPU driver options.**

---

## 3. Streaming PNG Integration: Channel Formats & JNI Overhead

VeilFrame's existing [`StreamingPngWriter.kt:82-98`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/inference/StreamingPngWriter.kt#L82-L98) accepts `IntArray` in packed ARGB_8888 format. Connecting it to RealSR requires solving two critical issues### 3.1. Extending `StreamingPngWriter` for Direct Buffers & Color Types
VeilFrame's existing [`StreamingPngWriter.kt:82-98`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/inference/StreamingPngWriter.kt#L82-L98) currently accepts `IntArray` through `writeRow()` and writes exclusively PNG Color Type 6 (RGBA, 4 bytes/pixel), converting each integer into RGBA bytes inside its preallocated `rowBuf`.

Connecting it to native NCNN requires **extending the writer**, not leaving it unmodified:
1. **Direct ByteBuffer Interface (`writeRowRgba`)**:
   Add a method accepting a `ByteBuffer` (or direct memory pointer + length) to stream pre-packed RGBA bytes straight into `DeflaterOutputStream`. This eliminates the intermediate Java `IntArray` and avoids unpacking integers pixel-by-pixel.
2. **Truecolor RGB Support (Color Type 2)**:
   RealSR outputs 3-channel RGB when no alpha channel exists. Today's writer forces RGBA, meaning 3-channel models must synthesize a dummy alpha channel ($0\text{xFF}$), inflating output file size and bandwidth by 33%. Supporting PNG Color Type 2 allows streaming raw 3-byte RGB scanlines directly.
3. **Zero-Allocation vs. Zero-Copy Precision**:
   A reusable direct `ByteBuffer` eliminates per-row Java `byte[]` GC allocations, but writing to `DeflaterOutputStream` (wrapping native zlib) still copies bytes into zlib's internal compression sliding-window buffer. The pipeline is **strictly zero-allocation per row**, but **not zero-copy end-to-end**.

---

## 4. Native Safety & Concurrency Hardening in `realsr.cpp`

### 4.1. The Destructor Null-Dereference Defect
In `RealSR-NCNN-Android-CLI/RealSR/src/main/jni/realsr.cpp:28-44`:
```cpp
// BUGGY ORIGINAL CODE:
RealSR::~RealSR() {
    delete realsr_preproc;
    delete realsr_postproc;
    bicubic_2x->destroy_pipeline(net.opt);  // SIGSEGV if null!
    delete bicubic_2x;
    bicubic_3x->destroy_pipeline(net.opt);  // SIGSEGV if null!
    delete bicubic_3x;
    bicubic_4x->destroy_pipeline(net.opt);  // SIGSEGV if null!
    delete bicubic_4x;
}
```
* If `load()` fails early (e.g., missing model files or invalid param files), it returns `-1` before `bicubic_2x`, `3x`, `4x` are initialized. Destroying the object crashes the app.
* **Required Patch**:
  ```cpp
  RealSR::~RealSR() {
      delete realsr_preproc; realsr_preproc = nullptr;
      delete realsr_postproc; realsr_postproc = nullptr;
      if (bicubic_2x) { bicubic_2x->destroy_pipeline(net.opt); delete bicubic_2x; bicubic_2x = nullptr; }
      if (bicubic_3x) { bicubic_3x->destroy_pipeline(net.opt); delete bicubic_3x; bicubic_3x = nullptr; }
      if (bicubic_4x) { bicubic_4x->destroy_pipeline(net.opt); delete bicubic_4x; bicubic_4x = nullptr; }
  }
  ```

### 4.2. Model Load Return-Code Propagation
In `realsr.cpp:71, 82`, `net.load_param()` and `net.load_model()` return status codes that are currently discarded. JNI must validate:
```cpp
int ret = net.load_param(parampath.c_str());
if (ret != 0) return -2;
ret = net.load_model(modelpath.c_str());
if (ret != 0) return -3;
```

### 4.3. Concurrency Serialization
While RealSR CLI demonstrates that multiple worker threads can invoke extractors concurrently on the same `RealSR` instance, shared GPU command pools and Vulkan staging allocators complicate cancellation, error propagation, and row-order sequencing in an Android app.
* **VeilFrame Architecture Rule**: Upscaling jobs must be **serialized per `RealSR` instance**. Concurrent execution should only be introduced after benchmarking multi-queue Vulkan submissions on mobile GPUs.

---

## 5. Comparison: Option B1 vs. Option B2

| Evaluation Dimension | Option B1: Native JNI In-Process Streaming | Option B2: Subprocess CLI Execution (Experimental Smoke-Test Only) |
| :--- | :--- | :--- |
| **Execution Architecture** | In-process native C++ (`libveilframe_ncnn.so`) loaded via `System.loadLibrary()`. | Out-of-process standalone binary (`realsr-ncnn`) invoked via `ProcessBuilder`. |
| **Peak Memory Footprint** | **Substantially Reduced**: Eliminates full output image allocation ($380\text{ MB}\text{–}1.5\text{ GB}$). Working memory is bounded by input dimensions, GPU output band, and model tensors. | **High**: Allocates full input AND full output image in process memory ($>1.6\text{ GB}$ for 8K $4\times$). Creates severe memory pressure on devices with limited RAM. |
| **Storage & Flash Wear** | **Zero Intermediate Files**: Feeds source rows from memory/stream $\rightarrow$ GPU $\rightarrow$ deflates directly to destination PNG file. | **Double Disk Thrashing**: Writes input PNG to disk $\rightarrow$ Subprocess reads it $\rightarrow$ Subprocess writes full output PNG to disk $\rightarrow$ VeilFrame reads it back. |
| **Streaming PNG Integration** | **Native Direct Delivery**: Reusable JNI buffer streams scanlines straight into extended `StreamingPngWriter`. | **Incompatible**: Cannot stream scanlines from closed CLI executable without complex stdout pipe parsing. |
| **Crash & Fault Isolation** | **Shared Fate**: Native segfault in C++ terminates VeilFrame's host JVM process. Requires strict native safety hardening. | **Process-Isolated**: Subprocess crash exits with non-zero code; VeilFrame catches `exitCode != 0` and presents error dialog. |
| **Build & Maintenance Surface** | Requires NDK, CMake, compiling NCNN and SPIR-V headers in VeilFrame's Gradle build. | Zero NDK configuration in app build; uses precompiled binary drops. |
| **Android OS Compatibility** | Standard NDK JNI. Compliant across all Android versions (API 26–35+). Requires ABI packaging (`arm64-v8a`) and 16 KB ELF page alignment for 16 KB page-size kernels. | **High OS Restriction Risk**: Apps targeting API 29+ (VeilFrame is `targetSdk 35`) are blocked by Android OS/SELinux from executing binaries in writable app home directories. Sideloading does NOT bypass OS kernel W^X restrictions. Strictly an experimental smoke-test path. |
| **Inference Latency** | Lowest: Zero IPC overhead, zero process spawn time. | Adds process spawning and binary loading overhead (varies by storage hardware and kernel scheduler). |

---

## 6. Phased Implementation Roadmap for VeilFrame

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    REVISED NCNN IMPLEMENTATION ROADMAP                        │
├───────────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Engine & Model Contract Hardening                                    │
│   • Retain onnxruntime-android in build.gradle.kts (required for PII & Rembg) │
│   • Introduce NcnnEngineKind { REALSR_ESRGAN, REALCUGAN, WAIFU2X }            │
│   • Update UpscalePresetRegistry to prevent silent Lanczos fallback           │
│   • Add regression tests ensuring all presets resolve to valid AI models      │
├───────────────────────────────────────────────────────────────────────────────┤
│ Phase 2: Dual-Artifact Model Repository & Download Manager                    │
│   • Refactor UpscaleModelRepository to store paired (.param, .bin) files      │
│   • Refactor ModelDownloadManager for transactional dual-file download + SHA  │
│   • Register curated initial roster: Real-ESRGAN (x4plus) and Real-ESRGANv3  │
│     anime (x4), with exact bytes and SHA-256 digests                          │
├───────────────────────────────────────────────────────────────────────────────┤
│ Phase 3: Hardware Verification & Engine Adaptation                            │
│   • Benchmark native NCNN on physical test devices (Adreno & Mali)            │
│   • Verify Adreno driver stability and record real end-to-end latencies       │
│   • Measure actual VRAM allocations across Branch 1 (int8) and Branch 2 (fp32)│
├───────────────────────────────────────────────────────────────────────────────┤
│ Phase 4: Native JNI In-Process Streaming (Option B1)                          │
│   • Extract TileGrid, SourceAccess, and StreamingPngWriter from AiProcessor   │
│   • Extend StreamingPngWriter with writeRowRgba(ByteBuffer) & Color Type 2    │
│   • Implement C++ JNI bridge supporting row-band callback streaming           │
│   • Retire upscale-specific ONNX files (OnnxSessionManager, ImageTensor, etc.)│
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Strategic Analysis: Stripping ONNX from Upscaler vs. App-Wide Integrity

### 7.1. Code Verification: Where is ONNX Used in VeilFrame?
An exhaustive audit across `android/app/src/main/java/com/veilframe/app` reveals three distinct subsystems utilizing ONNX:
1. **Automated PII Redaction (`com.veilframe.app.privacy`)**:
   [`PiiOnnxEngine.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/privacy/PiiOnnxEngine.kt) uses `ai.onnxruntime.*` for YOLOv8 neural inference to detect faces and license plates across video frames and images.
2. **AI Background Removal (`com.veilframe.app.cv.segmentation.rembg`)**:
   [`RembgModel.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/cv/segmentation/rembg/RembgModel.kt) and [`MaskPostprocessor.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/cv/segmentation/rembg/MaskPostprocessor.kt) use `ai.onnxruntime.*` to run BiRefNet Lite, ISNet General, and U2Net Human segmentation models.
3. **Image Upscaler (`com.veilframe.app.upscale`)**:
   Uses ONNX Runtime for neural super-resolution via [`OnnxSessionManager.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/inference/OnnxSessionManager.kt), [`ImageTensor.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/inference/ImageTensor.kt), and downloaded models in [`UpscaleModelRegistry.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/model/UpscaleModelRegistry.kt).

* **Gradle Dependency Verdict**: `implementation("com.microsoft.onnxruntime:onnxruntime-android:1.27.0")` must remain in `build.gradle.kts`. Removing it would break PII redaction and background removal.
* **Upscaler Subsystem Verdict**: The upscaler package is completely decoupled from `rembg` and `privacy`. It can be migrated independently.

### 7.2. The `model.ort` Filename Defect: Nuanced Analysis
In [`UpscaleModelRepository.kt:20`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/model/UpscaleModelRepository.kt#L20), all downloaded models are saved as `model.ort`.
In [`OnnxSessionManager.kt:283`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/inference/OnnxSessionManager.kt#L283), `modelName.endsWith(".ort")` selects `OptLevel.NO_OPT`.

This must be evaluated in two distinct cases:
1. **The 19 `.ort` Artifacts**:
   The filename matches their true format. ORT-format models are pre-compiled and optimized ahead of time using the ONNX Runtime conversion tool. Setting runtime `NO_OPT` on pre-optimized `.ort` models is standard practice and does not necessarily cause unoptimized execution.
2. **The 3 Raw `.onnx` Artifacts (`RealESR-AnimeVideo-x4v3.onnx`, `fbcnn_color_fp16.onnx`, `scunet_color-GAN.onnx`)**:
   These are raw ONNX graphs renamed to `.ort`. ONNX Runtime inspects file extensions to determine format; naming a standard ONNX graph `.ort` can cause session creation errors or format rejection. Furthermore, selecting `NO_OPT` prevents standard graph optimizations (node fusion, constant folding) from running on these unoptimized graphs.
3. **Model Family Detection Failure**:
   Checking `modelFile.name.startsWith("scunet_")` or `startsWith("fbcnn_")` in [`ModelInfo.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/inference/ModelInfo.kt#L25) and [`AiExtensions.kt`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/main/java/com/veilframe/app/upscale/inference/AiExtensions.kt#L188) fails completely because the filename is always `"model.ort"`. This disables SCUNet's chunk clamping and FBCNN's quality factor control inputs.

> **Conclusion**: The `model.ort` design defect creates genuine model-loading and metadata-routing hazards, but it should not be claimed as the sole explanation for upscaling latency. The inherent computational cost of neural upscaling on CPU/XNNPACK remains the primary factor.

### 7.3. Test Results & Coverage Analysis
Live execution of the upscaler unit test suite reported:
* **66 tests executed**: 66 passed, 0 failures, 0 errors, 0 skipped.
* **Caveat 1 (Silent Pass on Missing Fixtures)**: In [`RealESRGANPipelineIntegrationTest.kt:96, 130`](file:///c:/Users/parve/Downloads/PrivacyVideoCleaner_v1_source/android/app/src/test/java/com/veilframe/app/upscale/RealESRGANPipelineIntegrationTest.kt#L96), tests use `val modelFile = findModelFile(...) ?: return`. If the model file is not present on the test runner, the test returns silently without executing assertions, which JUnit records as a PASS rather than SKIPPED.
* **Caveat 2 (Qualification vs. Full Upscale)**: The test runtime of 234.9s in `RealESRGANPipelineIntegrationTest` reflects candidate session qualification benchmarking (15 passes on a small 256×256 synthetic tile across XNNPACK, NNAPI, and CPU), NOT a full multi-tile image upscale through `AiProcessor`.
* **Action Required**: Migrate `?: return` to `org.junit.Assume.assumeNotNull(modelFile)` and add dedicated integration tests verifying full-image PNG output dimensions and decoded pixel buffers.

### 7.4. Capability Scope: Intentionally Deferred Features
The initial NCNN migration will utilize a `REALSR_ESRGAN` engine adapter. It will **not** immediately replace specialized model families that require separate native pipelines:
* **CodeFormer** (Face restoration) — requires separate face detection (RetinaFace), alignment, and CodeFormer network.
* **FBCNN** (JPEG deblocking) — requires variable quality-factor conditioning tensor.
* **SCUNet** (Denoising) — requires U-Net architecture with noise level input.

These features must be explicitly categorized as **deferred / dropped in v1** rather than conflated with RealSR.

### 7.5. Initial Curated NCNN Model Roster
For the initial release, only verified model pairs will be registered:

1. **`realesrgan-x4plus`** (General Photo $4\times$):
   * Network graph: `realesrgan-x4plus.param`
   * Weights: `realesrgan-x4plus.bin` ($\approx 64\text{ MB}$, FP16)
   * Engine: `NcnnEngineKind.REALSR_ESRGAN`
2. **`realesrgan-v3-anime`** (Anime/Illustration $4\times$):
   * Network graph: `realesrgan-v3-anime-x4.param`
   * Weights: `realesrgan-v3-anime-x4.bin` ($\approx 17\text{ MB}$, FP16)
   * Engine: `NcnnEngineKind.REALSR_ESRGAN`
3. **`realsr-df2k`** (Camera Photo Restoration $4\times$):
   * Network graph: `realsr-df2k.param`
   * Weights: `realsr-df2k.bin` ($\approx 64\text{ MB}$, FP16)
   * Engine: `NcnnEngineKind.REALSR_ESRGAN`

### 7.6. Refactoring Blueprint: Replacing ONNX with Native NCNN in `upscale`
| File to Modify / Remove | Action | Architecture Detail |
| :--- | :--- | :--- |
| `model/UpscaleModel.kt` | **Modify** | Add `engineKind: NcnnEngineKind` (`REALSR_ESRGAN`, `REALCUGAN`, `WAIFU2X`, `ALGORITHMIC`). Replace `ModelType.AI_ONNX` with `ModelType.AI_NCNN`. |
| `preset/UpscalePresetRegistry.kt` | **Modify** | Update all preset `recommendedModelId`s to point to the new NCNN model IDs. Add a regression test asserting no preset resolves to `LANCZOS`. |
| `model/UpscaleModelRegistry.kt` | **Rewrite** | Purge the old ONNX models. Register the curated NCNN roster with paired `.param` and `.bin` metadata. |
| `model/UpscaleModelRepository.kt` | **Update** | Support paired `.param` and `.bin` storage per model ID. |
| `download/ModelDownloadManager.kt`| **Update** | Transactional download of both `.param` and `.bin` files with independent SHA-256 verification. |
| `inference/StreamingPngWriter.kt` | **Extend** | Add `writeRowRgba(buffer: ByteBuffer)` and PNG Color Type 2 (RGB) support. |
| `inference/UpscaleInferenceEngine.kt` | **Update** | Dispatch `ModelType.AI_NCNN` to `NcnnUpscaleEngine` instead of `AiProcessor`. |
| `ui/ImageUpscalerController.kt` | **Update** | Update diagnostic UI badges from `XNNPACK → NNAPI → CPU` to `NCNN Vulkan GPU`. |

---

## 8. Final Strategic Verdict:
* **Option B1 (Native JNI In-Process Streaming) is the sole production architecture.** Option B2 (CLI Subprocess) is strictly an experimental smoke-test path and cannot be relied upon due to Android API 29+ W^X SELinux restrictions.
* **Separation of Concerns**: Retain ONNX Runtime in `build.gradle.kts` for `rembg` and PII redaction, while migrating the decoupled `upscale` subsystem to high-performance native NCNN Vulkan.
* **Verified Baseline**: Attribute existing upscaler performance accurately (XNNPACK CPU execution + session qualification overhead), rather than blaming all slowdowns on the `model.ort` naming bug.


