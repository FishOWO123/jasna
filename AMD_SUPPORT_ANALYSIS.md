# Jasna AMD GPU 支援路徑分析報告

> **範圍**：本報告基於 Jasna 專案目前的工作樹（`g:\lada\jasna\jasna`）原始碼，梳理 AMD/ROCm 路徑的實現深度，與 NVIDIA 路徑做對照，並按 Windows / Linux 分別評估實際可用性。
> **結論先行**：AMD 並非「GPU 跑不起來」，而是**只完成了解碼 + YOLO 偵測 + AMF 編碼 + 純 PyTorch BasicVSR++ 修復**這一條最窄路徑；所有「加速 / 高階 / 進階」特性（TensorRT sub-engines、MIGraphX RF-DETR、unet-4x、RTX Super Res、SD1.5、Smart-render、AMD 上的 basicvsrpp TensorRT 路徑）都明確被禁用或回退到慢路徑。Linux 上這條窄路徑是「可用但慢」；Windows 上**目前更接近不可用**，因為 AMD 的核心依賴（onnxruntime-rocm、PyTorch ROCm wheels、ROCm 本身）在 Windows 官方並未發布，專案裡已經有相應的退路（`migraphx_provider_available`、CPU provider），但這代表 RF-DETR 在 Windows AMD 上會回退到 CPU 推理。

---

## 1. 作者已實現哪些 AMD 模組（盤點）

下表列出在程式碼中實際寫出的 AMD 條件分支，與對應的 NVIDIA 等價物。

| 模組 / 階段 | NVIDIA 路徑 | AMD 路徑（已實現） | 檔案 |
|---|---|---|---|
| 加速器偵測 | `is_nvidia_device` | `is_amd_device`，看 `torch.version.hip` 判定；ROCm 下用 `cuda` device type | [jasna/accelerator.py](jasna/accelerator.py#L43-L51) |
| 設備能力 | `tensorrt=True, nvcodec=True` | `migraphx=True, amf=True` | [jasna/accelerator.py](jasna/accelerator.py#L63-L70) |
| 影片解碼 | `cud hwaccel` + `nvcuda.dll/libcuda.so.1` 串流 | `amf hwaccel` + `h264/hevc/av1_amf` | [jasna/media/video_decoder.py](jasna/media/video_decoder.py#L94-L143) |
| 影片編碼 | NVENC (`h264/hevc/av1_nvenc`) + `cuda` pix_fmt + `CudaContext` 零拷貝 | AMF (`h264/hevc/av1_amf`) + 主機拷貝 + `pin_memory` 緩衝 | [jasna/media/video_encoder.py](jasna/media/video_encoder.py#L307-L334) (AMF specs), [jasna/media/video_encoder.py](jasna/media/video_encoder.py#L700-L735) (encode frame) |
| HLS 串流編碼 | ffmpeg `h264_nvenc` | ffmpeg `h264_amf` (qvbr) | [jasna/streaming_encoder.py](jasna/streaming_encoder.py#L125-L137) |
| RF-DETR 偵測 | TensorRT engine (`*.engine`) | MIGraphX cache (Linux 有 provider)；無 provider 時**回退到 onnxruntime CPU** | [jasna/mosaic/rfdetr.py](jasna/mosaic/rfdetr.py#L26-L42), [jasna/mosaic/migraphx_runner.py](jasna/mosaic/migraphx_runner.py#L108-L160) |
| YOLO 偵測 | YOLO `.pt` 直跑 PyTorch **或** TensorRT 編譯 | 直接走 PyTorch (`ultralytics AutoBackend`)；**不做** TensorRT/MIGraphX 編譯 | [jasna/mosaic/yolo.py](jasna/mosaic/yolo.py#L141-L150), [jasna/mosaic/detection_registry.py](jasna/mosaic/detection_registry.py#L154-L170) |
| BasicVSR++ 修復 | PyTorch 或 `torch_tensorrt` sub-engines (NVIDIA only) | **僅** PyTorch eager，TRT sub-engines 完全跳過 | [jasna/engine_compiler.py](jasna/engine_compiler.py#L126-L155), [jasna/restorer/basicvsrpp_mosaic_restorer.py](jasna/restorer/basicvsrpp_mosaic_restorer.py#L33-L46) |
| unet-4x 二次修復 | TensorRT 編譯 + 加密 engine | 直接 `RuntimeError("unet-4x currently requires the NVIDIA TensorRT build")` | [jasna/engine_compiler.py](jasna/engine_compiler.py#L130-L132) |
| RTX Super Res | NVIDIA Maxine (`nvvfx`) | 不存在（檔案存在但不走 AMD） | [jasna/restorer/rtx_superres_secondary_restorer.py](jasna/restorer/rtx_superres_secondary_restorer.py#L17) |
| SD 1.5 圖像修復 | checkpoint + PyTorch | checkpoint + PyTorch（理論上和 vendor 無關，但屬於 supporter 限定功能） | [jasna/restorer/sd15_inpaint_restorer.py](jasna/restorer/sd15_inpaint_restorer.py) |
| TVAI 二次修復 | ffmpeg 委外 | 同上 | — |
| VRAM offloader | `torch.cuda.*`（ROCm 下也是這個 namespace） | 同上，但 hard-code `self._offload_device_type = "cuda"` | [jasna/vram_offloader.py](jasna/vram_offloader.py#L60-L65) |
| 引擎編譯子行程 | `python -m jasna.engine_compiler` (TRT 編譯) | 同入口，但**只**做 detection engine；TRT 編譯全跳過 | [jasna/engine_compiler.py](jasna/engine_compiler.py#L207-L264) |
| Engine preflight (GUI) | 檢查 TRT engines 存在 | 檢查 MIGraphX cache 存在；無 provider 時跳過 | [jasna/gui/engine_preflight.py](jasna/gui/engine_preflight.py#L37-L92) |
| GPU wizard 顯示 | `torch.cuda.get_device_capability` | 顯示 `ROCm {version}` | [jasna/gui/wizard.py](jasna/gui/wizard.py#L415-L428) |
| 系統驅動檢查 | nvidia-smi | `torch.version.hip` 優先；無 nvidia-smi 才視為無驅動 | [jasna/os_utils.py](jasna/os_utils.py#L301-L312) |
| 支援性檢查 | `MIN_GPU_COMPUTE = (7,5)` | AMD 直接 `True`（無最低運算能力檢查） | [jasna/os_utils.py](jasna/os_utils.py#L23-L40) |

---

## 2. NVIDIA vs AMD 程式碼路徑差異（關鍵 if/else）

### 2.1 偵測階段（mosaic detection）

**NVIDIA（YOLO + RF-DETR 都有 TRT engine）**
- `rfdetr.py::RfDetrMosaicDetectionModel.__init__` → 走 `is_nvidia_device` → 載入 `.engine`。
- `yolo.py::YoloMosaicDetectionModel.__init__` → 走 `is_nvidia_device` → 載入 `*.pt.engine`。
- `engine_compiler.py::_subprocess_compile` → 透過 `trt.compile_onnx_to_tensorrt_engine` 編譯。

**AMD**
- `rfdetr.py`：走 `is_amd_device` → `MigraphxRunner`。
- `MigraphxRunner`：
  - 若 `MIGraphXExecutionProvider` 在 `onnxruntime` 內可用 → 用 MIGraphX + 把 cache 寫到 `<onnx>.migraphx/<digest>-<gcnArch>-<precision>/`。
  - 若**不可用** → 退到 `CPUExecutionProvider`，**無 cache**，不下任何檔。
  - 輸入必須是**固定形狀**（會主動 raise），MIGraphX 不支援動態維度。
- `yolo.py`：因為 `is_nvidia_device` 是 False，根本不檢查 `.engine` 是否存在，直接走 `ultralytics.AutoBackend(...)`，在 ROCm 上跑原生 PyTorch。
- `engine_compiler.py`：
  - `nvidia` 才編譯 `basicvsrpp` 與 `unet4x`（hard guard `if nvidia:`）。
  - `detection` 永遠會編；對 RF-DETR 走 `MigraphxRunner.__init__`（這會**觸發 MIGraphX 編譯**）；對 YOLO 在 AMD 上**不會**編譯。

### 2.2 修復階段（restoration）

- `basicvsrpp_mosaic_restorer.py`：`if self.use_tensorrt and is_nvidia_device(self.device)` 才載入 `create_split_forward`；AMD 一定走 `load_model(...)` 拿 PyTorch。
- 結果：AMD 跑的是未量化、eager 模式的 BasicVSR++，速度比 NVIDIA 慢 1 個量級。
- `unet4x_secondary_restorer.py`：`engine_compiler.py` 直接 `raise RuntimeError`，連 sub-engine 都不會被請求。

### 2.3 編解碼階段

- 解碼：`video_decoder.py::_setup_amf_decoder` 嘗試建立 `h264_amf`/`hevc_amf`/`av1_amf` decoder，**失敗**就 fallback 到 FFmpeg software decode + 軟解幀透過 `cuda` device type 上傳到 ROCm tensor。沒有 NVIDIA 那一條 `primary_ctx / current_ctx` 的 `CudaContext` 包裝。
- 編碼：`video_encoder.py` 為 AMD 維護了獨立的 `AMF_ENCODER_SPECS`（H.264/HEVC/AV1）、`DEFAULT_AMF_*_ENCODER_OPTIONS`、`_amf_host_input` 處理 10-bit reinterpret。`_encode_frame` 在 AMD 分支把 packed YUV 用 `pin_memory` 緩衝 + `non_blocking=True` 拷貝回 host，再 `av.VideoFrame.from_dlpack(planes, format=...)`；**沒有**像 NVIDIA 那樣建硬體 frame 物件。
- `cq` ↔ `qvbr_quality_level` 在 AMD 上是 alias，會丟錯避免兩者並存（[video_encoder.py#L388-L398](jasna/media/video_encoder.py#L388-L398)）。
- Smart-render（`forced-idr` + `closed_gop`）在 AMD 上**直接 `ValueError("Smart rendering is currently supported only with NVENC")`**（[video_encoder.py#L363](jasna/media/video_encoder.py#L363)）。

### 2.4 串流

- `streaming_encoder.py` 在 AMD 用 `h264_amf -usage lowlatency_high_quality -quality balanced -rc qvbr`；NVIDIA 用 `h264_nvenc -preset p4 -tune ll -spatial-aq -temporal-aq -rc-lookahead 8`。AMF 路徑少了 B 幀以外的 lookahead / AQ，**只輸出 H.264**（NVENC 還有 HEVC/AV1）。

### 2.5 VRAM 管理

- `vram_offloader.py` 不分流，統一呼叫 `torch.cuda.empty_cache / mem_get_info`（ROCm 下 `torch.cuda` namespace 仍可用），門檻 = `total - safetynet`。**邏輯上** AMD 能用，但 `self._offload_device_type = "cuda"` 是 hard-coded，與 `accelerator.py` 強調的「vendor-agnostic」原則不一致（小幅程式碼債）。

### 2.6 GUI / 入口護欄

- `[main.py#L638-L642](jasna/main.py#L638-L642)`：CLI 上若 `is_amd_device(device) and secondary_name != "none"` → `ValueError`。
- `[gui/video_session.py#L96-L101](jasna/gui/video_session.py#L96-L101)`：GUI 同樣 raise。
- GUI wizard：AMD 顯示 `ROCm {version}`，無最低運算能力檢查；NVIDIA 檢查 `(7,5)`。

---

## 3. Windows vs Linux 的可用性差異

| 元件 | Linux AMD | Windows AMD | 備註 |
|---|---|---|---|
| **PyTorch ROCm** | 官方 `pip install torch --index-url https://download.pytorch.org/whl/rocm{x.y}` 有 wheels（需對應的 ROCm 版本，例如 6.2/6.3/7.0） | **無官方 ROCm Windows wheels**（`pytorch.org` 僅 Linux），社群有 `pytorch-rocm-windows` 等移植，但**不在 Jasna 的官方支援矩陣** | ROCm 本身也只官方支援 Linux（ROCm 5.7+ 才有實驗性 Windows driver，但 PyTorch wheels 不發） |
| **onnxruntime-rocm**（MIGraphX） | `pip install onnxruntime-rocm` 可用，`MIGraphXExecutionProvider` 存在 | `onnxruntime` 官方只有 DirectML/CPU/TensorRT provider，**沒有 MIGraphX** | [migraphx_runner.py#L81-L86](jasna/mosaic/migraphx_runner.py#L81-L86) 有 `migraphx_provider_available()` 偵測；Windows AMD 上永遠 `False` |
| **AMF 視訊 SDK / ffmpeg amf 編解碼** | ffmpeg 內建 `h264_amf`/`hevc_amf`/`av1_amf`（需 ROCm Video SDK 或 AMD driver 帶有 amf.dll） | Windows 內建 `amfrt64.dll`（隨 AMD Adrenalin driver），ffmpeg 的 amf muxer 可用 | 兩平台解碼 / 編碼**都可用** |
| **MIOpen** | 隨 ROCm 一起裝；`MIOPEN_FIND_MODE=FAST` 是預設（[accelerator.py#L15-L16](jasna/accelerator.py#L15-L16)） | 不適用 | BasicVSR++ PyTorch 路徑會在 ROCm 上找 MIOpen；對 Windows 來說 PyTorch ROCm 本身就不存在，所以也談不上 MIOpen |
| **TensorRT** | NVIDIA only | NVIDIA only | AMD 不會走這條路 |
| **onnxruntime 標準 provider** | 總有 `CPUExecutionProvider` | 總有 `CPUExecutionProvider` | Windows AMD 上 RF-DETR 會**跑在 CPU**（很慢） |
| **跑得動的最低硬體** | ROCm 官方支援的 GPU（RX 6000/7000、MI100/200/300 等）；舊卡需手動加 `HSA_OVERRIDE_GFX_VERSION` | 需要先搞定 PyTorch ROCm（見上）才能談「可用」 | Jasna 程式碼不檢查 `gcnArchName`，MIGraphX 內部會自己判斷 |
| **預期可用路徑** | 解碼(AMF 失敗→software+upload) → YOLO(PyTorch on ROCm) / RF-DETR(MIGraphX) → BasicVSR++(PyTorch) → 編碼(AMF) | **MIGraphX 不可用 → RF-DETR 走 CPU**；其他同上 | Windows AMD 等同於「RF-DETR 走 CPU」的退化版本 |
| **官方起手腳本** | `run_jasna_amd.sh` | 未提及；Jasna 似乎把 Windows AMD 視為「能跑就跑」 | [README.md#L56](README.md#L56) |

### 3.1 Windows 為何實質不可用

1. `pip install torch` 抓不到 ROCm wheel → `torch.version.hip` 為 `None` → `is_amd_device(...)` 永遠 False。
2. 即便有人硬塞 PyTorch ROCm Windows build，`onnxruntime` 在 Windows 沒有 MIGraphX provider，於是：
   - `MigraphxRunner.__init__` 走 `use_migraphx = False`、`providers = [CPUExecutionProvider]`，RF-DETR 全程 CPU。
   - `MigraphxRunner._input_numpy_dtypes` 仍建構，但 `pin_memory=self.device.type != "cpu"` 在 ROCm 上成立（device.type 是 `cuda`）；這條小邏輯仍正常。
3. AMF 解碼/編碼雖然 Windows 有，但**整個 GPU 加速 pipeline 的瓶頸在 detection**（特別是預設的 `rfdetr-v5`）。CPU 跑 RF-DETR 一張 768×768 影像，FP16/INT8 都沒，幾秒到十幾秒；1 小時 30fps 影片是 108,000 張，**不可接受**。
4. 切到 YOLO 偵測模型（`lada-yolo-v4` 等）可繞過 RF-DETR → 改用 PyTorch on ROCm，**這條路在 Windows AMD 上理論上能跑**（只要 PyTorch ROCm 裝得起來）。

> 結論：Windows AMD 唯一的「可用」劇本是 **(PyTorch ROCm wheels 設法裝好) + 切到 YOLO 偵測 + 不開 secondary restoration + 不開 unet-4x + 不開 smart-render**。這非常脆弱，所以專案將其定位為「experimental」。

### 3.2 Linux 為何勉強可用

- PyTorch + onnxruntime-rocm 是官方支援矩陣。
- MIGraphX 真的能編譯並 cache 到 `<model>.migraphx/<key>/`，第二次跑就吃 cache。
- AMF 在 ffmpeg 中是經典路徑。
- 但 **BasicVSR++ 沒有 TensorRT**、**unet-4x 整個被拒**、**次修復（tvai 以外的）全被拒**，所以「Linux AMD 可用」是指「能把影片跑完」，**速度遠低於 NVIDIA**（粗估 1/3～1/5）。

---

## 4. 已「禁止」或回退的功能彙整

以下行為在 AMD 上**直接失敗 / 回退**，務必視為「AMD 不可用」：

| 功能 | AMD 行為 | 程式碼位置 |
|---|---|---|
| unet-4x 二次修復 | `RuntimeError` | [engine_compiler.py#L130-L132](jasna/engine_compiler.py#L130-L132) |
| `--secondary-restoration` (除 `none` 外) | CLI 與 GUI 都 raise | [main.py#L638-L642](jasna/main.py#L638-L642)、[gui/video_session.py#L96-L101](jasna/gui/video_session.py#L96-L101) |
| BasicVSR++ TensorRT sub-engines | 跳過，永遠 `use_tensorrt=False` | [engine_compiler.py#L134-L137](jasna/engine_compiler.py#L134-L137)、[basicvsrpp_mosaic_restorer.py#L33](jasna/restorer/basicvsrpp_mosaic_restorer.py#L33) |
| YOLO TensorRT engine | 不編譯，PyTorch 跑 | [detection_registry.py#L164-L170](jasna/mosaic/detection_registry.py#L164-L170) |
| Smart-render (`forced-idr`) | `ValueError` | [video_encoder.py#L363](jasna/media/video_encoder.py#L363) |
| HLS streaming codec ≠ H.264 | 寫死 H.264 AMF | [streaming_encoder.py#L126-L137](jasna/streaming_encoder.py#L126-L137) |
| RF-DETR on Windows AMD | 退到 onnxruntime CPU | [migraphx_runner.py#L120-L161](jasna/mosaic/migraphx_runner.py#L120-L161) |
| `MIOPEN_FIND_MODE` 環境變數 | 在 ROCm 環境下默認 `FAST` | [accelerator.py#L15-L16](jasna/accelerator.py#L15-L16) |

---

## 5. 測試覆蓋

`tests/test_amd_support.py` 對上述路徑有相當完整的 mock-based 測試：
- `test_rocm_uses_cuda_device_api_but_reports_amd`：ROCm 下 `cuda:0` 仍被識別為 AMD。
- `test_amd_basicvsrpp_skips_tensorrt_compilation`：確認 AMD 不會觸發 TRT 探測。
- `test_amf_encoder_settings_are_vendor_specific` / `test_video_encoder_selects_amf_and_normalizes_cq` / `test_smart_render_is_rejected_on_amd` / `test_streaming_encoder_selects_amf` / `test_amf_decoder_context_is_created` / `test_amf_p010_host_input_reinterprets_signed_storage`：把 AMF encoder/decoder/HLS 端到端跑過。
- `test_migraphx_runner_provider_and_tensor_bridge` / `test_migraphx_runner_falls_back_to_cpu_onnxruntime`：MIGraphX provider 與 CPU fallback 都有覆蓋。
- `tests/test_gui_engine_preflight.py::test_amd_preflight_checks_only_migraphx_cache` / `test_amd_preflight_needs_no_cache_for_cpu_onnxruntime`：GUI preflight 的兩條路徑。

**缺漏**：
- 沒有真的在 Linux AMD 機器上跑 e2e 的整合測試（CI 上跑不起來）。
- `vram_offloader` 沒有 AMD 專屬測試。
- `benchmark/` 對 AMD 是 0 覆蓋。
- 沒有覆蓋「`--secondary-restoration` 在 AMD 上會 raise」這個 CLI 路徑。

---

## 6. 仍存在的程式碼 / 文件小問題

1. **MIGraphX 動態形狀限制**：`MigraphxRunner` 一律要求固定 `batch × 3 × 768 × 768`；這對 RF-DETR 沒問題，但若日後想用在其他偵測模型，會立刻踩到 `RuntimeError`。
2. **VRAM offloader 沒分流**：`vram_offloader.py` 直接 hardcode `cuda`，沒走 `accelerator.py` 的 vendor 抽象；雖然 PyTorch 在 ROCm 下行為相同，但跟專案自己的「vendor-agnostic」抽象不一致。
3. **WINDOWS 上 RF-DETR 的「沈默回退」**：使用者可能選了 `rfdetr-v5`，系統卻悄悄改成 CPU 跑；`gui/engine_preflight.py` 在 provider 不可用時 `det_exists = True`，連 preflight warning 都不會出現。**建議**：在該分支加上顯眼提示（"RF-DETR will run on CPU; expect very long detection"）。
4. **`os_utils.check_supported_gpu` 對 AMD 沒做 compute capability 檢查**：可能讓不支援的舊 GCN 卡也「通過」檢查，運行時才踩到 MIOpen 報錯。
5. **README 說「AMD 準備時間比 NVIDIA 短」**：這對 Linux 是對的（不用編 TRT），但對 Windows AMD 隱含的假設是「會跑得起來」，未明示 Windows AMD 仍需要 PyTorch ROCm wheels，而這些 wheels 官方並不發。
6. **`_amf_host_input` 對 8-bit 路徑其實是 no-op**（直接回傳 `packed`）；但仍然每次呼叫，無害但略冗。
7. **YOLO 在 AMD 完全沒編譯**：`detect_engine_exists` 直接 `return True`；這對功能沒影響，但 preflight 不會要求使用者重新準備東西，**首跑時 PyTorch 第一次 JIT/warmup 會很長**，建議在 GUI 上加個提示。

---

## 7. 總結

**作者在 AMD 路徑上做了多少？** 解碼（AMF hwaccel + 軟解 fallback）、編碼（AMF 三個 codec + HLS）、RF-DETR（MIGraphX + CPU fallback）、YOLO（原生 PyTorch）、BasicVSR++（原生 PyTorch）、次修復護欄、preflight、wizard、driver check — **每一個都寫了**。

**為何仍實質不可用？** 兩個原因疊加：
1. **瓶頸模型（BasicVSR++ 與 unet-4x）沒有 AMD 等價物**。BasicVSR++ 在 NVIDIA 用了 TensorRT sub-engines 達到 3.1–3.5× 加速；AMD 上是純 PyTorch。unet-4x 直接被禁。這意味著「AMD 跑得起來」=「AMD 跑得比 NVIDIA 慢一個量級」。
2. **Windows 上 PyTorch ROCm 與 onnxruntime-rocm 都缺**，於是 RF-DETR 退化為 CPU 推理，整個 pipeline 不可用。**Linux 上**則基本可用但慢；**Windows 上**對預設的 RF-DETR 模型基本不可用，必須改用 YOLO 偵測模型才有機會跑完。

> 換句話說：對**終端使用者**而言，目前 AMD 是「實驗性的 Linux-only 慢路徑」；Windows AMD 屬於「開源社群 PyTorch-ROCm 玩家」的自助選項，不在產品級可用範圍。

---

# AMD GPU Support Path Analysis Report (English)

> **Scope**: This report analyses the AMD/ROCm code path in the Jasna project (working tree at `g:\lada\jasna\jasna`), compares it to the NVIDIA path, and evaluates real-world usability separately on Windows and Linux.
> **TL;DR**: AMD is *not* "completely broken", but only the narrowest path is wired up — **AMF decode + YOLO detection + AMF encode + plain PyTorch BasicVSR++**. Everything else (TensorRT sub-engines, MIGraphX RF-DETR on Windows, unet-4x, RTX Super Res, Smart-render) is **explicitly disabled or silently falls back to a slow path**. On Linux this narrow path is *usable but slow*; on Windows it is *effectively unavailable* because the upstream dependencies (PyTorch ROCm wheels, onnxruntime-rocm with MIGraphX provider) are not officially published for Windows — the project's own `migraphx_provider_available()` guard catches this, but it then routes RF-DETR through the ONNX Runtime CPU provider.

---

## 1. What the author actually wired up for AMD

| Stage | NVIDIA path | AMD path (as implemented) | Source file |
|---|---|---|---|
| Accelerator detection | `is_nvidia_device` | `is_amd_device` via `torch.version.hip`; ROCm uses `cuda` device type | [jasna/accelerator.py](jasna/accelerator.py#L43-L51) |
| Capability flags | `tensorrt=True, nvcodec=True` | `migraphx=True, amf=True` | [jasna/accelerator.py](jasna/accelerator.py#L63-L70) |
| Video decode | `cud hwaccel` + `nvcuda.dll/libcuda.so.1` stream | `amf hwaccel` + `h264/hevc/av1_amf` | [jasna/media/video_decoder.py](jasna/media/video_decoder.py#L94-L143) |
| Video encode | NVENC (`h264/hevc/av1_nvenc`) + `cuda` pix_fmt + `CudaContext` zero-copy | AMF (`h264/hevc/av1_amf`) + host-side pinned-memory copy | [jasna/media/video_encoder.py](jasna/media/video_encoder.py#L307-L334) (AMF specs), [jasna/media/video_encoder.py](jasna/media/video_encoder.py#L700-L735) (encode frame) |
| HLS streaming | ffmpeg `h264_nvenc` | ffmpeg `h264_amf` (qvbr) | [jasna/streaming_encoder.py](jasna/streaming_encoder.py#L125-L137) |
| RF-DETR detection | TensorRT engine (`*.engine`) | MIGraphX cache (Linux has provider); without provider → **falls back to onnxruntime CPU** | [jasna/mosaic/rfdetr.py](jasna/mosaic/rfdetr.py#L26-L42), [jasna/mosaic/migraphx_runner.py](jasna/mosaic/migraphx_runner.py#L108-L160) |
| YOLO detection | YOLO `.pt` via PyTorch **or** TensorRT engine | PyTorch only via `ultralytics.AutoBackend`; **no** TRT/MIGraphX compilation | [jasna/mosaic/yolo.py](jasna/mosaic/yolo.py#L141-L150), [jasna/mosaic/detection_registry.py](jasna/mosaic/detection_registry.py#L154-L170) |
| BasicVSR++ restoration | PyTorch or `torch_tensorrt` sub-engines (NVIDIA only) | **PyTorch eager only**, sub-engines always skipped | [jasna/engine_compiler.py](jasna/engine_compiler.py#L126-L155), [jasna/restorer/basicvsrpp_mosaic_restorer.py](jasna/restorer/basicvsrpp_mosaic_restorer.py#L33-L46) |
| unet-4x secondary | TensorRT compile + encrypted engine | `RuntimeError("unet-4x currently requires the NVIDIA TensorRT build")` | [jasna/engine_compiler.py](jasna/engine_compiler.py#L130-L132) |
| RTX Super Res | NVIDIA Maxine (`nvvfx`) | Not implemented | [jasna/restorer/rtx_superres_secondary_restorer.py](jasna/restorer/rtx_superres_secondary_restorer.py#L17) |
| SD 1.5 image | checkpoint + PyTorch | checkpoint + PyTorch (vendor-agnostic) | [jasna/restorer/sd15_inpaint_restorer.py](jasna/restorer/sd15_inpaint_restorer.py) |
| TVAI secondary | ffmpeg-out-of-process | Same (vendor-agnostic) | — |
| VRAM offloader | `torch.cuda.*` (also valid under ROCm) | Same; but `self._offload_device_type = "cuda"` is hard-coded | [jasna/vram_offloader.py](jasna/vram_offloader.py#L60-L65) |
| Engine compile subprocess | `python -m jasna.engine_compiler` (TRT compile) | Same entry, but **only** does detection engine; TRT compile is skipped | [jasna/engine_compiler.py](jasna/engine_compiler.py#L207-L264) |
| Engine preflight (GUI) | Check TRT engine files exist | Check MIGraphX cache exists; skip when no provider | [jasna/gui/engine_preflight.py](jasna/gui/engine_preflight.py#L37-L92) |
| GPU wizard display | `torch.cuda.get_device_capability` | Shows `ROCm {version}` | [jasna/gui/wizard.py](jasna/gui/wizard.py#L415-L428) |
| System driver check | nvidia-smi | `torch.version.hip` first; missing nvidia-smi counts as no driver | [jasna/os_utils.py](jasna/os_utils.py#L301-L312) |
| Support gate | `MIN_GPU_COMPUTE = (7,5)` | Always `True` (no minimum compute-capability check) | [jasna/os_utils.py](jasna/os_utils.py#L23-L40) |

---

## 2. NVIDIA vs AMD code-path differences (key branches)

### 2.1 Detection
- **NVIDIA** → both YOLO and RF-DETR load precompiled `.engine` files. Detection engine is built by `trt.compile_onnx_to_tensorrt_engine` in a subprocess.
- **AMD** → RF-DETR goes through `MigraphxRunner`, which:
  - If `MIGraphXExecutionProvider` is present → use MIGraphX, with cache at `<onnx>.migraphx/<digest>-<gcnArchName>-<precision>/`.
  - Otherwise → `providers=[CPUExecutionProvider]`, no cache, no artefact.
  - Requires **fixed** input shapes (MIGraphX has no dynamic-shape support).
- **YOLO on AMD** skips engine existence check entirely; falls through to `ultralytics.AutoBackend(...)` running the raw `.pt` on ROCm.
- **Engine compile subprocess** for AMD: `nvidia` guard skips `basicvsrpp` and `unet4x` compilation; the `detection` branch always runs but for AMD/RF-DETR simply *constructs* a `MigraphxRunner` (which is what builds the MIGraphX cache the first time).

### 2.2 Restoration
- `BasicVSR++` on AMD is **always** PyTorch eager; there is no MIGraphX path for the restorer.
- `unet-4x` is unconditionally rejected on AMD.

### 2.3 Encode / decode
- Decode: AMD uses `amf` hwaccel; on failure, falls back to FFmpeg software decode + a ROCm upload.
- Encode: AMD maintains separate `AMF_ENCODER_SPECS`, `DEFAULT_AMF_*_ENCODER_OPTIONS`, and `_amf_host_input` (10-bit reinterpret). The encode path is a host-side pinned-memory copy + `av.VideoFrame.from_dlpack(planes, format=...)` — there is no zero-copy hardware frame object like NVENC's `cuda` pix_fmt + `CudaContext`.
- `cq` is an alias for AMF's `qvbr_quality_level`; passing both raises.
- Smart-render (`forced-idr`) is rejected on AMD with `ValueError`.
- HLS streaming on AMD is H.264-only (NVENC supports HEVC/AV1 too).

### 2.4 VRAM management
- `vram_offloader.py` is vendor-agnostic via the `torch.cuda` namespace that ROCm also exposes, but the device-type string is hard-coded as `"cuda"` — minor inconsistency with `accelerator.py`.

### 2.5 GUI / entry guards
- `--secondary-restoration` (anything other than `none`) on AMD → `ValueError` in CLI ([main.py#L638](jasna/main.py#L638)) and `RuntimeError` in GUI ([gui/video_session.py#L96](jasna/gui/video_session.py#L96)).
- `compile_basicvsrpp` is silently forced to `False` on AMD.
- GPU wizard on AMD shows `ROCm {version}` and skips the `(7,5)` minimum.

---

## 3. Windows vs Linux usability

| Component | Linux AMD | Windows AMD | Notes |
|---|---|---|---|
| **PyTorch ROCm** | Official wheels on `download.pytorch.org/whl/rocm{x.y}` for several ROCm versions | **No official ROCm Windows wheels**; community ports exist but are not in Jasna's support matrix | ROCm itself is Linux-only officially; experimental Windows drivers appeared in ROCm 5.7+ but no PyTorch wheels |
| **onnxruntime-rocm (MIGraphX)** | `pip install onnxruntime-rocm` works; `MIGraphXExecutionProvider` is available | Official `onnxruntime` on Windows ships DirectML/CPU/TensorRT providers only — **no MIGraphX** | `migraphx_provider_available()` ([migraphx_runner.py#L81](jasna/mosaic/migraphx_runner.py#L81-L86)) catches this and forces the CPU fallback |
| **AMF SDK / ffmpeg AMF** | ffmpeg has `h264_amf/hevc_amf/av1_amf`; needs ROCm Video SDK or AMD driver with amf.dll | Windows bundles `amfrt64.dll` via Adrenalin driver; ffmpeg's amf path is built-in | Both platforms can do AMF encode/decode |
| **MIOpen** | Ships with ROCm; `MIOPEN_FIND_MODE=FAST` is the default in [accelerator.py#L15-L16](jasna/accelerator.py#L15-L16) | N/A | BasicVSR++ PyTorch on ROCm resolves through MIOpen; on Windows PyTorch ROCm doesn't exist, so this is moot |
| **TensorRT** | NVIDIA only | NVIDIA only | AMD never reaches this branch |
| **onnxruntime standard providers** | Always `CPUExecutionProvider` | Always `CPUExecutionProvider` | On Windows AMD, RF-DETR runs on **CPU** |
| **Minimum supported hardware** | Officially supported ROCm GPUs (RX 6000/7000, MI100/200/300); older GCN cards need `HSA_OVERRIDE_GFX_VERSION` | First requires a working PyTorch ROCm build | Jasna itself does not check `gcnArchName`; MIGraphX handles it internally |
| **Expected usable path** | decode (AMF or software+upload) → YOLO (PyTorch ROCm) / RF-DETR (MIGraphX) → BasicVSR++ (PyTorch) → encode (AMF) | **MIGraphX unavailable → RF-DETR on CPU**; rest same | Windows AMD is effectively "RF-DETR on CPU" |
| **Official entry script** | `run_jasna_amd.sh` | Not mentioned; Windows AMD is treated as "if it runs, it runs" | [README.md#L56](README.md#L56) |

### 3.1 Why Windows AMD is effectively unavailable
1. There is no official PyTorch ROCm wheel for Windows, so `torch.version.hip` is `None` and `is_amd_device(...)` returns `False` no matter what GPU is installed.
2. Even if a community ROCm-on-Windows PyTorch build is installed, `onnxruntime` on Windows has no MIGraphX provider, so:
   - `MigraphxRunner.__init__` sets `use_migraphx=False`, `providers=[CPUExecutionProvider]`, and RF-DETR runs entirely on CPU.
   - `pin_memory=self.device.type != "cpu"` still evaluates to `True` on ROCm (device.type is `cuda`), so that micro-path is fine.
3. AMF decode/encode work on Windows, but the pipeline's bottleneck is detection (default `rfdetr-v5`). CPU RF-DETR on 768×768 inputs takes seconds per image; a 1-hour 30fps video is 108,000 frames — unusable.
4. Switching to a YOLO detection model (`lada-yolo-v4` etc.) bypasses RF-DETR → runs via PyTorch on ROCm, which is the *only* plausible usable scenario on Windows AMD (assuming the PyTorch ROCm build can be installed).

> **Conclusion**: On Windows AMD, the only viable scenario is **(somehow install PyTorch ROCm wheels) + YOLO detection + no secondary restoration + no unet-4x + no smart-render**. This is extremely fragile, hence the project's "experimental" label.

### 3.2 Why Linux AMD is *barely* usable
- PyTorch + onnxruntime-rocm are officially supported.
- MIGraphX compiles and caches into `<model>.migraphx/<key>/`; subsequent runs reuse the cache.
- AMF in ffmpeg is a well-trodden path.
- However: **no TensorRT BasicVSR++**, **no unet-4x at all**, **secondary restoration (other than `none` / external TVAI) fully blocked** — so "Linux AMD works" means "Linux AMD can finish the job", at a fraction of NVIDIA speed (roughly 1/3 to 1/5 in informal estimates).

---

## 4. Features explicitly disabled or regressed on AMD

| Feature | AMD behaviour | Code location |
|---|---|---|
| unet-4x secondary | `RuntimeError` | [engine_compiler.py#L130-L132](jasna/engine_compiler.py#L130-L132) |
| `--secondary-restoration` (anything ≠ `none`) | CLI + GUI both raise | [main.py#L638-L642](jasna/main.py#L638-L642), [gui/video_session.py#L96-L101](jasna/gui/video_session.py#L96-L101) |
| BasicVSR++ TensorRT sub-engines | Skipped, `use_tensorrt=False` always | [engine_compiler.py#L134-L137](jasna/engine_compiler.py#L134-L137), [basicvsrpp_mosaic_restorer.py#L33](jasna/restorer/basicvsrpp_mosaic_restorer.py#L33) |
| YOLO TensorRT engine | Not compiled; PyTorch path | [detection_registry.py#L164-L170](jasna/mosaic/detection_registry.py#L164-L170) |
| Smart-render (`forced-idr`) | `ValueError` | [video_encoder.py#L363](jasna/media/video_encoder.py#L363) |
| HLS streaming codec ≠ H.264 | Hard-coded H.264 AMF | [streaming_encoder.py#L126-L137](jasna/streaming_encoder.py#L126-L137) |
| RF-DETR on Windows AMD | Falls back to onnxruntime CPU | [migraphx_runner.py#L120-L161](jasna/mosaic/migraphx_runner.py#L120-L161) |
| `MIOPEN_FIND_MODE` | Defaults to `FAST` on ROCm | [accelerator.py#L15-L16](jasna/accelerator.py#L15-L16) |

---

## 5. Test coverage

`tests/test_amd_support.py` covers most branches with mocks:
- `test_rocm_uses_cuda_device_api_but_reports_amd` — ROCm builds still surface as `AcceleratorVendor.AMD`.
- `test_amd_basicvsrpp_skips_tensorrt_compilation` — confirms AMD never probes TRT.
- `test_amf_encoder_settings_are_vendor_specific` / `test_video_encoder_selects_amf_and_normalizes_cq` / `test_smart_render_is_rejected_on_amd` / `test_streaming_encoder_selects_amf` / `test_amf_decoder_context_is_created` / `test_amf_p010_host_input_reinterprets_signed_storage` — exercise the AMF encoder/decoder/HLS paths.
- `test_migraphx_runner_provider_and_tensor_bridge` / `test_migraphx_runner_falls_back_to_cpu_onnxruntime` — exercise the provider detection and CPU fallback.
- `tests/test_gui_engine_preflight.py::test_amd_preflight_checks_only_migraphx_cache` / `test_amd_preflight_needs_no_cache_for_cpu_onnxruntime` — exercise both preflight branches.

**Gaps**:
- No real-hardware end-to-end AMD test (impossible in CI).
- No `vram_offloader` AMD-specific test.
- Zero AMD coverage in `benchmark/`.
- No test for the CLI guard `--secondary-restoration` raising on AMD.

---

## 6. Remaining code / documentation issues

1. **MIGraphX fixed-shape requirement**: `MigraphxRunner` requires static `batch × 3 × 768 × 768`. Fine for RF-DETR, but a footgun for any other model.
2. **VRAM offloader is not vendor-agnostic at the code level**: hard-coded `cuda` string. Works under ROCm, but inconsistent with `accelerator.py`.
3. **Silent CPU fallback on Windows AMD**: if a user picks `rfdetr-v5`, the system silently switches to CPU. The GUI preflight does *not* surface a warning in this branch — the user should be told ("RF-DETR will run on CPU; expect very long detection").
4. **No compute-capability check for AMD**: `check_supported_gpu` returns `True` for any AMD GPU that has a `torch.cuda.get_device_name(...)`; older GCN cards may pass the gate and fail at MIOpen init.
5. **README claim "AMD preparation is much shorter"** is true on Linux (no TRT compile) but on Windows the implicit assumption is that the pipeline will run, which is not guaranteed.
6. **`_amf_host_input` is a no-op for 8-bit** (returns `packed` unchanged); harmless but redundant.
7. **YOLO on AMD has no preflight warning**: the first run's PyTorch warmup can be very long; the GUI should surface that.

---

## 7. Summary

**How much did the author actually implement for AMD?** Decoding (AMF hwaccel + software fallback), encoding (AMF for H.264/HEVC/AV1 + HLS), RF-DETR (MIGraphX + CPU fallback), YOLO (PyTorch on ROCm), BasicVSR++ (PyTorch), secondary-restoration guards, preflight, wizard, driver check — **every step has a branch**.

**Why is it still effectively unavailable?** Two compounding reasons:
1. **The bottleneck models have no AMD equivalent.** BasicVSR++ uses TensorRT sub-engines on NVIDIA for a 3.1–3.5× speedup; on AMD it is plain PyTorch. unet-4x is outright banned. So "AMD runs" = "AMD runs an order of magnitude slower than NVIDIA".
2. **Windows has no PyTorch ROCm and no onnxruntime-rocm**, so RF-DETR degenerates to CPU inference — the whole pipeline becomes unusable. **Linux** is usable-but-slow; **Windows** for the default RF-DETR model is essentially unusable, and you must switch to a YOLO detection model to even have a chance of finishing a job.

> For end users, AMD today is an **experimental Linux-only slow path**. Windows AMD is a do-it-yourself option for the PyTorch-ROCm-on-Windows community, not a product-grade capability.
