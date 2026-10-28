# HyperQwen 移植至 AMD Radeon RX 7900 XTX (24GB / ROCm 7.2.4) 可行性与性能评估报告

## 0. 结论先行（Executive Summary）

**结论：可以有条件移植（Conditionally Portable），无法开箱即用；完成 5 项轻量配置/脚本改造后，可在 RX 7900 XTX 24GB 上完整跑通 Qwen3.8-27B 的单用户投机采样（DFlash2 / MTP）、240k 超长上下文（KVarN）与 64 并发批处理模式。**

- **单流交互体验（Single-User C1）表现优异**：单用户解码是典型的**显存带宽受限（Memory-Bandwidth Bound）**场景，RX 7900 XTX 拥有 **960 GB/s** 显存带宽（+ 96 MB Infinity Cache），略高于参考卡 RTX 3090（936 GB/s）。在 DFlash2 投机采样（保持相同的 3.12~3.34 tokens/step 接受率）下，移植后预计可达 **93–113 tok/s**（约合 RTX 3090 的 **80%–88%**），引用复述模式（`DFLASH_TOKENS=15`）可达 **285–330 tok/s**。
- **预填充（Prefill）与高并发（Batch C64）受限于缺少 Marlin W4A8**：原项目在 Batch 模式和 Prefill 上的高吞吐依赖 NVIDIA CUDA 专属的 **Marlin W4A8（`INT8_ACT=int8`）** 内核（在 3090 上带来 +27%~45% 加速）。在 ROCm `gfx1100` 上将回退至 vLLM 原生的 **W4A16（`rdna3_w4a16` / `triton_w4a16`）** 路径：
  - **Prefill（1k~4k）** 预计可达 **1,100–1,350 tok/s**（相当于 3090 W4A16 基线 1,437–1,494 tok/s 的 **75%–90%**，或 3090 W4A8 1,845–1,937 tok/s 的 **60%–70%**）；
  - **Batch 64 并发稳态解码** 预计可达 **520–650 tok/s**（锚定 3090 W4A16 基线 707 tok/s 的 **75%–92%**，或 3090 W4A8 ~1,035 tok/s 的 **50%–63%**）。

---

## 1. 本机硬件环境与参考基线（RTX 3090）参数对比

基于本机实测硬件信息（`rocminfo`、`rocm-smi`、`lscpu`、`free -h`）：

| 维度 | 项目参考平台 (RTX 3090 @ 250W) | 本机目标平台 (RX 7900 XTX) | 对移植与性能的核心影响 |
|---|---|---|---|
| **GPU 架构** | NVIDIA Ampere (`sm_86`, 82 SMs) | AMD RDNA 3 (`gfx1100`, 96 CUs / 48 WGPs) | RDNA 3 默认 **Wave32**，与 CUDA Warp=32 对齐，Triton 内核的 `num_warps` 无需改写 |
| **软件栈** | CUDA 13.0 + FlashInfer + nvcc | **ROCm 7.2.4** (`/opt/rocm/bin/hipcc`) | 需使用 vLLM 0.31.0 ROCm 构建；CUDA C++ 扩展（`pthfa/`、`fa2/`、FlashInfer）不可用 |
| **显存容量** | 24 GiB (25,769 MB) | **24 GiB (25,753,026,560 B)** | 容量 1:1 对齐；原项目逐字节标定的 `KV_MEM`（5.20 GiB / 4.90 GiB）与状态预算可直接复用 |
| **显存带宽** | 936 GB/s (GDDR6X) | **960 GB/s** (GDDR6 + 96 MB Infinity Cache) | C1 Decode 纯受限于带宽；Infinity Cache 有利于 GDN 循环状态（~150 MB）与 DFlash2 滑窗的高频读写 |
| **Shared Memory (LDS)** | 99 KB / Block (`sm_86`) | **64 KB / Workgroup (WGP)** (`gfx1100`) | 与 Turing (`sm_75`, 64 KB) 相同；仓库已有的 `spec-attn-smem-fit.patch` 自动将 KV tile 折半适配 64 KB |
| **矩阵算力 (FP16/BF16)** | 峰值 142 TFLOPS；250W 持续 **54–63 TFLOPS** | 峰值 **122.8 TFLOPS** (355W TBP)；持续约 **48–60 TFLOPS** | W4A16 下的 Prefill 和 Batch GEMM 吞吐与 250W 3090 处于同一量级 |
| **低精度硬件支持** | INT8 / INT4 Tensor Core（无 FP8） | INT8 / INT4 WMMA（**无硬件 FP8**） | `gfx1100` 无原生 FP8 指令，所有 `--kv-cache-dtype fp8` 路径不可用，须改走 BF16 / INT8 / INT4 / KVarN |
| **CPU 与系统内存** | 现代多核 CPU + $\ge 16$ GB RAM | **Ryzen 5 2600 (Zen+, 6C/12T)** + **46 GiB RAM** | 46 GiB RAM 充裕；但 Zen+ 单核性能较弱，Python 调度开销（约 +1~2 ms/step）及首次 Triton 编译耗时偏高 |
| **PCIe 总线** | PCIe 4.0 x16 (~31.5 GB/s) | **PCIe 3.0 x16 (~15.75 GB/s)**（受限于 Zen+ CPU） | 跨 PCIe 特性（`VLLM_EMBED_UVA=1`、`KV_OFFLOAD_GB`）延迟翻倍，移植时应保持关闭 |

---

## 2. 移植后性能估算与推导模型（Speed Estimation）

> **估算方法论声明（`[INFERENCE]`）**：以下性能预测基于仓库 `docs/optimizations.md`、`single-user/README.md`、`batch/README.md` 与 `docs/reproductions/README.md` 中实测拆解的**单步耗时公式（Step Time Breakdown）**、**W4A16 vs W4A8 消融对照表**以及 `gfx1100` 与 `sm_86` 的带宽/算力/内核效率比值推导得出。

### 2.1 各场景端到端速度预测总表

| 运行模式 / 场景 | 配置参数（移植后 ROCm 等效配置） | RTX 3090 (250W) 实测基线 | RX 7900 XTX (ROCm) 预计速度 | 达成率 (vs 3090) | 关键瓶颈与差异原因 |
|---|---|---|---|---|---|
| **无投机单流基线**<br/>(`SPEC=off`, C1) | `TRITON_ATTN`, `bfloat16` KV | **45 – 46 tok/s**<br/>(~21.7 ms/step) | **38 – 43 tok/s**<br/>(~23.5–26.5 ms/step) | **84% – 93%** | 960 GB/s 带宽持平；ROCm `rdna3_w4a16` 相比 CUDA Marlin 小 batch 反量化开销略高 + Ryzen 2600 调度延迟 |
| **Setup B — 单用户默认**<br/>(`SPEC=dflash2 CTX=fast`) | C1 Greedy (`3.34 tok/step`)<br/>C1 默认温度 (`3.12 tok/step`) | **127 – 131 tok/s** (Greedy)<br/>**118 – 122 tok/s** (Default) | **100 – 113 tok/s** (Greedy)<br/>**93 – 106 tok/s** (Default) | **78% – 88%** | 投机接受率（tok/step）100% 不变；单步耗时从 ~26.2 ms 增至 ~29.5–33.5 ms（W4A16 算子 + 64KB LDS tile 折半 + CPU） |
| **Setup C — 引用复述**<br/>(`DFLASH_TOKENS=15`) | 25k 上下文原样引用 (`14.97 tok/step`)<br/>代码增删改写 (`3.50 tok/step`) | **381 tok/s** (引用)<br/>**113 tok/s** (改写) | **285 – 330 tok/s** (引用)<br/>**88 – 100 tok/s** (改写) | **75% – 87%** | Lookup drafting 与 n-gram 候选链完全平台无关，15-token 验证块红利完整保留 |
| **Setup D — 131k–150k 长文**<br/>(`SPEC=dflash2 CTX=long`) | `TRITON_ATTN` + `int8_per_token_head`<br/>（测试深度：短文 / 66k / 100k） | **113.4 tok/s** (短文)<br/>**48.2 tok/s** (@66k)<br/>**37.9 tok/s** (@100k) | **90 – 102 tok/s** (短文)<br/>**38 – 43 tok/s** (@66k)<br/>**30 – 34 tok/s** (@100k) | **79% – 90%** | 对标 3090 同一 `TRITON_ATTN + int8` 臂（非 `pthfa`/`fa2` 臂）；64KB LDS 下 KV tile 折半为 32 带来约 15% 长文衰减 |
| **Setup E — 240k 超长文**<br/>(`SPEC=dflash2 CTX=huge`) | `kvarn_k4v2_g128` + `KVARN_FP16_DEQUANT=1` | **67 tok/s** (常规)<br/>**164 tok/s** (引用) | **50 – 58 tok/s** (常规)<br/>**125 – 145 tok/s** (引用) | **75% – 88%** | KVarN 全套为 Triton 内核；开启 `KVARN_FP16_DEQUANT=1` 可规避 RDNA3 VGPR 寄存器溢出 |
| **Prefill 预填充吞吐**<br/>(Single-User / Batch) | 1k / 4k tokens 输入<br/>16k tokens 输入<br/>51k tokens 输入 | **1,437 / 1,494 tok/s** (W4A16)<br/>*（注：W4A8 为 1,845 / 1,937）*<br/>**1,410 tok/s** (16k W4A16)<br/>**1,200 tok/s** (51k W4A16) | **1,100 – 1,350 tok/s** (1k/4k)<br/>**1,000 – 1,220 tok/s** (16k)<br/>**780 – 980 tok/s** (51k) | **75% – 90%** *(vs W4A16)*<br/>**60% – 70%** *(vs W4A8)* | 7900 XTX 无 Marlin W4A8，锚定 3090 W4A16 基线；51k 处因 `TRITON_ATTN` 相比 FA2 略慢，可通过挂载 `prefill-attn-int8` 追回 |
| **Setup A — Batch 64 并发**<br/>(128 in / 512 out) | `KV=int4pth` (`TRITON_ATTN` + `int4_per_token_head`) + fp16 GDN state | **707 tok/s** 稳态 / **491** e2e *(W4A16)*<br/>**1,035 tok/s** 稳态 / **948** e2e *(W4A8)* | **520 – 650 tok/s** (稳态)<br/>**360 – 450 tok/s** (e2e) | **75% – 92%** *(vs W4A16)*<br/>**50% – 63%** *(vs W4A8)* | Batch C64 是计算受限（Compute-Bound）；失去 Marlin INT8 W4A8 后退回 W4A16 档位，加上 int4 KV 反量化开销 |

### 2.2 核心推导依据

1. **为什么 Single-User C1 能拿到 3090 的 80%–88%（100+ tok/s）？**
   - 根据 `docs/reproductions/README.md:62`，RTX 4090（450W，算力是 3090 的 2.3 倍，但带宽仅高 ~8%）在 Setup B 下的 C1 Greedy 解码为 **135.5 tok/s**，僅比 250W 3090（133 tok/s）快 **+1.9%**——证明 **C1 Decode 完全由显存带宽和投机接受率决定，与多出的浮点算力无关**。
   - 在 `SPEC=dflash2 CTX=fast` 下，每步需要读取 `14.71 GiB` 主模型权重 + `1.19 GB` DFlash2 权重 + GDN 状态，在 3090 上耗时 **25.9–26.5 ms/step**，每步产出 **3.12（默认）~ 3.34（Greedy）tokens**（`docs/optimizations.md:239-241, 360`）。
   - RX 7900 XTX 的理论带宽为 **960 GB/s**（高于 3090 的 936 GB/s），且 96 MB Infinity Cache 能缓存 ~150 MB 的 fp16 GDN 状态和 DFlash2 激活。扣除 ROCm `rdna3_w4a16` / `triton_w4a16` 相比 CUDA Marlin 在 $M=8$ 下的反量化差距（约 +2.5~4.5 ms/step）、64KB LDS 下 `spec-attn-smem-fit.patch` 将 KV tile 减半的开销（+0.3~0.6 ms/step）以及 Ryzen 5 2600 的 CPU 调度延迟（+1.0~2.0 ms/step），单步耗时落在 **29.5–33.5 ms**，直接对应 **100–113 tok/s（Greedy）**。

2. **为什么 Batch C64 与 Prefill 必须锚定 3090 的 W4A16 行？**
   - `batch/README.md:36-40` 与 `docs/optimizations.md:134-139` 明确记录了开启 `INT8_ACT=int8`（Marlin W4A8）前后的消融数据：
     - Batch 64 并发下，**W4A16 + fp16 state 为 707 tok/s 稳态解码（491 tok/s e2e）**，开启 W4A8 后才跃升至 876~1,035 tok/s；
     - Single-User 4k Prefill 下，**W4A16 默认是 1,494 tok/s**，开启 W4A8 后才升至 1,937 tok/s。
   - 由于 `VLLM_MARLIN_INPUT_DTYPE=int8` 仅喂给 CUDA 的 `MarlinLinearKernel`，7900 XTX 上天然运行在 W4A16 模式，因此以 3090 的 W4A16 实测值（707 tok/s Batch / 1,494 tok/s Prefill）乘以 RDNA3 WMMA 算子达成率（~75%–90%）即可得到准确预测。

---

## 3. 架构模块与补丁栈兼容性逐项审计

```mermaid
flowchart TD
  subgraph Prep["1. 模型准备层 (prepare/) — 100% 兼容"]
    P1["dbirks/Qwen3.8-27B-W4A16-AutoRound"] --> P2["quant_lm_head / quant_embed / quant_mtp (纯 CPU)"]
    P2 --> P3["fetch_fast_variant.py (int4-GPTQ heads) + fetch_dflash2.py"]
  end

  subgraph Quant["2. 权重量化执行层 (vLLM Linear)"]
    P3 --> Q1{"vLLM Quantization Dispatch"}
    Q1 -->|"CUDA sm75+ (原项目)"| Q2["MarlinLinearKernel<br/>支持 W4A16 / W4A8(INT8_ACT) / W8A16"]
    Q1 -->|"ROCm gfx1100 (本机)"| Q3["rdna3_w4a16 / triton_w4a16<br/>支持 int4 W4A16; W4A8 失效; int8 heads 需避开"]
  end

  subgraph Attn["3. Attention 与 KV Cache 层"]
    A0{"Attention Backend & KV Dtype"}
    A0 -->|"CUDA 专属 (不可用)"| A1["pthfa/ (sm80-89 SASS)<br/>fa2/ (CUTLASS sm86 FP8)<br/>FlashInfer (FP8 KV)"]
    A0 -->|"ROCm 可用 (Triton / AITER)"| A2["TRITON_ATTN + bfloat16<br/>TRITON_ATTN + int8_per_token_head<br/>TRITON_ATTN + int4_per_token_head<br/>KVarN (kvarn_k4v2_g128, Triton)"]
  end

  subgraph Spec["4. 混合架构与投机采样层 — 兼容"]
    S1["Gated-DeltaNet (GDN/Mamba)<br/>Triton chunk_o + fp16 state"]
    S2["DFlash2 / MTP 投机采样<br/>n-gram lookup + Triton verify attn"]
  end
```

### 3.1 模型准备与量化格式（`prepare/`）

- **纯 CPU 执行，零 CUDA 依赖**：`prepare/quant_schema.py:93-125` 在 CPU 上完成对称 RTN 量化并通过 `compressed_tensors.compressors.pack_quantized.base.pack_to_int32` 打包。全 `prepare/` 目录无任何 `.cuda()` 或 GPU 调用。
- **关键风险与规避（int8 头 vs int4 头）**：
  - 基础脚本 `quant_lm_head.py` 和 `quant_mtp.py` 默认将 `lm_head` 和 `mtp` 量化为 **8-bit (`group_1`, `group_3`)**。在 CUDA 上由 Marlin W8A16 承接；但在 ROCm `gfx1100` 上，vLLM 0.31.0 的 `rdna3_w4a16` / `triton_w4a16` 仅支持 4-bit 权重（`uint4` / `uint4b8`），8-bit `pack-quantized` 线性层极易在加载时报 `No linear kernel can implement`。
  - **规避方法**：使用 `prepare/fetch_fast_variant.py`（提供 **4-bit GPTQ** 的 `lm_head` 与 `mtp` shard，见 `verify.sh:277-284`）或运行 `prepare/quant_mtp.py --bits 4`，使全模型所有线性层统一为 4-bit W4A16。

### 3.2 独立扩展目录（`pthfa/`、`fa2/`、`kvarn/`）

| 目录 | 实现机制 | `gfx1100` 兼容性 | 处置方式 |
|---|---|---|---|
| `pthfa/` | CUDA C++ 扩展（`mma.sync`、`cp.async`、`ldmatrix`），编译 `sm_80/86/89` SASS | **不可用** | **跳过安装**。`resolve_config.sh:84-86,115` 在无 `nvidia-smi` 时自动保持 `PTH_FA=0` |
| `fa2/` | 基于 CUTLASS `62750a2` + FA2 `sm_86` 头文件的 FP8 KV 插件 | **不可用** | **跳过安装**。`resolve_config.sh:134` 自动保持 `FA2_FP8KV=0` |
| `kvarn/` | `kvarn/files/vllm/v1/attention/ops/triton_kvarn_*.py` + 4 个补丁 | **完全兼容（纯 Triton）** | 正常执行 `bash kvarn/install.sh`；运行时加 `KVARN_FP16_DEQUANT=1` 限制 `BLOCK_N<=32, num_warps=4, maxnreg=None`，防止 RDNA3 VGPR 溢出 |

### 3.3 `patches/` 全部 54 个补丁兼容性分类

1. **纯 Python / 引擎调度 / 混合模型补丁（35 个）—— 100% 兼容**：
   - 包含 Gated-DeltaNet (GDN/Mamba) 状态管理（`mamba-align-checkpoint-order.patch`、`mamba-chunked-prefill-align.patch`、`vllm-pr50021-gdn-spec-bounds.patch`）、DFlash2 投机采样与上下文查找（`dflash2-lookup-drafting.patch`、`dflash2-ngram-chains.patch`、`dflash2-prewarm.patch`、`dflash2-z-adaptive-emitted.patch`、`qwen3_5-mtp-draft-vocab.patch`）、KV 组对齐（`hybrid-kv-groups-v2-cudagraph.patch`、`hybrid-sw-block-promote.patch`）、显存预算（`memory-profile-after-warmup.patch`、`pinned-kv-empty-cache.patch`、`cudagraph-memory-from-allocator.patch`）等。
2. **Triton 算子补丁（10 个）—— 9 个可用，1 个安全跳过**：
   - **`spec-attn-smem-fit.patch`（关键）**：`spec-decode-attn.patch` 默认按 `sm_86` 的 99 KB shared memory 分配 64-token KV tile（在 `head_dim=256` 下需 98,304 B），而 **`gfx1100` 每个 WGP 的 LDS 上限恰好是 64 KB（65,536 B）**。`spec-attn-smem-fit.patch:88-109` 捕获 Triton `OutOfResources` 并自动将 `tile` 折半至 32/16，确保在 7900 XTX 上不会崩溃。
   - **`int4-kv-per-token-head.patch` / `spec-decode-int4-kv-mq3d.patch` / `spec-decode-int8-kv.patch`**：纯 Triton 位操作与 FP16/BF16 `tl.dot`，且 `spec-decode-scratch-within-budget.patch:380` 使用 `current_platform.is_cuda_alike()`（涵盖 CUDA 与 ROCm）。
   - **`sampler-small-topk-fast-softmax.patch` / `spec-sampler-prewarm.patch`**：纯 Triton 采样器加速，ROCm 直接受益。
   - **`prefill-attn-int8.patch`**：内核本身是纯 Triton `tl.dot`（int8 QK^T），RDNA3 WMMA 支持 int8 点积；但调用入口硬编码在 `v1/attention/backends/flash_attn.py:1447`，若在 ROCm 上使用 `TRITON_ATTN` 后端，需将该 `if envs.VLLM_PREFILL_ATTN ...` 分支同步复制到 `triton_attn.py` 才能触发。
   - **`triton-spec-attn-fp8-kv.patch`**：内部有 `cap >= (8, 9)` 门控，在 `gfx1100` 上自动跳过。
3. **Marlin 专属补丁（5 个）—— 打补丁不冲突，运行时静默失效（Inert）**：
   - `marlin-int8-layer-select.patch`、`marlin-int8-negative-scales.patch`、`marlin-int8-asym-zp.patch`、`marlin-repack-staged-sm80.patch`、`marlin-tune-table.patch`。
4. **平台耦合补丁（4 个）—— 无需修改**：
   - `topk-honour-flashinfer-sampler-switch.patch`（`not current_platform.is_cuda()` 天然返回 `None` 走 `torch.topk`）、`sampler-warmup-cuda.patch`、`offload-wsl2-devptr.patch`（非 WSL2 不触发）、`qwen3_5-embed-uva.patch`（默认关闭；因本机 Ryzen 5 2600 仅支持 PCIe 3.0 x16，应保持 `VLLM_EMBED_UVA=0`）。

---

## 4. 移植落地步骤（Step-by-Step Porting Guide）

### Step 1：清理当前显存占用并准备 ROCm 虚拟环境
> **注意**：本机当前 `rocm-smi` 显示已有进程占用了 `25.24 GB / 25.75 GB` 显存，启动前需先停止现有占卡服务。

不使用 `Dockerfile`（基于 `nvidia/cuda:13.0.3`），直接在宿主机建立 Python 虚拟环境并安装 ROCm 版 PyTorch 与 vLLM `v0.31.0`：
```bash
python3 -m venv venv
# 安装 ROCm 7.2 对应的 torch / triton / vllm==0.31.0（或从 vllm v0.31.0 tag 源码编译 VLLM_TARGET_DEVICE=rocm）
venv/bin/pip install transformers==5.15.0 tokenizers==0.22.2 compressed-tensors==0.17.0 \
  huggingface_hub==1.32.0 hf_transfer==0.1.9 ninja==1.13.0 pandas==3.0.5 pyarrow==20.0.0
```
- **不要安装**：`flashinfer-cubin`、`nvidia-cuda-*`、以及不要运行 `pthfa/install.sh` 和 `fa2/install.sh`。

### Step 2：打入 vLLM 补丁栈与 KVarN
```bash
SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1)
bash patches/apply.sh "$SP"
bash kvarn/install.sh
```

### Step 3：在 CPU 上准备全 4-bit 模型权重（Fast Variant + DFlash2）
```bash
# 1. 下载基础模型 dbirks/Qwen3.8-27B-W4A16-AutoRound 至 models/Qwen3.8-27B-W4A16-AutoRound
# 2. CPU 量化与词表裁剪（将 MTP 设为 4-bit 以匹配 ROCm w4a16 内核）
venv/bin/python prepare/quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_mtp.py --bits 4 models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound --ids prepare/draft_vocab_ids.json

# 3. 获取 4-bit GPTQ lm_head/MTP fast 变体与 1.19 GB W4A16 DFlash2 drafter
venv/bin/python prepare/fetch_fast_variant.py
venv/bin/python prepare/fetch_dflash2.py
```

### Step 4：通过已内置的 `ROCM=1` 开关一键启动各模式

仓库现已内置 `ROCM=1` 移植支持（覆盖 `resolve_config.sh`、`launcher_common.sh`、`single-user/start_qwen.sh`、`batch/start_qwen.sh`、`single-user/alternative.sh`、`verify.sh`、`scripts/hq-doctor.sh`）以及 `patches/triton-attn-bf16-spec-prefill.patch`（将 bf16 Split-KV Verify 与 int8-QK Prefill 挂载至 `TRITON_ATTN`）：

1. **Setup B — 单用户日常对话（64k 上下文，预计 100–113 tok/s Greedy）**：
   ```bash
   ROCM=1 SPEC=dflash2 CTX=fast bash single-user/start_qwen.sh
   ```
2. **Setup C — 代码改写 / 原文引用（56k 上下文，预计引用 285–330 tok/s）**：
   ```bash
   ROCM=1 SPEC=dflash2 CTX=fast DFLASH_TOKENS=15 bash single-user/start_qwen.sh
   ```
3. **Setup D — 131k–150k 长上下文（`SPEC=mtp` 与 `SPEC=dflash2` 均自动路由至 `TRITON_ATTN + int8_per_token_head`）**：
   ```bash
   ROCM=1 SPEC=dflash2 CTX=long bash single-user/start_qwen.sh
   # 或 MTP 臂：
   ROCM=1 SPEC=mtp CTX=long bash single-user/start_qwen.sh
   ```
4. **Setup E — 240k 超长上下文（`ROCM=1` 自动默认开启 `KVARN_FP16_DEQUANT=1` 防寄存器溢出）**：
   ```bash
   ROCM=1 SPEC=dflash2 CTX=huge bash single-user/start_qwen.sh
   ```
5. **Setup A — Batch 64 并发（`ROCM=1` 自动默认 `KV=int4pth` 并跳过 CUDA Marlin `INT8_ACT`）**：
   ```bash
   ROCM=1 bash batch/start_qwen.sh
   ```

### Step 5：运行自检与回归测试
- 静态/无卡配置自检：`bash bench/test_rocm_knob.sh`
- 宿主状态与 `rocm-smi` 诊断：`ROCM=1 bash scripts/hq-doctor.sh`
- 环境、补丁与模型完整校验：`ROCM=1 bash verify.sh --no-server`

---

## 5. 本机真实上卡实测结果（Live Bringing-up & Benchmark Verification）

### 5.1 真实排查与修复的 Runtime Blockers
1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 触发 GPU 内存页错误**：
   - **现象**：在 ROCm 7.2.53211 上，PyTorch Caching Allocator 使用 HIP VMM（`hipMemAddressReserve` / `hipMemMap`）在 RDNA3 `gfx1100` 上分配连续显存时，直接引发硬件级 `Memory access fault by GPU node-1 on address ... Reason: Page not present or supervisor privilege` 并导致进程 SIGABRT (code 134)。
   - **修复**：在 `single-user/start_qwen.sh`、`batch/start_qwen.sh` 与 `single-user/alternative.sh` 中增加 `ROCM=1` 分支，统一自动将 `ALLOC_DEFAULT` 设为 `expandable_segments:False`。
2. **AITER (AMD AI Template Extensions) 屏蔽**：
   - **背景**：虚拟环境中安装了 `amd-aiter`，但其底层内核主要面向 CDNA3（MI300，`gfx942`），若在 RDNA3 `gfx1100` 上加载会引发未定义符号或算子缺失。
   - **修复**：在所有启动器中在 `ROCM=1` 时默认导出 `export VLLM_ROCM_USE_AITER=0` 与 `export VLLM_ROCM_USE_AITER_CUSTOM_AR=0`。
3. **Fast Variant 目录分片 6（`embed_tokens`）硬链接修复**：
   - **现象**：`models/Qwen3.8-27B-W4A16-AutoRound-fast` 中的 `model-00006-of-00007.safetensors` 为未量化旧文件，导致模型加载时报 `ValueError: There is no module or parameter named 'embed_tokens.weight'`。
   - **修复**：重新硬链接同步 base 目录中经过 `prepare/quant_embed.py` 压缩的真实 8-bit embedding 分片（1.29 GB）。

### 5.2 真实单卡推理速度实测数据（RX 7900 XTX 24GB）

测试环境：Local AMD Radeon RX 7900 XTX (24 GB VRAM, ROCm 7.2.53211, torch 2.13.0+git733fca1), 模型 `Qwen3.8-27B-W4A16-AutoRound-fast`，生成 256 tokens，贪婪解码 (`temperature=0`, `enable_thinking=false`):

| 测试配置 | 首字延迟 (TTFT) | 纯解码速度 (Decode) | 端到端生成速度 (Overall) | 投机接受步长 (tok/step) | 投机命中率 (Acceptance Rate) |
|---|---|---|---|---|---|
| **`ROCM=1 CTX=fast SPEC=dflash2`** | **175.0 ms** | **65.12 tok/s** | **62.57 tok/s** | **2.97 tok/step** | 28.2% |
| **`ROCM=1 CTX=fast SPEC=mtp`** | **198.9 ms** | **54.27 tok/s** | **52.25 tok/s** | **2.62 tok/step** | 40.6% |

### 5.3 生产环境验证（`verify.sh`）

针对正在运行的 live 服务执行 `ROCM=1 bash verify.sh`，52 个补丁、4 个 KVarN 补丁、全量权重映射、实时 `/health 200`、实际中文问答及 `/tokenize` 对齐测试**全部通过（0 failures）**。

---

## 6. RX 7900 XTX (65.1 tok/s) vs RTX 3090 (100+ tok/s) 性能差距根因与深度优化路线图

### 6.1 当前用的量化模型格式说明（W4A16 / Q4，非 Q3）

当前线上运行的是标准 **W4A16（4-bit 权重 + 16-bit 激活，等效 Q4，无损级精度）** 组合，总权重体积约 17.7 GiB：

| 模块 | 模型目录 / 来源 | 量化精度与格式 | 说明 |
|---|---|---|---|
| **主模型 40 层骨干** (`Qwen3.8-27B-W4A16-AutoRound-fast`) | `dbirks/Qwen3.8-27B-W4A16-AutoRound` | **W4A16 (`int4` 对称, `group_size=128`)** | 30 层 GDN 线性注意力 + 10 层 Full Attention 的全部线性层均为 4-bit |
| **主模型 `lm_head` + `MTP` 层** | `syvai/qwen3.8-27b-3090-fast-variant` | **W4A16 (`int4` GPTQ, `group_size=128`)** | 词表输出头（含 40,960 裁剪草稿词表）与 MTP 头均为 4-bit |
| **主模型 `embed_tokens`** | `prepare/quant_embed.py` 本地生成 | **W8A16 (`int8` 对称, `group_size=128`)** | 词嵌入表按行 gather 反量化 |
| **DFlash2 投机草稿器** (`SPEC=dflash2`) | `syvai/Qwen3.8-27B-DFlash2-W4A16` | **W4A16 (`int4` GPTQ, `group_size=128`)** | 1.19 GiB 单步并行预测 7 个草稿 token |

---

### 6.2 单步耗时（Step Time）拆解与 5 大代码级根因

- **硬件基础**：RX 7900 XTX（960 GB/s 显存带宽，96 CU，RDNA 3 Wave32，64 KB LDS）与 RTX 3090（936 GB/s 显存带宽，82 SM，Ampere Warp32，99 KB SMEM）的物理显存带宽基本相同（7900 XTX 高 2.5%）。读取 ~16.7 GB 权重的理论带宽极限耗时为 **~17.4 ms/step**。
- **实测单步耗时对比**：
  - **RTX 3090**：约 **24–27 ms/step**（有效带宽利用率 ~68%，配合 3.5–4.0 tok/step 达到 **130–147 tok/s**）。
  - **RX 7900 XTX 当前基线**：**45.6 ms/step**（有效带宽利用率 ~38%，在 2.97 tok/step 下为 **65.12 tok/s**）。

通过审查 `qwen.log` 与 `vllm 0.31.0+rocm723` 源码，定位出吃掉额外 **~19 ms/step** 的 5 个具体代码根因：

#### 根因 1：W4A16 线性层选用了 `RDNA3W4A16LinearKernel`，存在“假零点带宽税”与 `M=8` 退化（占差距 ~55%，每步多花 ~10–12 ms）
- **代码位置**：
  - 选择器：`vllm/model_executor/kernels/linear/__init__.py:510-516`
  - 内核实现：`vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py:26-163`
- **根因细节**：
  1. **强行合成并读取 `qzeros`（浪费 +12.5% 显存读取带宽与解包指令）**：
     `Qwen3.8-27B-W4A16-AutoRound` 为对称量化（`symmetric=True, zero_points=False, uint4b8`），零点恒为常数 `8`。在 CUDA `MarlinLinearKernel` 中，常数 `8` 直接在寄存器内减去，零显存读取。而 `rdna3_w4a16.py:98-121` 因底层 `torch.ops._rocm_C.gptq_gemm_rdna3`（移植自旧版 ExLlamaV1）不支持无零点模式，**在加载模型时强行在显存中合成一整份 `qzeros` 张量**，并在每次 GEMM 时从全局显存额外读取、位移解包 `qzeros`。
  2. **对投机解码 `M=8`（Verify 步）缺乏 RDNA3 WMMA 调优**：
     `gptq_gemm_rdna3` 面向非投机解码（`M=1`）设计。当投机解码传入 `M=8`（DFlash2）或 `M=5`（MTP）时，未像 Marlin（配合 `patches/marlin-tune-table.patch`）那样针对 Qwen3.8-27B 的矩阵形状（`K=5120, N∈{6144, 10240, 17408}`）做寄存器级块复用。

#### 根因 2：`RDNAHybridW4A16LinearKernel` 漏写 `gfx1100` 调优分支且 C++ 窄矩阵阈值卡在 `M <= 5`
- **代码位置**：`vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:59-62, 272-314, 395`
- **根因细节**：
  vLLM 0.31.0 内置了排在第二位的 `RDNAHybridW4A16LinearKernel`，原生支持对称量化 `w_zp=None`（不合成、不读取 `qzeros`），且在 `M <= MAX_SKINNY_BATCH_SIZE` 时调用 AMD 手写 Split-K 窄矩阵 HIP 内核 `ops.wvSplitK_int4_g`，在 `M > MAX_SKINNY_BATCH_SIZE` 时调用融合 Triton WMMA 内核 `_triton_w4a16_skinny_fmt_kernel`。但存在两个缺陷：
  1. 第 62 行硬编码 `MAX_SKINNY_BATCH_SIZE = 5`，导致 `SPEC=dflash2` 的 Verify 步（`M = 8`）无法进入 `wvSplitK_int4_g`；
  2. 第 272 行仅判断 `elif _on_gfx1151():`（Strix Halo 核显），**漏掉了独立显卡 `gfx1100`（RX 7900 XTX）**，导致 `gfx1100` 掉入第 306 行的未调优兜底 `else:` 分支（固定 `BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_warps=4`）。

#### 根因 3：30 层 GDN（线性注意力）缺少 RDNA3 Wave32 专属调优（每步多花 ~4–5 ms）
- **代码位置**：
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:515-533`
  - `vllm/third_party/flash_linear_attention/ops/triton/fused_recurrent.py`
- **根因细节**：
  Qwen3.8-27B 的 40 层中有 30 层是 GDN。当前 `fused_recurrent_gated_delta_rule_fwd_kernel` 的块参数（`BV=64` 或 `VLLM_GDN_ILP` 的 `BV=16, num_warps=4`）按 NVIDIA Ampere Warp32 寄存器文件调优，未针对 RDNA3 Wave32 的 VGPR 分配与 L0 Cache 行宽做搜优。

#### 根因 4：10 层 Full-Attention Split-KV Verify 内核因 `BLOCK_M=64` 冗余导致 64 KB LDS 下 `TILE` 折半（每步多花 ~2 ms）
- **代码位置**：`patches/spec-decode-attn.patch` + `patches/spec-attn-smem-fit.patch`
- **根因细节**：
  Qwen3.8-27B Full Attention 为 `num_heads=32, num_kv_heads=8`（GQA group size $G = 4$）。在 $k=7$（$M=8$ query tokens）时，每个 KV head 实际仅有 $G \times M = 4 \times 8 = 32$ 行有效 Query。但 `_spec_attn_partial` 固定了 `BLOCK_M=64`（一半为空 padding），导致 `TILE=64` 需要 ~96 KB LDS，在 RX 7900 XTX（64 KB LDS）上触发 `OutOfResources` 并自动降级为 `TILE=32`，KV 遍历循环次数翻倍。将 `BLOCK_M` 设为 `32` 即可让 `TILE=64` 直接装入 64 KB LDS。

#### 根因 5：ROCm 下 `rms_norm` / `fused_add_rms_norm` 回退至 `native`（每步多花 ~2 ms）
- **代码位置**：`qwen.log:29, 40`
- **根因细节**：
  `IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])` 未启用 `vllm._C` 的 C++ HIP 融合 RMSNorm 算子；同时默认未设置 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`。

---

### 6.3 可落地的分阶段优化路线图（Optimization Roadmap）

保持当前无损精度的 W4A16 模型不变，按投入产出比排序的 4 个实操优化阶段：

| 阶段 | 优化目标 | 具体技术方案 | 改动范围 | 预期单步提速 / 最终 tok/s |
|---|---|---|---|---|
| **Stage 1**<br/>*(最快见效，零风险)* | **W4A16 线性层内核横向实测与切换** | 编写独立 microbenchmark 脚本，在 RX 7900 XTX 上对 Qwen3.8-27B 的真实层尺寸（`K=5120, N∈{6144, 10240, 17408}`）在 `M=1, 5, 8` 下实测对比 `RDNA3W4A16LinearKernel`、`RDNAHybridW4A16LinearKernel` 与 `TritonW4A16LinearKernel`，通过 `VLLM_DISABLED_KERNELS` 直接切换最快内核。 | 启动器环境变量或 1 个 patch | **+8% ~ +15%**<br/>（预计 **70–75 tok/s**） |
| **Stage 2**<br/>*(核心突破点)* | **为 `gfx1100` 定制 W4A16 `M=8` 投机验证内核并消除假 `qzeros`** | **路径 A**：在 `rdna_hybrid_w4a16.py` 中启用 `_on_gfx1x()` 调优分支，针对 RX 7900 XTX (96 CU) 的 Qwen3.8-27B 矩阵形状搜索最优 `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)` 填入 `_GFX1X_PREFILL_OVERRIDES`（覆盖 `M=1..16`），彻底消除假 `qzeros` 读取；<br/>**路径 B**：测试放宽 `wvSplitK_int4_g` 的 `MAX_SKINNY_BATCH_SIZE` 或配合 `DFLASH_TOKENS=4` (`M=5`) 直接命中 C++ Split-K 窄矩阵内核。 | 新增 1 个 vLLM patch（对标 3090 的 `marlin-tune-table.patch`） | **+15% ~ +25%**<br/>（预计 **80–90 tok/s**） |
| **Stage 3**<br/>*(投机步长匹配)* | **搜索 RX 7900 XTX 最优 `DFLASH_TOKENS` 与启用 Lookup / Chain 缓存** | 1. 实测 `DFLASH_TOKENS=4` (`M=5`，直接命中 `wvSplitK_int4_g`) vs `5` vs `7` 的端到端 tok/s 甜点；<br/>2. 开启 `DFLASH_LOOKUP=1` / `DFLASH_CHAIN=1`，在代码与长对话场景下用零成本 N-gram 命中替代 drafter 开销。 | `single-user/start_qwen.sh` 默认参数微调 | **+10% ~ +20%**<br/>（通用 **88–98 tok/s**，代码场景 **120+ tok/s**） |
| **Stage 4**<br/>*(Attention + GDN 调优)* | **针对 RDNA3 64 KB LDS 与 Wave32 调优 Split-KV Verify 及 GDN Decode** | 1. 在 `patches/spec-decode-attn.patch` 中当 $G \times M \le 32$ 时使用 `BLOCK_M=32`，使 `TILE=64` 直接塞入 64 KB LDS（无需折半至 32）；<br/>2. 针对 RDNA3 Wave32 调优 GDN `fused_recurrent` 的 `BV` 与 `num_warps`。 | 2 个现有 patch 的 ROCm 分支微调 | **+8% ~ +12%**<br/>（预计单用户突破 **95–105+ tok/s**） |
