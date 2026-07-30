# Step-Audio2 DiT Token2Wav 加速：FlashInfer vs TensorRT

## 背景

为优化 MiniCPM-o 的 token2wav 组件做技术选型。实验对象是与其同构的
Step-Audio2-mini DiT flow-matching token2wav（CosyVoice2 DiT 架构）：

- **模型**：16 层 DiT block（hidden 512，8 头 x 64 head_dim attention +
  causal conv block + MLP x4），输入 320ch（noisy mel / mu / spk / cond 拼接），
  输出 80 维 mel；
- **推理形态**：每条语音 10 步 ODE（flow matching），CFG 使有效 batch=2；
  序列长度 = prompt mel + 生成 mel，逐样本变化（实验数据 392–1028 帧）；
- **精度**：fp16（TensorRT 为 FP16 builder flag 的混合精度，FlashInfer 路径为
  fp16 存储 + fp32 累加），单张 H100。

对比了两种加速方案：

1. **TensorRT**：fp32 ONNX 以 FP16 flag 构建 engine，动态 shape 由
   optimization profile 原生支持；
2. **FlashInfer**：与原模型 state_dict 兼容的 estimator 重写——ragged
   attention、kernel 融合、可选 CUDA graph——作为 `nn.Module` 无侵入替换，
   权重零改动。

**基准**：26 条 wenetspeech4tts 样本（约 170s 音频），batch=1，仅统计
estimator 时间（cuda-sync 精确计时，26 x 10 = 260 次调用）。每一版改动都用
paraformer ASR 逐条转录校验，全程 26/26 无损。

## Profile：钱花在哪

朴素 torch eager 每次调用墙钟 ~12ms，但 GPU 实际 kernel 时间只有 ~6.3ms
（约 1500 个 kernel，CPU 提交速度跟不上，GPU 空转近半——典型的小模型
launch-bound 形态）。kernel 时间本身的构成（T=500）：

| 类别 | 时间 | 占比 | 根源 |
|---|---|---|---|
| CausalConv1d | 1.36 ms | 35% | cudnn implicit-gemm 附带的 NCHW<->NHWC 布局转换 kernel |
| elementwise 链 | 1.19 ms | 30% | adaLN modulate / gate / residual 未融合，每 block 12+ 次全张量读写 |
| LayerNorm x97 | 0.57 ms | 15% | 3 个 adaLN 前置 norm + qk norm + conv 内 norm |
| GEMM | 0.60 ms | 15% | qkv / proj / MLP / adaLN |
| attention 本体 | 0.18 ms | 4.6% | 从头到尾都不是瓶颈 |

两个直接推论：只换 attention kernel 收益有限（实测仅 1.26x）；要接近
TensorRT 必须同时解决 **kernel 时间**（融合）和 **launch 空转**（CUDA graph）。

## FlashInfer 路径的优化项

数值等价的改造（均通过 ASR 校验）：

- **causal conv 改单 GEMM**：k=3 因果卷积等价于
  `cat([x_{t-2}, x_{t-1}, x_t]) @ W_flat`，直接在 (B,T,C) 布局上做 linear，
  cudnn 布局转换全部消失（-1.2ms 量级，最大单项）；
- **modulate / residual 单 kernel 化**：把 `(1+scale)` 的 +1 折进 adaLN
  Linear 的 bias，`LN(x)*(1+s)+shift` 与 `x+gate*y` 各变为一次 `addcmul`；
- **qkv 三合一 GEMM**；
- **ragged attention**：`BatchPrefillWithRaggedKVCacheWrapper`，plan 按
  shape 缓存（10 步 x 16 层共享一次 plan）；
- **CUDA graph**：10 步 ODE 形状恒定，一张图回放 10 次。
  - *per-shape*：每种序列长度一张图（数值与 eager 完全一致）；
  - *分桶*：pad 到桶长；因 FlashInfer plan 会把序列长烘进图里，桶内
    attention 改为 SDPA + 运行时更新的 key-padding mask，保证 pad 不污染
    softmax，结果精确。桶化的代价实测严格等于平均 pad 比例。

### 失败的尝试

- **torch.compile**（per-block 与整图、dynamic=True 均试过）：比 eager 更慢
  ——16 处自定义 kernel 调用引发 graph break，叠加动态 shape guard 开销；
- **fp8 GEMM（fp8_blockscale_gemm_sm90）**：反而慢 2x——GEMM 太瘦
  （M~650、K=512/2048），online activation 量化 kernel 的开销超过 GEMM 节省；
  且有损（单步最大相对误差 ~20%，ASR 出现 WARN，输出 RMS 系统性漂移）。

## 最终对比

estimator 总时间（260 次调用）与单次调用耗时：

| 方案 | 总时间 | ms/call | vs torch |
|---|---|---|---|
| torch eager | 2.57 s | 9.9 | 1x |
| FlashInfer eager（仅换 attention） | 2.05 s | 7.9 | 1.26x |
| FlashInfer 融合版 eager | 1.52 s | 5.83 | 1.69x |
| FlashInfer 融合版 + per-shape CUDA graph | **0.76 s** | **2.93** | **3.38x** |
| FlashInfer 融合版 + 2s 步长分桶 graph（8 张图） | 0.95 s | 3.66 | 2.70x |
| FlashInfer 融合版 + 粗桶 5/10/20s graph（2 张图） | 1.14 s | 4.38 | 2.26x |
| **TensorRT fp16** | **0.78 s** | **3.01** | **3.28x** |

## 结论

1. **per-shape CUDA graph 下 FlashInfer 可以略微反超 TensorRT**
   （2.93 vs 3.01 ms/call），证明"flashinfer kernel + 手工融合 + graph"
   能在 PyTorch 生态内拿到 TRT 级性能，且代码保持可改、权重零转换；
2. **但 per-shape graph 不具备生产可行性**：线上序列长度任意变化，意味着
   每个新长度捕一张图（显存与捕图开销无界）。可部署的是分桶 graph，而
   桶化必然引入 pad 白算 + 桶内 attention 降级，2s 步长桶 3.66 ms、
   粗桶 4.38 ms，均慢于 TensorRT；
3. **因此在无法为 per-shape 设置 CUDA graph 的前提下，TensorRT 更快**
   （3.01 vs 3.66+）。其根本优势在于 optimization profile 原生支持动态
   shape——不 pad、不分桶、无图管理成本，同时编译期完成了与我们手工融合
   等效的 kernel 融合。对 MiniCPM-o token2wav 这类序列长度连续变化的
   组件，TensorRT 是更稳妥的选择；FlashInfer 路径的价值在于长度分布
   可控（可粗粒度分桶）或需要快速迭代模型结构的场景。
