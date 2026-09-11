# 已验证结论速查（截至 2026-09-11）

完整实验记录在 `../exp1.md ../exp2.md ../exp3.md`，这里只留决策要用的。

## 最优配方

**token 级 GRPO + TIS 防漂栈**（`..._tau1_tis.yaml`）：
`use_importance_sampling_correction: true` + TIS 截断 5.0 +
`seq_logprob_error_threshold: 1.5` + clip 0.28 + overlong filtering。
最优 ckpt **tis_410**：tau2 三域均值 58.9（基座 55.9），voice 三域全胜
think470。防漂归因于 TIS/阈值组件而非 GSPO（消融闭环）。

## 两大 nano 顽疾

### 1. 脏 system 渲染（已修）

vLLM 多模态路径把 string content 规范成 parts list，nano 模板 system 分支
对 list 做 `| string` → 每条训练/评测 prompt 的 system 是 Python list-repr。
修法：`nano_omni_chat_template_sysfix.jinja` + config 键
`http_server_serving_chat_kwargs.chat_template`（只影响以后训的模型，
不碰模型目录）。**对照实验结论：脏渲染不直接伤分**（paired t=0.61），
修它是卫生（消除空输出 dump 里的 echo attractor 格式），不是涨分手段。
配套铁律：**干净训干净测、脏训脏测**。

### 2. TIS 臂随机空输出 / 重复循环（B 臂验证中）

- 症状：voice/文本解码随机发作，在结构 token 上循环（臆造模板标记、
  `<parameter>` 标签、政策句）烧满预算或复读后早停；剂量效应：
  基座 2-6% → plain GRPO 0-2% → TIS 臂 16-29%。加 token 预算无用
  （正常输出 p99 才 5.5K）。
- 假说:TIS 权重把脆弱结构 token 的梯度整段置零，连带静音了
  "学会不重复"的信号。
- 修法候选：`loss_fn.truncated_importance_sampling_ratio_min: 0.1`
  （IS 权重钳到 [0.1, 5.0]，保 10% 梯度流；None→0 是旧行为，实现在
  `nemo_rl/algorithms/loss_functions.py`）。**B 臂（tis-rmin）在验**，
  成功判据：token_mult_prob_error 保持平稳 且 voice 空输出率回到 ≤6%。
- 临时缓解：serving 加 repetition_penalty + 空输出重试。

## 数据结论

- **d60（论文数据整包替换）= 偏科交易**：retail 冲 73.7 史高、airline
  崩到 45.7 史低，三域均值从未超基座。方向应是"TIS 配方 × retail/telecom
  子集掺混"，不是整包。
- **难度过滤必须自画像**（跨模型不迁移，两次实证）。
- **DAPO（动态采样）在 dadv0 上有害**：浓缩弱信号 lottery 组；
  dadv0 的开局 val 下探是数据性质，与算法无关。

## 方法论铁律

1. **内部 val 与真实评测可背离**（两次实证：d60 val 新高 = tau2 新低）。
   选点必须 airline 快筛 → 全三域。
2. val 集 512 行永不改动，所有臂共用。
3. 对照实验做 per-task paired 分析（不是只看均值）。
4. infra-error 任务定向重跑合并后再报数；voice 报"计分均值 + 覆盖率"两列。
5. 报告数字前先确认协议一致（luna、4 trials、telecom 半场合并）。

## 未决 / 在跑（迁移时检查状态）

- A 臂 tis-cleantpl（全量数据 × 干净模板，对照 tis_410 隔离模板对训练的
  影响）与 B 臂 tis-rmin：各 7 窗 → 600 步。跑完后 airline 快筛
  →胜者全三域→B 臂加测 voice 空输出率。
- tisdadv0 step_82 的 airline 解码（TAG nano_tisdadv0_82）。
- telecom think470 的 t1 半场 79.8 疑似正向离群，待复核。
