# 自画像难度重打分（rescore）与难度过滤数据

## 为什么必须自画像

**跨模型难度不迁移**（实证两次）：qwen30b 自画像的 d60 数据给 nano 训练时
nano 实际 train reward ~0.43（设计均值 0.163）；反向看 dadv0 这批 nano
"难题"在 qwen 画像下均值 0.332、8.5% 的行 qwen 得分 >0.6。
→ 给谁训练就用谁打分，别复用别的模型的难度标签。分布对比详见
`../exp3.md` 附录。

## 工具

`yuekai_scripts/rescore_pivot_with_omni.py`：对 pivot JSONL 每行，
用目标模型采 K 个 rollout（按训练温度、训练同款 chat template 渲染），
用 Gym verifier（`single_step_tool_use_with_argument_comparison` 的
faithful 本地拷贝）逐个判分，写回每行的 reward 统计。

```bash
# 容器内、8 卡节点；两个 4-GPU shard 并行是标准姿势
python yuekai_scripts/rescore_pivot_with_omni.py \
  --model <hf_model_dir> \
  --in-jsonl train.jsonl --out rescored.k0.jsonl \
  --recipe-yaml examples/nemo_gym/grpo_nano_omni_30ba3b_tau1.yaml \
  --samples 8 --temperature 1.0 --tp 4 \
  --mamba-f32-cache --shard 0/2     # 另一进程 --shard 1/2, CUDA_VISIBLE_DEVICES=4-7
```

要点：
- `--recipe-yaml` 模式让 prompt 渲染与训练完全一致（含干净模板臂——
  换了模板后重画像也要挂同款模板）；
- nano 记得 `--mamba-f32-cache`；`--chunk` 控制 generate/flush 批量，
  断点续跑靠输出文件已有行数（重跑前看脚本 resume 逻辑）；
- 96K 行 × 8 samples 在 2×4GPU shard 上约需数个 4h 窗，按 shard/续跑拆。

## verifier 判分语义（构数据前必须懂）

二值 reward：pivot 处该发消息的，模型输出任意文本即 1.0；该调工具的，
工具名必须对 + 参数逐 key 比较（长字符串 word-bag 模糊匹配阈值 0.1，
key/短字符串精确匹配）。所以"难"集中在工具选择与参数正确性。

## 难度过滤集构造

对 rescore 产物按 `reward_mean` 过滤：

- **dadv0**（当前在用）：nano 自画像 8 rollouts，保留 `reward_mean <= 0.6`
  **含全零行**，96K 公开集 → 66,522 行（57.1% 全零，均值 0.116）。
- 过滤后务必：① 按内容哈希剔除与标准 512 行 val 的重叠；② 不改 val。
- 注意 8 rollouts 只有 5 个难度档位，全零档混着真实成功率 5-12% 的题；
  要更纯的 0 档用 16 rollouts（对齐论文 d60）。
- 全零行占比高的数据在普通采样下近半配额零梯度空转（advantage 全 0），
  开局 val 会下探——要么接受（等模型长进解锁），要么配 DAPO 动态采样
  （但 DAPO 臂已实证会浓缩弱信号 lottery 组，慎用）。
