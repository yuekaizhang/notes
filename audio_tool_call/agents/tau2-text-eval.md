# tau2-bench 文本评测：标准协议与提交

## 标准协议（所有历史数字都按此，不许改）

- 用户模拟器：**gpt-5.6-luna**（NVIDIA 网关，temperature 0.0），
  报告数字 = average reward × 100，**每任务 4 trials**。
- 三域：airline(50 任务) / retail(114) / telecom(114)。
- telecom 慢，必须**单独提交**且常拆两个 57 任务半场（`TASK_IDS_FILE`），
  合并后再算域分；三域均值 = airline/retail/telecom 等权平均。
- 评测目录：`/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/tau2_eval/`
  （tau2-bench checkout + venv + 结果 + 日志）。

## 提交（登录节点）

launcher：`yuekai_scripts/submit_tau2_paper_luna_eval.sh`（1n8g × ray.sub，
agent vLLM 占 GPU 0-3，其余闲置——sbatch comment 里已带 idle-GPU 豁免）。

```bash
# 默认两连发：(airline+telecom) 和 (retail)
TAG=nano_tis_410 CKPT=/path/to/hf_ckpt \
TOOL_PARSER=qwen3_coder REASONING_PARSER=nemotron_v3 \
bash yuekai_scripts/submit_tau2_paper_luna_eval.sh

# 单域 / telecom 半场
DOMAINS="telecom" TASK_IDS_FILE=$EVAL/telecom_t1_ids.txt JOB_SUFFIX=_t1 ... 同上
```

要点：
- **nano 系 parser**：`TOOL_PARSER=qwen3_coder REASONING_PARSER=nemotron_v3`
  （脚本默认是 qwen 系的 hermes/qwen3，nano 必须显式传）。
- nano serve 需 `--mamba-ssm-cache-dtype float32`（脚本内已带则不用管；
  自己起 serve 时别漏）。
- 干净模板训的 ckpt：`EXTRA_VLLM_FLAGS='--chat-template <sysfix.jinja>'`
  或先 patch 评测目录（铁律"干净训干净测"）。
- 时间预算（luna 实测）：airline ~1h、retail ~1.5h、telecom 半场各 ~2h。
- **必须走 ray.sub**（其 srun 带 `--no-container-mount-home`，评测 venv
  依赖镜像内 /root/.cache/uv）；容器用 `nemo_rl.0803.sqsh`。

## 结果解读

- 结果目录 `tau2_eval/` 下按 TAG 命名的 log / json;infra-error（网关 5xx、
  超时）任务要**定向重跑合并**，不算模型分。
- 选点流程：wandb val 粗筛 2-4 个候选 step → **airline 快筛**（最便宜）→
  胜者跑全三域 → 必要时 voice。
- 对照基线（同协议）：nano 基座 55.9 / think470 57.3 / GSPO420 56.5 /
  tis_410 **58.9** / 2507-RL 58.7。新臂先和这排数字比。

## Checkpoint → HF 转换

Megatron ckpt 需先转 HF 再评测（转换脚本在仓库 examples/ 下，原集群惯例
是提 1 节点转换作业）。转换前验 ckpt 完整性：目录里必须有 `config.yaml`
（async save 中断会留残缺目录）；被 keep_top_k 删掉的 step 只能从 `_keep`
硬链接副本转。
