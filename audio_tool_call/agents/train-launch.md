# 训练：提交 GRPO 训练与冒烟

## 标准 4n8g 训练（登录节点提交）

launcher：`yuekai_scripts/submit_tau1_4n8g.sh`（sbatch 包 `ray.sub`）。
所有可变项都是环境变量：

```bash
CONFIG_PATH=examples/nemo_gym/grpo_nano_omni_30ba3b_tau1_tis_rmin.yaml \
JOB_NAME=nano-omni-30b-tau1-tis-rmin \
EXTRA_OVERRIDES_STR="grpo.max_num_steps=600" \
bash yuekai_scripts/submit_tau1_4n8g.sh
```

- 单窗 4h × `--dependency=singleton` 链：**同一个 JOB_NAME 连提 N 次**即得
  N 个 4h 窗口自动接力（ckpt 自动 resume）。600 步 nano 约需 7 窗。
- dadv0 / d60 等换数据臂：数据路径经 `EXTRA_OVERRIDES_STR` 里的
  `data.train.data_path=...` 传（val 永远用标准 512 行）。
- 产出：`results/<JOB_NAME>/`（slurm 日志、logs/、checkpoints/）。
- ckpt 语义:`checkpointing.keep_top_k`（按 val 分留 top-k，会删档），
  `save_period` 与 `grpo.val_period` 通常同设 10。要长期保留的 step 立刻
  `cp -al` 硬链接出去。

## 实验 yaml 组织

defaults 单链继承，叶子只写 delta。当前族谱（每个文件头部注释写了设计意图）：

```
grpo_nano_omni_30ba3b_tau1_smoke_1n8g.yaml     # 1n8g 冒烟基座
└─ grpo_nano_omni_30ba3b_tau1.yaml             # 4n8g 正式（32K seq, 64x16）
   └─ ..._tau1_gspo_tis.yaml                   # GSPO+TIS
      └─ ..._tau1_tis.yaml                     # token 级 GRPO + TIS（最优配方）
         └─ ..._tau1_tis_dadv0.yaml            # + 干净模板 + keep_top_k 7（dadv0 数据经 overrides）
            └─ ..._tau1_tis_cleantpl.yaml      # A 臂：全量数据 × 干净模板
               └─ ..._tau1_tis_rmin.yaml       # B 臂：A + ratio_min 0.1
```

TIS 配方关键键（在 `_tis.yaml`）：`token_level_loss: true`、
`use_importance_sampling_correction: true`、TIS 截断 5.0、
`seq_logprob_error_threshold: 1.5`、clip 0.28。
B 臂新键：`loss_fn.truncated_importance_sampling_ratio_min: 0.1`
（IS 权重下限，None→0；语义见 findings-and-recipes.md）。

## 干净 chat template 的接线（重要）

**不改模型目录、不打镜像**——训练/rollout 用 repo 托管模板，经 config 键：

```yaml
policy:
  generation:
    vllm_cfg:
      http_server_serving_chat_kwargs:
        chat_template: <绝对路径>/examples/nemo_gym/nano_omni_chat_template_sysfix.jinja
```

（在 `_tis_dadv0.yaml` 里，下游臂全部继承；**迁移后必须改这个绝对路径**。）
nemo_rl 的 http server 会用 vLLM `load_chat_template` 加载文件——不要走
per-request chat_template（需要 `--trust-request-chat-template`，server 会 400）。

评测侧对应：干净模板训出的 ckpt，解码时也要挂同一个模板
（vllm serve `--chat-template ...` 或对**转换后的评测目录**跑
`yuekai_scripts/patch_nano_chat_template.py`——该脚本只许碰评测副本，
不许碰共享 HF snapshot）。

## 1n8g 冒烟（容器内直接跑）

进入交互 8 卡容器（`$USER=root`）后：

```bash
RESULTS_DIR=$PWD/results/smoke-xxx \
CONFIG_PATH=examples/nemo_gym/grpo_nano_omni_30ba3b_tau1_tis_rmin.yaml \
bash yuekai_scripts/run_nano_omni_smoke_1n8g.sh \
  cluster.num_nodes=1 grpo.max_num_steps=2 \
  grpo.num_prompts_per_step=8 grpo.num_generations_per_prompt=8 \
  policy.train_global_batch_size=64 policy.max_total_sequence_length=16384 \
  checkpointing.enabled=false grpo.val_period=1000000
```

任意 4n8g 臂 yaml 都能这样降配冒烟（正式臂默认 32K seq / 1024 样本每步，
单节点必须压小）。冒烟通过标准：完成 2 个 step、loss/token_mult_prob_error
正常打印、无 OOM。

## 1n8g 长跑（部分验证后放弃，2026-09-11）

stock 配置在单节点**新开跑没问题**（唯一必须覆盖 `cluster.num_nodes=1`；
TP8/EP8 单节点自包含，实测 ~387s/步，只比 4n8g 慢 1.8 倍），第一窗训到
step_35 全程健康。**但断点续跑必挂**：DP=1 时 Adam 状态不能沿 DP 分片，
每卡"权重+fp32主参数+Adam" ~71GB 贴顶，resume 加载 checkpoint 的临时
拷贝（~2GB）叠上 vLLM sleep 免不掉的残留（2 引擎 CUDA context+graphs
~5GB）就 OOM——新开跑没有这步所以能过。已排除的方案：vLLM 显存比
0.5→0.35（残留不随它变）、expandable_segments（缺口>碎片）、部分优化器
CPU offload（框架只支持 fraction=1.0）、TP1+EP8（加载能过但关 SP 后
训练步 backward OOM）。**理论可行方案（未走完验证）**：生成侧改
vLLM TP8 单引擎 + enforce_eager——这正是上游官方 1n8g 配方
`vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8.v1.yaml`
的配置，代价是生成段慢 ~30-40%。最终决定回 4n8g。提交方式（存档）：

```bash
CONFIG_PATH=... NUM_NODES=1 JOB_NAME=<独立名> SBATCH_PARTITION=interactive \
EXTRA_OVERRIDES_STR="grpo.max_num_steps=600 logger.wandb.name=<独立名>" \
bash yuekai_scripts/submit_tau1_4n8g.sh   # 连提 N 次成链
```

- interactive 分区同样 4h 上限，singleton 链照用；JOB_NAME/wandb 名必须
  与 4n8g 臂区分开（singleton 按名字去重，撞名会互相串行）。
- vLLM 可降 TP2（4 引擎提吞吐），但必须配
  `++policy.generation.vllm_kwargs.max_num_batched_tokens=4096`——nano 的
  mamba 缓存页 TP2 时 2096 > vLLM 默认调度上限 2048（TP4 无此问题）。

**必须走 sbatch ray.sub 提交训练，不要用 slurm-broker MCP 直接起容器**：
broker 容器会把用户 home 挂到 /root 上，遮住镜像的 /root/.cache/uv——
而 0821 镜像 /opt/nemo_rl_venv 的包文件全是软链进那个 cache 的
（uv symlink link-mode），结果 venv"目录在、文件全悬空"（click 变
namespace 包等诡异报错）。ray.sub 的 `--no-container-mount-home` 正是
为保住镜像 /root 而存在（eval 脚本注释"do NOT hand-roll sbatch"同源）。

## 训练健康指标（wandb）

- `token_mult_prob_error`：防漂核心指标，TIS 配方下应全程 ~1.07 平稳；
  持续走高 = 训推漂移，先查模板/渲染一致性。
- val（内部 512 行）：**与真实 benchmark 可背离**（已两次实证），只用于
  粗筛趋势，选点必须过 tau2 快筛（见 tau2-text-eval.md）。
- dadv0 类难数据臂开局 val 下探（0.34→0.30@80）是数据性质，不是坏——
  别急着杀（tis-dadv0 臂 82 步早停即此教训）。

## Babysitter（自动续链/看护）

原集群用后台脚本轮询 `sacct` 状态、失败重提。要点：
- 用 `sacct -j <id> --format=State` 判状态，**不要用 squeue**（socket 偶发
  失败会误判成"作业消失"）。
- 链式续跑本身靠 singleton 依赖，babysitter 只兜"整链死掉"的底。
