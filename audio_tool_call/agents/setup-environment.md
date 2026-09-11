# 环境与资产：位置、容器、迁移清单

## 代码

- **主仓**：`github.com/yuekaizhang/RL`，分支 **`feat/nano-omni-tau1`**
  （NVIDIA-NeMo/RL 的 fork）。原集群工作树：
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/pivot_rl/RL`。
- 关键子目录：
  - `examples/nemo_gym/grpo_nano_omni_30ba3b_tau1*.yaml` — 全部实验配置
    （defaults 链继承，叶子 yaml 只写差异；读一个臂先顺着 defaults 链读到底）。
  - `examples/nemo_gym/nano_omni_chat_template_sysfix.jinja` — 修好 system
    渲染的 chat template（repo 托管，训练经 config 键加载，见 train-launch.md）。
  - `yuekai_scripts/` — 所有 launcher / 工具脚本。
  - `3rdparty/Gym-workspace/Gym`、`3rdparty/Megatron-Bridge-workspace/` —
    pinned 子模块，PYTHONPATH 前置压过容器内 /opt 版本（无需重 build 镜像）。
- 未提交的本地改动（有意不 commit）：`pyproject.toml` 等 3 处
  requires-python 放宽 patch——只为让 `uv run --no-sync` 在容器里通过版本检查，
  新集群按报错情况现做即可。

## 数据

原集群根：`/lustre/fsw/portfolios/coreai/users/yuekaiz/others/data/`（迁移需拷走）+
仓库内 `datasets/conversational_tool_use_pivot/`（prepare 脚本产物）。

| 数据 | 说明 |
|---|---|
| `datasets/conversational_tool_use_pivot/train.jsonl` | 公开 96K pivot 训练集（HF 全量，不过滤），由 `examples/nemo_gym/prepare_conversational_tool_use_pivot_data.py` 生成 |
| `datasets/conversational_tool_use_pivot/val.jsonl` | **标准 512 行 val，铁律：永不改动**，所有臂共用 |
| dadv0 训练集（66,522 行） | nano 自画像 ≤0.6 含全零行，见 self-profile-rescore.md |
| `train_difficulty_60.jsonl`（314,565 行） | 论文原始数据，qwen30b 自画像；行内 `qwen_235b_info` 字段是旧命名，实为 30B 自画像。使用前要改 agent_ref 服务名 + 剔除与 val 重叠 60 行 |

## 模型 / checkpoint

| 资产 | 位置（原集群） |
|---|---|
| 基座 HF snapshot | `$HF_HOME`（`/lustre/fsw/.../users/yuekaiz/.cache/huggingface`）内 Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16；训练离线加载（HF_HUB_OFFLINE=1） |
| 最优 ckpt tis_410（HF 格式） | `tau2_eval/ckpts/nano_tis_step410_hf` |
| 各臂训练 ckpt | `RL/results/<job-name>/checkpoints/`；注意 keep_top_k 会删旧 step，要保的用 `cp -al` 硬链接备份；async save 可能留下缺 `config.yaml` 的残缺 ckpt，选点前先验 |
| 论文原始 ckpt tau_rl | `/lustre/fsw/portfolios/llmservice/users/jkyi/checkpoints/pivot_rl/tau_rl` |

## 容器

- **训练**：`nemo_rl.0821.sqsh`（vLLM ≥0.25.x；0724 镜像会 ImportError）。
- **文本评测**：`nemo_rl.0803.sqsh`（评测脚本依赖其 /opt/ray_venvs 布局）。
- 镜像目录：`/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/`。
- **更新镜像的兼容性坑**（2026-09-11 实测）：更新的镜像带 transformers
  ≥5.12，会触发 `nemo_rl/models/policy/__init__.py` 里 deepseek_v3
  tokenizer 补丁的版本断言——分支上已改为 ≥5.12 直接跳过补丁（上游已修）。
- 关键机制：**PYTHONPATH 仓库前置**（launcher 的 COMMAND 里 export），让工作树
  的 nemo_rl / Gym / Megatron-Bridge 压过镜像内置版本——改代码不用重打镜像。

## API / 密钥

- `/lustre/fsw/portfolios/coreai/users/yuekaiz/.bashrc_api` — source 后得到
  `OPENAI_BASE_URL` / `OPENAI_API_KEY`（NVIDIA 内部网关，路由 gpt-5.6-luna）。
  **不要把内容打印到日志/对话里**。
- wandb：project `nemo-rl-tau`；mlflow 同名 experiment。

## Slurm 使用约定（原集群，新集群按其政策适配）

- 登录节点（`$USER=yuekaiz`）**不跑计算、不装包**，只编辑 + sbatch；
  容器内（`$USER=root`）可直接跑。
- account `coreai_dlalgo_nemorl`，partition `batch`，单作业上限 4h →
  长训练靠 `--dependency=singleton` 链式续跑（launcher 已内置）。
- 提交后看状态用 `sacct -j <id>`（squeue 偶发 socket 报错造成误判）。
- git commit 用 `git commit -s --no-verify`（pre-commit 的 uv sync 会写满
  home cache），提交前手动 `uv run --no-sync ruff check` + `ruff format`。

## 新集群落地清单

1. clone fork 分支；按新集群路径改 `yuekai_scripts/*.sh` 顶部的
   CONTAINER / HF_HOME / EVAL / 数据路径（都做成了环境变量可覆盖）。
2. 拷贝或重新生成数据（prepare 脚本 + rescore 产物）、HF snapshot、
   tau2_eval 整目录（含 venv 需重建，见 tau-voice-eval.md）。
3. **改 yaml 里的绝对路径**：`grpo_nano_omni_30ba3b_tau1_tis_dadv0.yaml`
   的 `chat_template:` 指向 sysfix.jinja 的绝对路径，必须改成新集群路径。
4. 先跑 1n8g 冒烟（train-launch.md）验证栈通，再上多节点。
5. 确认新集群到 NVIDIA 网关的连通性（luna user-sim 与 judge 都依赖它）。
