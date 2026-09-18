# Nano-Omni Pivot-RL 实验与解码结果总表（交接用）

覆盖所有 nano-omni 相关实验（截至 2026-09-17）。通用信息：

- **基座模型**：`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`（HF snapshot 在
  `$HF_HOME/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/`，
  `$HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface`）。
- **训练代码**：github.com/yuekaizhang/RL 分支 `feat/nano-omni-tau1`;各实验 yaml 在
  `examples/nemo_gym/grpo_nano_omni_30ba3b_tau1*.yaml`（defaults 链，叶子只写差异）。
- **wandb**：project `nemo-rl-tau`,每臂多个 run（每 4h 窗口一个），**按下表 run name 过滤**即得整臂曲线。
- **think**：所有训练与解码默认开启（模板默认 `enable_thinking=True`，serve 带
  `--reasoning-parser nemotron_v3`），唯一例外是 nothink 消融臂。
- **HF ckpt** 统一在 `$EVAL/ckpts/`（`$EVAL=/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/tau2_eval`）。
- **评测协议**：tau2 文本 = luna 用户模拟器、每任务 4 trials 取均值、telecom 拆两半场合并、
  **空输出算 0 分计入分母**（表内以 Infra Error 形式统计）；tau-voice = 半双工自研装置
  （github.com/yuekaizhang/tau2-bench 分支 `voice-halfduplex`），**空输出不计分**（报覆盖率 n_scored/n_jobs）。
- **模板**：脏 = 上游模板（system 渲染成 Python list-repr,训练时的实际输入）；净 = sysfix 模板
  （`examples/nemo_gym/nano_omni_chat_template_sysfix.jinja`）。铁律：干净训干净测、脏训脏测。

## 表 1 · 实验清单（配置 / 数据 / 产物）

| # | 实验 | wandb run name | 关键配置（相对 plain GRPO think 基线） | 训练数据（来源） | Megatron ckpt | HF ckpt（$EVAL/ckpts/） | 状态 |
|---|---|---|---|---|---|---|---|
| 0 | base（未训练） | — | — | — | — | 直接用 HF snapshot | 仅评测 |
| 0b | SFT18k（外部,未 RL） | —（deshr 训练） | text SFT 中间产物 | 见 deshr | — | `/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_chat/users/deshr/models/kevin_nvc4_sft_text_por_ord_hf_ckpt_18000_mcore_to_hf` | 仅 voice 评测 |
| 1 | think 基线（plain GRPO） | `tau1-nano-omni-4n8g` | 基线：token 级 GRPO、clip 0.2/0.2、无 IS 修正、think on、脏模板 | HF 公开 96K（1.4M 原始池经 **Super3** 画像 0≤score≤0.6 过滤;repo `datasets/conversational_tool_use_pivot/`） | `results/tau1-nano-omni-4n8g/checkpoints` | `nano_omni_step{160,470,570,600}_hf` | 完成 600 步 |
| 2 | nothink 消融 | `tau1-nano-omni-nothink-4n8g` | 基线 + enable_thinking=false | 同上 | `results/tau1-nano-omni-nothink-4n8g/...` | `nano_nothink_step260_hf` | 早停（结果差） |
| 3 | GSPO+TIS | `tau1-nano-gspo-tis-4n8g` | + IS 修正 + TIS 截断 5.0 + seq 熔断 1.5 + clip 0.28 + **GSPO sequence 级 loss** | 同上 | `results/tau1-nano-gspo-tis-4n8g/...` | `nano_gspo_step{110,160,200,290,300,350,410,420,520,570,600}_hf` | 完成 600 步 |
| 4 | TIS-only | `tau1-nano-tis-4n8g` | 同 #3 但**回到 token 级 loss**（GSPO 消融） | 同上 | `results/tau1-nano-tis-4n8g/...` | `nano_tis_step{110..600}_hf`（12 个） | 完成 600 步；**文本冠军臂** |
| 5 | subset-A（mixed-only） | `tau1-nano-dadv-4n8g` | 基线配方,数据换 nano 自画像混合行 | nano 8-rollout 自画像 0<score≤0.6（剔全零),28,538 行 | `results/tau1-nano-dadv-4n8g/...` | `nano_dadv_step{160,210}_hf` | 231 步早停 |
| 6 | subset-B DAPO | `tau1-nano-dapo-dadv0-4n8g` | #3 栈 + DAPO 动态采样 ×2 | nano 自画像 ≤0.6 含全零（dadv0）,66,522 行 | `results/tau1-nano-dapo-dadv0-4n8g/...` | `nano_dapo_step{390,440}_hf` | 598 步；崩坏 |
| 7 | d60（论文数据） | `tau1-nano-d60-4n8g` | 基线配方,数据整包替换 | 论文 `train_difficulty_60.jsonl`：**Qwen3-think** 16-rollout 自画像 0≤score≤0.6,314,505 行（已剔 60 行 val 重叠） | `results/tau1-nano-d60-4n8g/...` | `nano_d60_step{200..785}_hf`（12 个） | 785 步停 |
| 8 | tis-dadv0 | `tau1-nano-tis-dadv0-4n8g` | #4 栈 + **净模板** + dadv0 数据 | dadv0（同 #6） | `results/tau1-nano-tis-dadv0-4n8g/...`（仅 step_82） | `nano_tisdadv0_step82_hf`（评测副本已打净模板） | 82 步早停;**未评测** |
| 9 | ctpl（对照 A） | `tau1-nano-tis-cleantpl-4n8g` | #4 栈 + **净模板**（单变量对照） | HF 公开 96K | `results/tau1-nano-tis-cleantpl-4n8g/...` | `nano_ctpl_step{160,290,410,420,470,560,570,600}_hf` | 完成 600 步 |
| 10 | rmin（对照 B） | `tau1-nano-tis-rmin-4n8g` | #9 + `ratio_min 0.1`（IS 权重钳 [0.1,5]） | HF 公开 96K | `results/tau1-nano-tis-rmin-4n8g/...`（至 step_491） | `nano_rmin_step{110,200,260,290,360,410,460}_hf` | ~491 步停（让位其他实验） |

另有废弃尝试：1n8g rmin 长跑（`results/tau1-tis-rmin-1n8g`，step_35，resume 显存悬崖放弃，见 agents/train-launch.md）。

## 表 2 · tau2-bench 文本解码（avg reward ×100;每任务 4 trials;空输出算 0 计入）

只列已解码 ckpt 的代表点；**全曲线数字见 exp2.md Table 1 / Part 2**。空输出列 = Infra Error
sims/总 sims。traces 已统一整理到 `$TR = /lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/traces_organized`
（telecom 两半场已合并;文件自带来源路径;原始目录未动）。规律：
`$TR/<实验>/step_<N>/tau2_text/<实验>_step<N>_tau2text_<域>.json`。

| 实验/ckpt | airline | retail | telecom (t1/t2) | 三域均值 | 文本空输出 | 模板 | traces（$TR/） |
|---|---|---|---|---|---|---|---|
| base | 58.50 | 64.25 | 44.96 | 55.90 | 0 全域 | 脏 | `base/step_0/tau2_text/` |
| think160 (#1) | 56.50 | 65.13 | 44.83 | 55.49 | 少 | 脏 | `think/step_160/tau2_text/` |
| **think470 (#1 最优)** | 61.31 | 60.09 | 50.44 | **57.28** | ~0（air 1/200） | 脏 | `think/step_470/tau2_text/`（含 `_fullrun` 整域 telecom） |
| think570 (#1) | 60.50 | 62.28 | 47.14 | 56.64 | 少 | 脏 | `think/step_570/tau2_text/` |
| think600 (#1) | 64.00 | 59.65 | 46.71* | 56.79 | 少 | 脏 | `think/step_600/tau2_text/`（telecom 取完整的 `_rerun_r2`;首轮 426/456 sims 存 `_fullrun`） |
| nothink260 (#2) | 37.00 | 25.44 | 18.9 (27.6/10.1) | 27.1 | — | 脏 | `nothink/step_260/tau2_text/` |
| gspo420 (#3 最优) | 60.80 | 62.72 | 46.08 | **56.53** | 少（t2 4/144） | 脏 | `gspo/step_420/tau2_text/` |
| tis410 (#4) | 63.49 | 71.71 | 41.53 | 58.91 | t1 7/163, t2 18/123 | 脏 | `tis/step_410/tau2_text/` |
| **tis600 (#4 最优,全场第一)** | **68.45** | **74.34** | **48.69** (72.64/24.74) | **63.83** | t1 16/193, t2 34/152 | 脏 | `tis/step_600/tau2_text/` |
| dadv160/210 (#5) | 51.02/57.84 | 62.94/68.42 | 41.23/27.90 | 51.73/51.39 | 少 | 脏 | `dadv/step_{160,210}/tau2_text/`（210 含 `_rerun_*` 文件） |
| dapo390/440 (#6) | 28.27/16.33 | 17.54/13.16 | 15.79/14.91 | 20.53/14.80 | 少（崩坏是重复调用非空输出） | 脏 | `dapo/step_{390,440}/tau2_text/` |
| d60_380 (#7 最优) | 56.20* | 72.61 | 46.18 | **58.33** | airline 49/200(25%)+重跑合并 | 脏 | `d60/step_380/tau2_text/`（含 `_rerun_a2`） |
| d60_600 (#7) | 45.70 | 67.00 | 40.00 | 50.90 | — | 脏 | `d60/step_600/tau2_text/` |
| ctpl160 (#9 最优) | 57.51 | 65.13 | 45.61 | **56.08** | air 7/200,tel 0 | **净** | `ctpl/step_160/tau2_text/` |
| ctpl600 (#9) | 54.27 | 64.91 | 37.15 | 52.11 | air 1/200 | **净** | `ctpl/step_600/tau2_text/` |
| rmin260 (#10 最优) | 59.90 | 61.84 | 46.21 | **55.98** | ~0 | **净** | `rmin/step_260/tau2_text/` |
| rmin460 (#10) | 59.28 | 62.28 | 41.69 | 54.42 | 少 | **净** | `rmin/step_460/tau2_text/` |

`*` 含 infra-error 任务定向重跑合并。tis-dadv0 (#8) 未解码。表 2 未列的其余全曲线 ckpt（gspo 110-600、tis 110-570、think 160/570/600、d60 200-785、
ctpl/rmin 各步、base_cleantpl 对照）traces 同规律,见 `$TR/README.md`。

## 表 3 · tau-voice 解码

**单元格格式**：`判负分（计分分, 计分数/总sims, 空N）`——括号外 = 空输出按 0 分计入全体
（可跨模型直接比较）;括号内 = 仅计分集的均分与覆盖。avg = 三域判负分均值。
**avg@1** = 每任务 1 trial（早期协议,airline 单 trial 噪声 ±7,见 eval_noise_and_paired_test.md）;
**avg@4** = 每任务 4 trials（2026-09-16 统一重测）。

### avg@4（五模型同协议,支持任意两两 per-case 配对）

| 模型 | airline | retail | telecom | avg | traces |
|---|---|---|---|---|---|
| SFT18k | 38.0（42.9, 177/200, 空23） | 28.5（29.5, 441/456, 空15） | 19.5（21.0, 423/456, 空33） | 28.7 | `$TR/sft18k/step_18000/tau_voice/` |
| base | 51.0（53.1, 192/200, 空8） | 41.2（41.3, 455/456, 空1） | 21.9（22.9, 436/456, 空20） | 38.1 | `$TR/base/step_0/tau_voice/` |
| **think470** | 58.5（59.1, 198/200, 空2） | 42.1（42.2, 455/456, 空1） | **27.6**（27.9, 452/456, 空4） | **42.7** | `$TR/think/step_470/tau_voice/` |
| gspo420 | 56.0（60.2, 186/200, 空14） | 38.4（39.2, 446/456, 空10） | 18.9（20.0, 430/456, 空26） | 37.7 | `$TR/gspo/step_420/tau_voice/` |
| tis600 | 48.0（**63.2**, 152/200, 空48） | 41.9（**43.8**, 436/456, 空20） | 14.7（24.0, 279/456, 空177） | 34.9 | `$TR/tis/step_600/tau_voice/` |

voice traces 同样整理进 `$TR`（分片已合并为单文件,`*_avg4.json` 内嵌全部轨迹与
per-(task,trial) rewards;空输出无轨迹,只在 results 里以 reward=null 出现）。

判负口径下 **voice 排名：think470 > base > gspo420 > tis600**——tis600 空输出把它拖到基座
以下;计分集口径它 airline/retail 仍最高。配对分析（共同计分格）：think470→tis600
airline −2.0 / telecom −6.5,voice 实际冠军为 think470。失败模式 A-E 分类统计见 exp3.md。

### avg@1（历史协议;部分 airline 格实为 4 trials,已注明）

| 模型 | airline | retail | telecom | avg | traces（$TR/） |
|---|---|---|---|---|---|
| base | 48.0（50.3, 191/200⁴ᵗ, 空9） | 39.5（39.8, 113/114, 空1） | 24.6（25.9, 108/114, 空6） | 37.3 | `base/step_0/tau_voice/`（airline avg4_legacy / retail avg1_r3 / telecom avg1_r2） |
| think470 | 56.0（56.6, 198/200⁴ᵗ, 空2） | 44.7（45.1, 113/114, 空1） | 24.6（24.6, 114/114, 空0） | 41.8 | `think/step_470/tau_voice/`（airline avg4_legacy） |
| gspo420 | 50.0（54.3, 46/50, 空4） | 28.9（44.6, 74/114, 未计40*） | 18.4（20.2, 104/114, 空10） | 32.5* | `gspo/step_420/tau_voice/`（retail 两轮均为 judge-bug 时代） |
| tis410 | 50.0（59.5, 42/50, 空8） | 46.5（46.9, 113/114, 空1） | 18.4（23.1, 91/114, 空23） | 38.3 | `tis/step_410/tau_voice/`（retail 取 judgefix 轮 / telecom 取 r2） |
| tis600 | 48.0（72.7, 33/50, 空17） | 44.7（46.4, 110/114, 空4） | 17.5（25.3, 79/114, 空35） | 36.8 | `tis/step_600/tau_voice/`（`*_avg1.json`） |
| dapo390 | 18.0（19.1, 47/50, 空3） | — | — | — | `dapo/step_390/tau_voice/` |
| dapo440 | 12.0（12.0, 50/50, 空0） | — | — | — | `dapo/step_440/tau_voice/` |
| base（non-think 模式） | 8.0（8.0, 50/50, 空0） | — | — | — | `base_nothink_mode/step_0/tau_voice/` |

traces 列为 `$TR` 下路径（原始目录在 `$EVAL/voice_results`,文件内 source_dirs 可溯源）。
`⁴ᵗ` 该格实为 4 trials（当年为压 airline 噪声补测）。
`*` gspo420 retail 的 40 个未计分是 judge 路由 bug（评测装置问题）而非模型空输出,其判负分
28.9 与 avg 32.5 因此系统性偏低,不能与其他行直接比。
tis410 telecom 取第二轮（r2）;第一轮在 `voice_nano_tis_step410_telecom`。

## 附 · 关键资产与文档索引

| 资产 | 位置 |
|---|---|
| 文本评测装置 | `$EVAL/`（tau2-bench checkout + venv + run_tau2_baseline.sh）;提交脚本 RL 仓库 `yuekai_scripts/submit_tau2_paper_luna_eval.sh` |
| voice 评测装置（可对外交接） | github.com/yuekaizhang/tau2-bench 分支 `voice-halfduplex`（含英文 README）;4t 分片作业脚本 `yuekai_scripts/run_voice_4t_job.sh` |
| 干净模板 | RL 仓库 `examples/nemo_gym/nano_omni_chat_template_sysfix.jinja`;评测副本打补丁工具 `yuekai_scripts/patch_nano_chat_template.py` |
| 自画像重打分工具 | `yuekai_scripts/rescore_pivot_with_omni.py`（用法见 agents/self-profile-rescore.md） |
| 实验记录 | 本目录 `exp.md / exp2.md / exp3.md`;新集群交接包 `agents/`;统计方法 `eval_noise_and_paired_test.md` |
| 失败模式分析参考 | `pivot_rl/tau-voice-plan-2026-09.pdf`（五模式 A-E 定义） |
| 容器 | 训练 `nemo_rl.0821.sqsh`,评测 `nemo_rl.0803.sqsh`（`/lustre/fsw/.../yuekaiz/containers/`） |

已知缺口：#8 tis-dadv0 未解码;#10 rmin 高 step（360/410）telecom t2 因超时缺失;
gspo 290/410/520、d60 710/770/780-telecom 按指示取消未测;voice avg@4 未覆盖 tis410。
