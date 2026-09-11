# agents/ — 新集群 Claude 交接包

本目录是为迁移到新集群后接手工作的 Claude（或人类）准备的操作手册。
项目：PivotRL（arXiv:2603.21383）对话工具调用 pivot RL 复现与改进，
基座 Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16，框架 NeMo-RL + NeMo-Gym。

实验结论与数据分析在同目录的 `../exp1.md` `../exp2.md` `../exp3.md`
（exp3 是最新），本目录只讲**怎么干活**。

## 文件索引

| 文件 | 内容 |
|---|---|
| [setup-environment.md](setup-environment.md) | 代码/数据/模型资产位置、容器、环境变量、新集群落地清单 |
| [train-launch.md](train-launch.md) | 怎么提交 GRPO 训练（4n8g 链式 sbatch）、1n8g 冒烟、babysitter |
| [tau2-text-eval.md](tau2-text-eval.md) | tau2-bench 文本解码评测的标准协议与提交方法 |
| [tau-voice-eval.md](tau-voice-eval.md) | 自建 tau-voice 半双工语音评测装置的启动与坑表 |
| [self-profile-rescore.md](self-profile-rescore.md) | 自画像难度重打分（rescore）与难度过滤数据构造 |
| [findings-and-recipes.md](findings-and-recipes.md) | 已验证结论速查：最优配方、两大 nano 顽疾及修法、铁律 |

## 30 秒速览（先读这个）

- **代码**：github.com/yuekaizhang/RL，分支 `feat/nano-omni-tau1`（上游 NVIDIA-NeMo/RL fork）。
  所有实验 yaml 在 `examples/nemo_gym/grpo_nano_omni_30ba3b_tau1_*.yaml`，
  launcher 在 `yuekai_scripts/`。
- **当前最优 ckpt**：TIS 臂 step_410（tau2 三域均值 58.9，全场第一），
  HF 格式在 `tau2_eval/ckpts/nano_tis_step410_hf`。
- **在跑/排队的实验**（原集群，2026-09-11 时点）：
  A 臂 tis-cleantpl（全量数据 × 干净模板）、B 臂 tis-rmin（A + ratio_min 0.1）。
  迁移后如需重跑，两个 yaml 都已提交在分支上。
- **三条铁律**：① val 集 512 行永不改动；② 干净模板训的模型必须用干净模板测
  （反之亦然）；③ 难度过滤数据必须用目标模型自画像，跨模型难度不迁移。
- **两大 nano 顽疾**：脏 system 渲染（已修，repo 模板 + config 键）与
  TIS 臂随机空输出/重复循环（B 臂正在验证 ratio_min 0.1 修法）。
  详见 findings-and-recipes.md。
