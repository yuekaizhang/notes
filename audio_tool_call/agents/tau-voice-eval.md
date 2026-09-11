# tau-voice：自建半双工语音评测

## 装置是什么

自研评测（不在 tau2-bench 上游里）：用户轮文本先经 **Chatterbox-Turbo TTS**
合成音频，agent（omni 模型）**听音频、文本作答**，其余沿用 tau2-bench 任务、
工具与判分。半双工 = 一轮音频进、一轮文本出。用户模拟器大脑仍是
gpt-5.6-luna。代码在 `tau2_eval/run_voice_halfduplex.py` +
`tau2_eval/chatterbox_server.py`。

## 提交

launcher：容器内脚本 `yuekai_scripts/run_voice_eval_job.sh`
（经 sbatch ray.sub 包一层提交，1n8g）。GPU 布局：0-3 agent vLLM TP4，
4 = Chatterbox TTS。

```bash
CONTAINER=<nemo_rl.0803.sqsh> MOUNTS=/lustre:/lustre \
COMMAND="CKPT=/path/hf_ckpt TAG=nano_xxx DOMAIN=retail bash $REPO/yuekai_scripts/run_voice_eval_job.sh" \
sbatch --nodes=1 --gres=gpu:8 --time=4:00:00 ... ray.sub
```

- `DOMAIN` ∈ airline/retail/telecom（默认 airline）；
- 干净模板 ckpt 加 `EXTRA_VLLM_FLAGS='--chat-template <sysfix.jinja>'`；
- 结果：`tau2_eval/voice_results/voice_<TAG>_<DOMAIN>/summary.json`，
  serve/TTS 日志在 `tau2_eval/voice_logs/`。

## 坑表（每条都踩过）

1. **judge 路由（最大坑）**：retail 有恰好 40 个 `nl_assertions` 任务需要
   LLM judge，默认 gpt-4.1 被网关拒 → 这 40 个全报错像"模型失败"。
   脚本已内置修复：`TAU2_NL_ASSERTIONS_MODEL=openai/switchyard/openai/gpt-5.6-luna`
   + `TAU2_NL_ASSERTIONS_ARGS`（带网关 base_url/key）。新集群换网关要同步改。
2. **chatterbox venv 的 python 是悬空的**（原来 build 在容器 /root 下的
   uv CPython，batch 容器里不存在）。脚本的解法：lustre 常驻
   `UV_PYTHON_INSTALL_DIR=/lustre/.../tools/uv_pythons` 的 py311 +
   `PYTHONPATH=chatterbox_venv/.../site-packages` 起 server。
   迁移时 chatterbox_venv 整个目录拷走可复用（纯 site-packages），
   py311 在新集群重装一次即可。
3. **容器 ray venv 的 openai 包过老**会挂 vLLM CLI，脚本已自动
   `pip install -U openai` 兜底。
4. nano serve 三件套别漏：`--mamba-ssm-cache-dtype float32
   --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`。
5. **空输出 ≠ infra 错**：nano（尤其 TIS 臂）会随机发作重复循环烧满预算
   （见 findings-and-recipes.md）。报告时分开写"计分均值"与"覆盖率
   n/总数"，两轮重跑取共同任务对比。缓解：serving 加 repetition_penalty、
   空输出重试。
6. 判分口径：不带 nl_assertions 的任务纯程序判分；带的走 judge。
   summary.json 里逐任务有 reward 与错误标记。

## 已有基线（airline / retail / telecom，计分均值）

- think470：56.6 / 45.1 / 24.6
- tis_410：**59.5 / 46.9 / ~30**（三域全胜，但空输出率更高，
  airline 8/50、telecom 20-29%）
- 详表与勘误见 `../exp3.md`。
