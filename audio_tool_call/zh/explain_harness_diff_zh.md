# Tau-voice 三种评测 harness 对比说明

背景：同事整理了一张表 (`harness_tau_voice.png`)，对比 Tau-voice 官方标准评测方法、Yuekai 的方法（`tau2-bench-yuekai/voice_halfduplex/run_voice_halfduplex.py`）、Desh 的方法。本笔记核对代码后确认表格与实际实现一致，并逐行解释每个指标含义及三种方法的具体做法。

代码核对涉及的文件：
- `tau2-bench-yuekai/voice_halfduplex/run_voice_halfduplex.py`
- `tau2-bench-yuekai/src/tau2/user_simulation_voice_presets.py`
- `tau2-bench-yuekai/src/tau2/voice_config.py`
- `tau2-bench-yuekai/voice_halfduplex/chatterbox_server.py`

## 核对结论

同事总结的 "Yuekai's harness" 这一列和代码里实际跑的东西对得上，逐行核对如下：

| 表格行 | 表格里写的 | 代码里验证到的 | 是否一致 |
|---|---|---|---|
| User sim Model | GPT-5.6-Luna (temp=0) | `--user-llm-args` 默认 `'{"temperature": 0.0}'`，示例用 `openai/switchyard/openai/gpt-5.6-luna` | ✅ |
| Persona | Minimal and patient | `CONTROL_CONFIG`: `"verbosity": "minimal"`, `"interrupt_tendency": "waits"` | ✅ |
| TTS | Chatterbox with 1 voice | `chatterbox_server.py` 用 Chatterbox-Turbo 内置默认音色（`conds.pt`），除非设 `CHATTERBOX_REF_WAV` 才换成另一个固定音色，也是 1 个 | ✅ |
| 早停 STOP | None | 代码里没有任何 premature-STOP / streaming-protocol 处理逻辑（半双工设计，不需要处理打断） | ✅ |
| Agent prompt | Same as Tau text-agent | `AudioLLMAgent(LLMAgent)` 只重写了 `_generate_next_message` 把用户消息换成音频，`domain_policy=env.get_policy()` 沿用文本版一致的 policy/prompt | ✅ |
| Decoding | temp=0.6, top-p=0.95 | `--agent-temperature` 默认 0.6，`--agent-top-p` 默认 0.95 | ✅ |
| Reasoning | Optional | 通过 `--agent-extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'` 可关可开 | ✅ |
| Tool protocol | 多次工具调用或文本+工具混合 | `AssistantMessage(content=choice.content, tool_calls=tool_calls, ...)` 两者可同时非空，litellm 单次 completion 可返回多个 `tool_calls` | ✅ |
| Episode limit | 100 steps，无 wall-clock 超时 | `--max-steps` 默认 100，Orchestrator 只有 `max_steps`，代码里没有任何 timeout 设置 | ✅ |
| Reward | Official | `run_simulation(orch, evaluation_type=EvaluationType.ALL)`，用 `sim.reward_info.reward`，官方 reward 逻辑 | ✅ |
| Denominator | Runner exceptions omitted | `ok = [r for r in results if r["reward"] is not None]`，`avg_reward` 只在 `ok` 上算；异常任务 reward 为 `None`，被排除在平均分母外，但仍计入 `n_jobs`/`summary.results` | ✅ |

## 每一行指标 & 三种方法的具体做法

### User simulation（用户模拟器怎么"演"用户）

- **Model**：驱动虚拟用户对话逻辑的 LLM。
  - Tau-voice 官方标准用 GPT-4.1。
  - Yuekai 的方法用 GPT-5.6-Luna，temp=0（确定性、可复现）。
  - Desh 的方法同样用 GPT-5.6-Luna，但 temp=1（更随机，用户行为更多样）。
- **Persona**：用户模拟器的性格/说话风格配置。
  - 官方是按任务单独配置（每个 task 有不同人设难度）。
  - Yeukai 的全局固定成"最省话、有耐心"（`verbosity=minimal`, `interrupt_tendency=waits`，对应 `CONTROL_CONFIG`）——即最简单/最干净的用户设定，不测试鲁棒性用的复杂人设（比如打断、口音、背景噪音等 `REGULAR_CONFIG` 及各种 ablation 都没启用）。
  - Desh 的用"标准冗长度"人设。
- **TTS（文字转语音引擎）**：
  - 官方用 ElevenLabs（云端商用 TTS，音色多）。
  - Yuekai 的本地起 Chatterbox-Turbo server，只用 1 个内置音色（除非手动指定参考音频换音色）。
  - Desh 的用 GPT-4o-mini-TTS，11 种音色轮换，更贴近真实电话场景的音色多样性。
- **Premature STOP handling**（用户模拟器过早喊停/打断的处理）：
  - 官方是靠"流式协议"处理（一边收音频一边判断是否要打断/停止）。
  - Yuekai 的方法完全没做这个（半双工设计：用户说完整一轮，agent 才收到，不存在中途打断判断）。
  - Desh 的是"如果 agent 正在等一个 write 动作，就抑制 STOP"（避免用户在 agent 该执行工具调用时抢话打断，导致任务卡死）。

### Agent response generation（被测 agent 怎么生成回复）

- **Agent prompt**：喂给被测模型的 system prompt/policy。
  - 官方固定用官方对话版 prompt。
  - Yuekai 的直接复用 tau2 文本版 agent 的 prompt（没有为语音场景做任何改写）。
  - Desh 的针对"说出来要自然"做了改写（更口语化，去掉一些文本 agent 里适合书面阅读但读出来别扭的格式）。
- **Decoding（解码采样参数）**：
  - 官方跟随各 provider 默认值（不额外指定）。
  - Yuekai 的固定 temperature=0.6, top-p=0.95（有一定采样随机性，但比较保守）。
  - Desh 的用 greedy（贪心解码，完全确定性，无采样随机性）。
- **Reasoning（是否开启模型内部思维链/reasoning）**：
  - 官方跟随 provider 默认。
  - Yuekai 的可选（通过 `--agent-extra-body` 里的 `enable_thinking` 开关控制，看被测模型是否支持）。
  - Desh 的强制开启且不做长度限制（unconstrained）。

### Episode constraints（对话轮次/工具调用的规则约束）

- **Tool protocol**（一轮里能不能既说话又调用工具/能不能连续调用多个工具）：
  - 官方：流式工具调用，由 model orchestrator 处理（更贴近真实语音助手一边说话一边执行动作）。
  - Yuekai 的：允许一轮里多次工具调用，或者文字+工具混合输出（宽松协议，评测友好但不完全模拟真实流式约束）。
  - Desh 的：严格限制成每轮"恰好 1 次工具调用 或 纯文本输出"二选一（更贴近真实电话语音助手一次只能做一件事的约束，也更接近真实产品形态）。
- **Episode limit（对话轮数/时长上限）**：
  - 官方按"时长/tick"来限制（模拟真实通话时长）。
  - Yuekai 的按步数限制：100 步，且没有 wall-clock 超时（不管跑多久，只要没到 100 步就一直跑）。
  - Desh 的：100 步 + 1200 秒（20 分钟）双重限制，哪个先到就停。

### Evaluation（怎么打分）

- **Reward**：
  - 官方 / Yuekai 的都用"Official"（即 tau2-bench 官方定义的 reward：DB 状态检查 + 动作检查 + NL 断言检查的组合）。
  - Desh 的：Official reward 之外，针对"只读类"航空任务（airline read-only tasks）额外加了一层"NL gate"（自然语言层面的额外校验门槛，可能是防止模型嘴上说做了但实际没调用工具的情况）。
- **Denominator（算平均分时，分母怎么定）**：
  - 官方 / Desh：所有 episode 都计入分母（哪怕跑挂了/出异常也算一次失败，分母不变）。
  - Yuekai 的：跑挂的（runner exception，比如网络超时、解析失败）被**从分母里剔除**，不计入平均分——代码里 `ok = [r for r in results if r["reward"] is not None]` 然后 `avg_reward` 只在 `ok` 上求平均，`n_jobs`（分子来源的总任务数）和 `n_scored`（实际计分数量）是分开记录的两个数字。

## 需要注意的一点差异

同事表格里说"Runner exceptions omitted from denominator"是相对官方/Desh"所有 episode 都计入分母"而言的一个**方法论上的差异点**，这会让 Yuekai 的平均分看起来比"把跑挂的也算失败"的方法要乐观一些。如果要跟 Desh 或官方数字做严格对比，可以额外算一个"把跑挂的按 reward=0 计入分母"的版本作为参考，会更 apples-to-apples。
