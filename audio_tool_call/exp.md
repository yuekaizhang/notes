# Conversational Tool Use (Pivot RL) — Experiment Results

All numbers are tau2-bench average reward (x100), 4 trials per task unless noted.
Three domains: airline / retail / telecom, plus the unweighted 3-domain average.

---

## Part 1: Qwen3-30B-A3B-Thinking-2507, text-only reproduction

We reproduce the PivotRL paper (arXiv:2603.21383) conversational tool use setting.
The paper uses gpt-4.1 as the user simulator, which we cannot access. We therefore
evaluate under two substitute user simulators: Qwen3-30B-A3B-Instruct-2507 (local)
and gpt-5.6-luna (internal gateway). The user simulator quality strongly affects
absolute scores, so rows are only comparable within the same user sim.

### Table 1: paper numbers vs. our re-evaluation of the released checkpoints

| Model | User sim | airline | retail | telecom | avg |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Thinking-2507 (paper reported) | gpt-4.1 | 50.00 | 54.39 | 28.65 | 44.35 |
| + Pivot RL (paper reported) | gpt-4.1 | 58.67 | 69.01 | 63.74 | 63.81 |
| Qwen3-30B-A3B-Thinking-2507 (local reproduced) | Qwen3-30B-Instruct-2507 | 53.50 | 48.46 | 24.56 | 42.17 |
| + Pivot RL (paper ckpt, local reproduced) | Qwen3-30B-Instruct-2507 | 58.50 | 62.42* | 46.71 | 55.88 |
| Qwen3-30B-A3B-Thinking-2507 (local reproduced) | gpt-5.6-luna | 59.00 | 67.54 | 26.75 | 51.10 |
| + Pivot RL (paper ckpt, local reproduced) | gpt-5.6-luna | 73.50 | 81.58 | 59.43 | 71.50 |

`*` partial run (3+ full trials).

Takeaway: a stronger user simulator lifts scores a lot (esp. telecom, where the
user must execute instructions), and the lift is much larger for the RL model
than for the base model — a capable agent is needed to exploit a capable user.

### Table 2: our own RL training vs. the released RL checkpoint (luna user sim)

We train from the same base with GRPO on the public pivot dataset
(`nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`, 96K rows;
the paper's internal training set is ~314K rows with a harder difficulty mix).
Best checkpoint: step 360.

| Model | airline | retail | telecom | avg |
|---|---|---|---|---|
| Qwen3-30B-A3B-Thinking-2507 (base) | 59.00 | 67.54 | 26.75 | 51.10 |
| + Pivot RL (paper ckpt) | 73.50 | 81.58 | 59.43 | 71.50 |
| + Pivot RL (ours, step 360) | 65.50 | 72.81 | 37.72 | 58.68 |

Takeaway: our RL run clearly improves over base (+7.6 avg) but is 12.8 behind
the released checkpoint; the largest gap is telecom (-21.7). The most likely
cause is training data: we use the 96K public subset while the paper used the
~314K internal set with a harder difficulty distribution.

---

## Part 2: Nemotron-3-Nano-Omni-30B-A3B trained on text data

Same GRPO recipe and the same text dataset
(`nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`), applied to
Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16. Evaluated on text tau2-bench and
on our half-duplex tau-voice harness (user turns are TTS audio, agent must
listen; Chatterbox-Turbo TTS, gpt-5.6-luna user sim).

### Table 3: train on text, test on text (luna user sim)

| Model | airline | retail | telecom | avg |
|---|---|---|---|---|
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (no training) | 58.50 | 64.25 | 44.96 | 55.90 |
| + RL step 160 | 57.65 | 65.13 | 43.79 | 55.52 |
| + RL step 470 | 61.31 | 60.09 | 50.44 | 57.28 |
| + RL step 570 | 60.50 | 62.28 | 47.02 | 56.60 |
| + RL step 600 | 64.00 | 59.65 | 46.71 | 56.79 |
| + RL w/ difficulty-filtered data, step 160 | 51.02 | 62.94 | 41.23 | 51.73 |
| + RL w/ difficulty-filtered data, step 210 | 57.84 | 68.42 | 27.90 | 51.39 |
| + RL no-think, step 260 | 37.00 | 25.44 | 18.86 | 27.10 |

Takeaways:
- On text, RL gives Nemotron-3-Nano-Omni almost nothing (55.9 → 57.3 at best); its telecom
  strength is already in the base model.
- The difficulty-filtered data arm is net negative (retail up, telecom keeps
  degrading with training).
- No-think collapses on the real benchmark even though its training validation
  score matched the think arm — single-step pivot validation does not predict
  multi-turn agent capability.

### Table 4: train on text, test on audio (tau-voice, half-duplex)

airline is 4 trials (200 sims); retail / telecom are 1 trial (±7 noise at 1σ).

| Model | airline (4 trials) | retail | telecom | avg |
|---|---|---|---|---|
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (no training) | 50.26 | 39.82 | 25.93 | 38.67 |
| + RL step 470 | 56.57 | 45.13 | 24.56 | 42.09 |

Takeaway: the text-only RL transfers to speech input: +6.3 on airline at 4
trials (~90% confidence), +5.3 on retail (1 trial). The RL checkpoint also
produces fewer no-output failures (198/200 valid vs 191/200). So although RL
adds almost nothing on text for this model, it does help in the speech
setting.
