# exp4 · Part 1: Wrapping up the data-side experiments — two action types, mix boundaries, and the final pick

Interactive overview (dataset lineage / inverted-U scatter / mechanism evidence / results table, zh-en toggle):
https://claude.ai/code/artifact/c6b9c72d-1f63-4aaa-89cc-d76db779523a

Coverage: 7 datasets (D1–D7), 11 training arms (every data-swap arm among #1–#17 in the
master table). Source: `zh/nano_omni_master_table.md`, tables 1/2/2b/4 (2026-10-12 edition).

## Background: the data splits into two action types, and the verifier treats them unequally

Each pivot row expects exactly one of two actions:

| Action type | How it is scored | Mean nano score |
|---|---|---|
| message (reply to the user) | any text reply earns 1.0; content is not judged | **0.724** |
| function_call (call a tool) | tool name + every argument compared item by item | **0.168** |
| of which transfer_to_human_agent (a fc subtype) | same as above | 0.161 |

This asymmetry is the root of everything below: **filtering by difficulty score ≡ filtering
by action type**. Verified by bucketing the 96K set by nano's own scores: 76.2% of the
dropped "easy" bucket (r > 0.6) is message, while the kept all-wrong bucket is 96.4%
function_call — in practice, difficulty filtering removes messages.

## Conclusion 1: the message share must not get too low; 29–32% is the only safe range (inverted U)

| Training set | msg share | Best 3-domain avg. vs. same-protocol baseline |
|---|---:|---|
| D1 rand30k (no filter) | 53.1% | −1.4, near neutral |
| **D2 HF 96K (Super3 ≤0.6)** | **32.4%** | **0 to +1.4, the only set that never loses to base** |
| D3 d60 (Qwen3 self-scored ≤0.6) | 29.0% | +2.4 lenient / withdrawn under strict |
| D7 strat60k v1/v3 (mix locked at 30%) | ≈30% | −3 (quality ceiling) |
| D4 dadv (nano refilter, all-wrong dropped) | 24.7% | −4 to −7 |
| D6 all_100k (nano filter on raw pool) | 13.8% | −15 to −23 |
| D5 dadv0 (nano filter incl. all-wrong, DAPO) | 12.7% | behavioral collapse (→14.8) |

- **Too few (≤25%) is always harmful; 13–14% collapses.** The model stops using words:
  telecom transfers to a human early, and v2 adds tool hallucination. This is the single
  data-side variable that can break the model.
- **Too many (53%) is not fatal, just useless.** Any reply earns full marks → zero
  within-group variance, no gradient; half the data idles, net effect near neutral,
  slightly negative.
- The message rows split into two roles, both required (strat60k v1/v2/v3 ablation):
  mixed-score rows teach "talk instead of calling a tool" (v2 lacked them → telecom
  tool-hallucination collapse); naturally all-correct rows stabilize airline
  (v1 lacked them → airline suffered).

### Addendum: who does the scoring decides how much message gets stripped (four-model comparison)

The same "≤0.6" rule, different scoring models, completely different post-filter mixes.
qwen3-omni and nano-omni were scored on **the same 96K rows under the same protocol
(8 samples/row)** — a strict apples-to-apples comparison:

| Scoring model | Filter base | fc / msg after ≤0.6 filter | mean score: msg / fc |
|---|---|---|---|
| Super3 | raw pool → 96K (= D2) | 67.6% / **32.4%** | — |
| Qwen3-think (2507) | raw pool → d60 (= D3) | 71.0% / **29.0%** | — |
| qwen3-omni | 96K, 8x | 72.9% / **27.1%** | 0.455 / 0.290 |
| **nano-omni** | 96K, 8x (= D5) | **87.3% / 12.7%** | **0.724 / 0.168** |
| nano-omni | raw pool, 16x fixed template (= D6) | 86.2% / 13.8% | 0.343 / 0.159 |

- The other three all land at 27–32% msg after filtering; **only nano cuts it to 13%**,
  reproduced across two filter bases and two sampling protocols — not measurement noise.
- The outlier is explained by the mean scores: nano almost never errs on message rows
  (0.724 — it rarely calls a tool when it should talk) and is the weakest at fc (0.168).
  A self-profile has zero noise about "can I do this row", so nano's ≤0.6 filter is a
  precision message remover. Cross-model scores correlate weakly with nano
  (nano↔2507 only 0.31, nano↔qwen3-omni 0.44), so scoring noise "mistakenly" preserves
  message rows — the mix is protected by noise, not by design.
- Hence "self-profiled filtering is harmful" is not a universal rule: in the paper,
  2507 trained successfully on its own self-profiled data. The damage scales with the
  trained model's message error rate, and nano (never errs on messages × weak at tools)
  happens to be the worst combination.

## Conclusion 2: transfer-to-human rows must not get enriched

Transfer (`transfer_to_human_agent`) is an ordinary tool call in the data (a fc subtype,
counted inside the fc share), but behaviorally it is the terminal "give up on the
conversation" action. It scores low (0.161), so difficulty filtering enriches it from
5.1% of the raw pool to 10.5%, and it is the single most frequent tool in all three
filtered sets.

The damage concentrates on telecom t1: multi-fault troubleshooting tasks (4+ chained
user-side device faults on average, the agent has no usable tools and can only guide the
user through multi-turn text), where transferring means certain loss (base's transferred
sims pass 0/93):

| Model | t1 transfer rate | When it transfers (median) |
|---|---:|---|
| base | 40.8% | after the 12th message |
| D4 dadv_260 (msg 24.7%) | 66.2% | 5th message |
| D6 all100k_130 (msg 13.8%) | **79.4%** | **1st message** |
| D1 rand30k_300 (msg 53.1%, control) | 13.6% | — |

t2 tasks are mostly account-side issues where a transfer is often the correct endpoint,
so the damage there is small — the score loss lands precisely on the tasks that must be
solved by talking.

Note: transfer enrichment is not an independent knob. It and the message drop are **two
side effects of the same difficulty filter**, pushing in the same direction. After
strat60k locked transfer back to 5%, telecom no longer transferred early — so the
variable is controllable; but locking the mix only stops the bleeding (see Conclusion 3).

## Conclusion 3 (easy to miss): a correct mix does not save the raw pool — row-level quality is a third variable

"Everything else performs about the same" does not hold. The three D7 variants control
difficulty, mix, and message composition on the raw pool to match 96K, and still stall
at 52–53 (about 3 pts below base); 96K reaches 56. With every measured variable ruled
out, the remaining explanation is that **the Super3 filter also performed row-level
quality selection**. In short: the mix decides whether training collapses; row-level
quality decides where the ceiling sits.

## Final pick

**Return to the open-source HF set
(nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1, 96,456 rows) as is,
with no second-pass filtering or re-balancing.**

Wording matters: the 96K set is itself the product of Super3 ≤0.6 difficulty filtering
(which is exactly why it lands in the msg-32% safe range). "No filtering" means no
additional self-profiled filter / difficulty filter / mix surgery on top — every such
second pass was net negative in our experiments. The truly unfiltered raw pool (D1)
scores −1.4 and is not the optimum.

Caveat: all of the above holds under **the current verifier** (any reply on a message
row earns full marks). If message scoring is reworked to judge content, difficulty
scores would reflect actual difficulty for the first time and self-profiled filtering
would deserve a retest; until then, the data track is closed here.

---

# exp4 · Part 2: Starting RL from an SFT intermediate checkpoint (#0b / #17)

Setup: swap the starting point for deshr's text-SFT intermediate product (18k steps,
`.../users/deshr/models/kevin_nvc4_sft_text_por_ord_hf_ckpt_18000_mcore_to_hf`).
The RL recipe is identical to D2's best fixed-template arm (#9): token-level TIS +
fixed template + HF 96K, 600 steps. Question: does "SFT first, then RL" beat
"RL directly from base"?

## Text tau2 (3-domain average, luna protocol)

| Start | Before RL | Best after RL | RL gain |
|---|---:|---:|---:|
| base | 55.9 (legacy tpl.) / 53.6 (fixed tpl.) | 57.28 (think470) / 56.08 (ctpl160) | +1.4 to +2.5 |
| SFT18k | **41.37** | **47.68** (step390) | **+6.3** |

Per domain (SFT18k line, before RL → best after RL):

| Domain | Before RL | sft390 | Change |
|---|---:|---:|---|
| airline | 48.24 | **61.00** | +12.8, fully repaired, on par with the base-start arms |
| retail | 52.41 | 53.51 | +1.1, unrecoverable |
| telecom | 23.46 | 28.51 | +5.1, still far below the base-start line (~45) |

## Conclusions

1. **The SFT intermediate checkpoint is a strictly worse starting point than base;
   the line is closed.** Its own baseline sits 12–14 pts below base. The RL gain is
   indeed larger (+6.3 vs. +1.4 to +2.5 on the base line — more headroom at a low
   start), but it cannot repay the starting deficit: the endpoint 47.68 is still
   ~6 pts below the base-start line, and the whole RL run plateaus at 44–48.
2. **RL repairs behavior, not capability.** The airline deficit was behavioral
   (conversation flow / formatting) and RL fully repaired it within 600 steps, even
   matching the main line; the retail/telecom deficits are capability-level (domain
   knowledge, tool-argument accuracy) and the same RL barely moves them. This mirrors
   Part 1: this single-step pivot RL reshapes the action prior; it does not add
   capability.
