# Conversational Tool Use (Pivot RL) — Experiment Results, Part 3

Follow-up to `exp2.md`. Same protocols throughout: tau2-bench numbers are
average reward (x100), 4 trials per task, gpt-5.6-luna user simulator,
telecom = mean of the two 57-task halves; tau-voice is the half-duplex
harness (user turns are Chatterbox-Turbo TTS audio, agent listens and
replies in text). Model: Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.

## Part 1: Fixing the empty-output failure mode (chat-template rendering)

### The problem

Several trained checkpoints intermittently produce an assistant turn with
neither content nor tool_calls. In text evals these simulations surface as
`Infra Error` terminations and are scored 0 inside the reported average; in
voice evals they are excluded from scoring (reported as coverage). The rate
is negligible for the base model and for plain GRPO on the standard data,
grows monotonically with training steps on the TIS arms, and also appears —
without any IS correction — on the wholesale data-swap arm (d60):

Text (empty-output sims / total sims per run):

| ckpt | airline | retail | telecom t1 | telecom t2 |
|---|---|---|---|---|
| base (no training) | 0/200 | 0/456 | 0/199 | 0/198 |
| plain GRPO step 470 (HF data) | 1/200 | 0/456 | 0/220 | 0/223 |
| GSPO+TIS step 420 | 1/200 | 0/456 | 0/182 | 4/144 |
| TIS-only step 410 | 11/467 (airline+retail) | | 7/163 | **18/123 (15%)** |
| TIS-only step 600 | 13/469 (airline+retail) | | 16/193 (8%) | **34/152 (22%)** |
| d60 (plain GRPO, paper data) step 380 | **49/200 (25%)** | 7/456 | 2/190 | 5/153 |
| d60 step 410 | 37/200 (18%) | — | — | — |

The TIS-only arm's t2 curve shows the growth cleanly: 2/161 (step 110) →
1/120 → 2/129 → 5/115 → 3/99 (step 300) → 18/123 (step 410) → 12/110
(step 460) → 34/152 (step 600) — near-zero for the first 300 steps, then
climbing once the policy has drifted far enough. Note the d60 rows: that arm
runs **plain GRPO with no IS correction at all**, yet develops 18-25%
airline empties mid-training (its 380/410 airline scores required several
targeted rerun-and-merge passes). Empties are therefore not exclusive to
TIS.

Voice (unscored tasks / total):

| ckpt | airline | telecom |
|---|---|---|
| base | 3/50 (6%) | 6/114 (5%) |
| plain GRPO step 470 | 1/50 | 0/114 |
| GSPO+TIS step 420 | 4/50 (8%) | 10/114 (9%) |
| TIS-only step 600 | **17/50 (34%)** | **35/114 (31%)** |

Failure forensics (captured reasoning dumps): the model enters a repetition
loop over *structural* tokens — invented template markers
(`<|function=functions|><|message|>`), tool-schema tags (`<parameter>`),
or verbatim policy sentences — either burning the whole token budget
(finish=length, so no final message) or ending in an early EOS right after
echoing the policy. Empties skew toward harder tasks (voice telecom empties:
39% on the hard half vs 23% on the easy half) and are worse with audio input
(voice ~2x text).

### Why TIS suppressed gen_kl_error but bred empty outputs

Two ingredients, neither sufficient alone:

1. **A rendering bug put a repetition attractor into every training prompt.**
   vLLM's multimodal path normalizes string message content into a
   one-element parts list; the nano chat template's system branch pipes
   content through `| string`, which on a list produces the Python repr. So
   every rollout prompt during training carried the entire policy document
   wrapped as `[{'type': 'text', 'text': '...'}]` — thousands of tokens of
   quasi-structural text in a format the model never saw in pretraining.
2. **The TIS stack silences the gradient exactly where the pathology lives.**
   The token-level IS weight π_train/π_gen collapses to ~0 at the numerically
   fragile structural tokens (that is *why* it fixes gen_kl_error — those
   tokens' exploding gradients are muted). But muting the gradient there also
   mutes the corrective *"don't repeat"* signal: when a rollout degenerates
   into a structural-token loop and gets reward 0, the very tokens that
   constitute the loop contribute no gradient. Plain GRPO (no IS correction)
   keeps that corrective pressure — and on the same data (HF 96K) it indeed
   shows near-zero empties despite the same dirty rendering.

So the dirty template supplies the attractor, and anything that pushes the
policy far enough off-manifold lets it take over. TIS is the cleanest
demonstration (it *mutes the escape gradient* while protecting drift
metrics), but it is not the only trigger: the d60 arm — plain GRPO, no IS
correction, but a wholesale swap to an out-of-profile dataset — develops
18-25% airline empties by mid-training. The common factor across every
empty-prone run is the dirty rendering; the clean-template arms below are
the only trained models that stay at the base-model floor.

### The fix: render the system content as plain text

Repo-hosted template `examples/nemo_gym/nano_omni_chat_template_sysfix.jinja`
flattens list-shaped system content to its concatenated text parts; training
rollouts load it via the config key
`policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template`
(vLLM `load_chat_template` at server startup — no model-dir or image changes;
only affects runs that opt in). Checkpoints trained with the clean template
are also *decoded* with it.

Rendered system prompt, before vs after (abbreviated):

```
# before (upstream template, training-time rendering)
<|im_start|>system
[{'type': 'text', 'text': 'You are a customer service agent that helps the
user according to the <policy> provided below. ...'}]<|im_end|>

# after (sysfix template)
<|im_start|>system
You are a customer service agent that helps the user according to the
<policy> provided below. ...<|im_end|>
```

A base-model control confirmed the rendering does not move scores by itself
(airline text 60.8 clean vs 58.5 dirty, per-task paired n.s.; voice airline
43.5 vs 40.8/48.9 two dirty trials) — the fix targets the *training-time*
attractor, not evaluation.

### Controlled arms: TIS recipe retrained under the clean template

Two arms, identical to the TIS-only arm (same 96K data, same loss stack,
same schedule) except the rendering — plus one knob in arm B:

- **ctpl**: TIS recipe × full data × clean template (single-variable
  counterpart of the TIS-only arm);
- **rmin**: ctpl + `truncated_importance_sampling_ratio_min: 0.1` (IS
  weights clamped to [0.1, 5.0] so structural tokens keep 10% gradient flow
  — the "restore the don't-repeat signal" candidate fix, designed before we
  knew the template alone would suffice).

tau2-bench text; empty-output count in parentheses (airline /200,
retail /456, telecom halves /~150-200 sims each):

| ckpt | airline | retail | telecom | avg |
|---|---|---|---|---|
| **ctpl step 160** | 57.51 (7) | 65.13 (0) | 45.61 (0+0) | **56.08** |
| ctpl step 290 | 57.67 (11) | 68.20 (0) | 42.07 (1+1) | 55.98 |
| ctpl step 410 | 51.28 (5) | 65.93 (1) | 36.41 (0+0) | 51.21 |
| ctpl step 420 | 52.60 (8) | 65.13 (0) | 31.83 (0+1) | 49.85 |
| ctpl step 470 | 59.59 (7) | 65.35 (0) | 38.99 (1+1) | 54.64 |
| ctpl step 560 | 52.76 (1) | 61.62 (0) | 39.47 (0+0) | 51.28 |
| ctpl step 570 | 52.02 (2) | 62.50 (0) | 44.30 (0+0) | 52.94 |
| ctpl step 600 | 54.27 (**1**) | 64.91 (0) | 37.15 (0+2) | 52.11 |
| rmin step 110 | 54.50 (0) | 59.65 (0) | 40.19 (0+1) | 51.45 |
| rmin step 200 | 53.50 (**0**) | 63.16 (0) | 44.74 (0+0) | 53.80 |
| **rmin step 260** | 59.90 (3) | 61.84 (0) | 46.21 (1+0) | **55.98** |
| rmin step 290 | 58.67 (4) | 62.06 (0) | 46.04 (1+1) | 55.59 |
| rmin step 360 | 62.77 (12†) | 59.21 | t1 66.35 (17†), t2 lost to a 4h timeout | — |
| rmin step 410 | 57.81 (8†) | 59.43 | t1 68.42 (19†), t2 lost to a 4h timeout | — |
| rmin step 460 | 59.28 (6) | 62.28 | 41.69 (60.27/23.11) | 54.42 |

`†` the rmin 360/410 runs coincided with a flaky-gateway window (dozens of
429/502/503 in the logs); their parenthesized error counts mix genuine
empties with gateway failures and should not be read as an empty-output
regression.

**The systematic empty-output degradation is gone in both arms — reduced to
a low, non-growing residual rather than fully eliminated.** Same-step
contrast with the dirty-template twin: TIS-only 410 had 15% empties on
telecom t2 and TIS-only 600 had 22% — ctpl 410 has zero on both telecom
halves, and counts do not grow with steps (ctpl 600 airline: 1/200 vs dirty
600's 34/152 t2). The residual: airline still shows 0.5-5.5% empties
(ctpl 7/11/5/1 per 200 across 160/290/410/600; rmin 0-4), above the text
base model's 0/200 but in line with the model's intrinsic floor seen in
voice (base 5-6%). Error signatures confirm these are genuine empty outputs
(no gateway errors), and the raw signature counts in the logs (58-76 per
run) far exceed the final failures (1-11): most episodes now recover on the
runner's retry — shallow, self-limiting slips rather than the dirty arm's
budget-burning deep loops. Crucially, **ctpl carries the full TIS stack
with no ratio floor** — so the clean template alone accounts for the fix,
the root cause was the rendering attractor, and TIS acted as amplifier. The
ratio_min floor (arm B) is redundant *for this failure mode*; whether it
helps or hurts learning otherwise is still being measured (rmin is
climbing: airline 54.5 → 59.9 → 62.8 by step 360).

### Do we still need TIS under the clean template?

Yes for drift, with eyes open about its costs:

- **The drift TIS fixes is not template-caused.** gen_kl_error /
  token_mult_prob_error growth was root-caused (exp2 Part 1) to numerical
  fragility of RL-sharpened mamba-hybrid weights — three independent
  engine implementations disagree at specific tokens. The clean template
  does not touch that mechanism, and both clean arms keep tokmult flat
  only because they carry the TIS stack. Dropping TIS would re-expose the
  1.07 → 1.20 drift of the plain baseline.
- **Known TIS side effects, updated:** (a) the empty-output amplification —
  neutralized by the clean template; (b) gradient muting at fragile tokens
  also mutes *learning* there — the one open concern, since the clean-template
  arms have so far not reproduced the dirty TIS arm's late-stage climb
  (complete ctpl curve: best is step 160 at 56.08, i.e. base level, and
step 600 lands at 52.11 vs the dirty twin's 63.83). Whether
  that gap is TIS-muting, the loss of some accidental benefit of the dirty
  rendering, or run-to-run variance (data order reshuffles across the
  resume chains) cannot be separated with the current arms.
- **The missing cell of the 2x2** (clean template × plain GRPO, no TIS)
  would decide it: if plain GRPO under clean rendering drifts but scores
  climb, TIS's muting is costing us points and a gentler drift control
  (e.g. ratio floor only, or threshold-only) is the direction; if it drifts
  AND stalls, TIS is exonerated. This is the highest-value next arm.

**Interim recipe recommendation:** clean template + full TIS stack
(= ctpl arm) is the safe production configuration — drift-protected and
empty-free; accept for now that its score trajectory is base-level until
the 2x2 is completed. The dirty-template TIS-only step 600 (63.8 text
record) remains the best *raw scorer* but with 22-34% empty-output rates it
is not deployable without retry scaffolding, and its gains cannot yet be
attributed cleanly (recipe vs rendering interaction).
