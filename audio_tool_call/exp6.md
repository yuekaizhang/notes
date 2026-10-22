# Exp 6 — RL on the new pivot-stateful / areal 7.6K data: results and root-cause analysis

Experiment: `pivotstateful-ep1-vc-4n8g`. Starting checkpoint = deshr's `tau_nano35` step-7000 SFT ckpt
(RoPE audio tower, voicechat fork). Data = `datasets_pivot_stateful/train_pivot_ep1_7604.jsonl`
(desh's areal pivot bundle, 7,604 rows, single-turn prediction). 1 epoch = 118 steps, tau1 TIS recipe.
Training reward climbed 0.06 → 0.75 (plateau 0.73–0.87) — the curve looks great, but see below for what
the model actually learned. Evaluated ckpts: step_60 (start of the plateau) and step_118 (end of epoch).

## 1. Headline results

tau2 text (luna protocol, 4 trials; telecom merged from four quarter-splits, task-weighted):

| ckpt | airline | retail | telecom | weighted avg (278) |
|---|---|---|---|---|
| sft7000 start | 42.0 | 44.5 | 25.2 | 36.1 |
| pstate step_60 | 48.0 | 7.3 | 28.0 | 23.1 |
| pstate step_118 | 48.0 | 7.6* | 34.4 | 25.9 |

*step_118 retail recovered from 445/456 sims (job hit the 4h wall).

tau-voice-simple (s2t, 278 tasks, 1 trial):

| ckpt | airline | retail | telecom | weighted avg |
|---|---|---|---|---|
| sft7000 start | 12.0 | 44.7 | 34.2 | 34.5 |
| pstate step_60 | 12.0 | 7.0 | 21.1 | 13.7 |
| pstate step_118 | 8.0 | 5.3 | 16.7 | 10.5 |

## 2. Training-data analysis

### 2.1 Data construction and verifier semantics

Each row is a single-turn prediction: given a conversation prefix, predict the next turn; the verifier
(`tau_pivot_stateful/.../pivot_verify.py`) scores it by `expected_turn_kind`. Construction logic
(`replay_five.py`): at every mined boundary, probe the environment's native authorization checker with
the gold write action — **if it would be accepted** → label `write`; **if it is rejected because the
preceding consent was defective** (any of gate C1 detail correctness / C2 payment description /
C3 exact charge amount / C4 mandated policy reminder unmet) → label `consent_message`, and the gold
action is a *consent repair*: re-issue a confirmation question with the missing/corrected information.
The user's earlier "Yes." does not constitute valid authorization, because the question they answered
was itself non-compliant (verifier docstring: "No synthetic caller affirmation is accepted as evidence
of real authorization").

### 2.2 Label distribution: 94% of rows say "don't act — re-confirm"

The semantics are self-consistent within the policy frame, but the **sampling is extremely skewed**
(rows are mined from the failure boundaries of glm52 production trajectories, whose consents almost
always fail at least one gate):

**7,604 rows by (label × domain):**

| | consent_message (re-confirm) | write (execute) | ratio |
|---|---|---|---|
| airline | 4,138 | 102 | 40 : 1 |
| retail | 2,734 | 88 | 31 : 1 |
| telecom | 262 | 2 | almost no samples |
| other domains (insurance claims / marketplace sellers etc.) | — | 278 | — |
| total | 7,134 (94%) | 470 (6%) | |

Moreover, in 71% of rows the last user message in the prefix is literally "Yes/OK/confirm". The model
cannot see which gate failed; the cheapest policy that fits 94% of the data is "**user says Yes →
unconditionally re-confirm**". The two labels are also asymmetric in reward difficulty (write requires
a single tool call exactly matching the remaining gold action; consent only needs text predicates to
pass), pushing the gradient further toward consent.

**Finding 1: telecom has only 262 consent rows (1/16 of airline), so the "re-confirm" reflex was
barely trained in telecom contexts — telecom is comparatively spared and retains partial write
ability** (write calls per conversation 0.93 → 0.28, not zero; retail/airline go exactly to zero).

**Finding 2: all 470 tool-call-labeled rows are write-type** — `expected_turn_kind` has only two
values (consent_message / write), and a write row's gold must be "exactly one write tool call that
matches the remaining gold action". **Read tools never appear as a training target**; they only occur
inside the prompt prefixes.

### 2.3 Eval-side task structure: write-task share determines blast radius

Share of tau2 tasks whose evaluation_criteria require the assistant to call write tools:

| domain | write-task share | main write ops |
|---|---|---|
| retail | **104/114 = 91%** | return/modify/exchange/cancel order |
| telecom (base 114) | **71/114 = 62%** | enable_roaming, refuel_data, send_payment_request, resume_line |
| airline | **26/50 = 52%** | update_reservation_flights, cancel/book_reservation |

## 3. Result analysis

### 3.1 Cause of death: write actions wholly displaced by a "confirmation loop"

Text and voice die the **same way**: conversations run to completion (user_stop, slightly longer),
read tools are called as usual, but **write actions vanish entirely**. Typical trajectory: user says
"I confirm, do the exchange" → agent prints a markdown summary table and asks "please confirm" again →
user confirms again → another summary … until the user gives up. The DB check then necessarily fails
(401 of 411 failed sims fail on DB state).

| evidence | sft7000 start | pstate step_118 |
|---|---|---|
| text retail write calls (return/modify/exchange/cancel) | 555 (1.22/conv), 71% of convs have ≥1 write | **0 calls, 0%** |
| voice retail write calls | 113 (0.99/episode), 63/114 = 55% of episodes | **0 calls, 0%** |

(The absolute text/voice counts differ purely due to scale: text = 4 trials × 114 = 456 conversations,
voice = 1 trial × 114 = 114 episodes; normalized rates 1.22 vs 0.99 per conversation agree. The residual
comes from the voice protocol being harder — mis-heard IDs/order numbers, a 1200s wall clock and a
100-step cap mean some episodes never reach the execution stage.)

### 3.2 Why only write-type tool calls are killed, not read-type

| domain | read calls / conv | write calls / conv |
|---|---|---|
| airline | 3.68 → 3.79 (slightly up) | 1.23 → **0.00** |
| retail | 5.40 → 4.88 (−10%) | 1.22 → **0.00** |
| telecom | 11.05 → 5.13 (halved, see §3.4) | 0.93 → 0.28 |

The training signal is **position-specific**:

1. Every training boundary sits at the single moment "the user just confirmed — execute or re-confirm?";
   the consent rows' turn_shape predicate requires that turn to be **message-only (any tool call in that
   turn scores 0)**. The penalty is conditioned only on the "confirmation moment" context.
2. Looking up IDs / orders / flights happens in the information-gathering phase earlier in the dialogue,
   whose context looks completely different (the user just gave their identity, not a "Yes"); those
   positions were never touched by training, and the prefixes themselves are full of read function_call
   exemplars, so read behavior keeps being "demonstrated".
3. What the model learns is the precise mapping "**confirmation moment → emit a confirmation message
   (never a tool)**"; and the only place write tools ever appear is right after the confirmation moment —
   so the write action's entire slot in the conversation is displaced, while reads are out of the line
   of fire.

### 3.3 Split by "needs-write / no-write": every gain and loss decomposes cleanly

(text, per-task 4-trial means)

| domain × task type | start | pstate118 | change |
|---|---|---|---|
| airline no-write (24) | 67.7 | **100.0** | +32, perfect |
| airline needs-write (26) | 18.3 | **0.0** | wiped out |
| retail no-write (10) | 87.5 | 85.0 | flat |
| retail needs-write (103) | 40.3 | **0.0** | wiped out |
| telecom no-write (43) | 57.0 | 58.9 | flat |
| telecom needs-write (71) | 6.0 | **15.3** | +9 (the only needs-write gain) |

The telecom quarter-split is consistent with this decomposition:

| quarter | step_60 | step_118 |
|---|---|---|
| q1 (first half of half1) | 30.7 | 54.8 (whole half1) |
| q2 (second half of half1) | 50.5 | ↑ |
| q3 (first half of half2) | 30.7 | 27.7 |
| q4 (second half of half2) | **0.0** | **0.0** (110+ sims end normally via user_stop, reward all 0 — not infra) |

Per-domain total change = f(domain's write-task share × domain's consent-row exposure in training):
retail (91% write × 2,734 consent rows) collapses hardest; airline (52% × 4,138) looks +6 on the surface
because perfect no-write scores mask the needs-write wipeout; telecom (62% × only 262) nets a gain.

### 3.4 Per-domain attribution

**Why did airline no-write go 67.7 → 100 (perfect)?**
These 24 tasks are scored on "**don't act when you shouldn't** + explain policy precisely" (typical:
the user demands cancellation of a non-cancellable booking — basic economy older than 24h, or an
already-departed flight — executing the cancel scores 0). Consent training teaches exactly: emit
confirmation text with exact amounts (C3), compliant payment descriptions (C2), policy reminders (C4),
while **strongly suppressing any action**. On tasks whose correct answer is "do nothing and explain",
that is a strict positive transfer — the start model could be worn down into wrongly executing 25–50%
of the time; pstate never acts, scoring perfect.

**Are the 18 improved telecom tasks no-write? — No: 12/18 are needs-write tasks.**
To be clear: the telecom gain is **not** in the no-write bucket (43 tasks, 57.0 → 58.9, flat) but in the
needs-write bucket (71 tasks, 6.0 → 15.3). The improved scenarios are `mobile_data_issue /
service_issue` multi-fault troubleshooting chains (airplane_mode + bad_network_preference + VPN in
series; overdue_bill_suspension → send_payment_request + resume_line), with several 0 → 1.0 flips.

**Why didn't telecom no-write improve?**
No-write tasks (e.g. guiding a user abroad to toggle roaming; the fix is the user-side toggle_roaming)
test multi-turn guidance **patience and diagnostic correctness**; the start model was already passable
(57). Consent-style "precise confirmation text" helps little here, and telecom has only 262 training
rows — staying flat is expected: neither hurt by the consent reflex (no writes needed) nor taught
anything new.

**Why did telecom needs-write actually improve (6.0 → 15.3)?**
Three factors compound:
1. **Write ability survived**: telecom consent exposure is tiny, write calls didn't go to zero
   (27/33 sims in the improved tasks actually call enable_roaming/send_payment_request/resume_line);
2. **Wasteful diagnostics got pruned**: the start model spams diagnostic tools in troubleshooting tasks
   (check_network_status 594 → 103 globally), burning the dialogue in loops; pstate halves read calls
   but reaches the actionable step faster;
3. **A very low baseline** (6.0): the start model fails these multi-fault chains almost entirely
   (diagnostic loops, premature human transfer); pstate's calmer "confirm-then-act" cadence nets gains
   in the untainted telecom contexts.
Note telecom q4 (28 tasks, 25 needs-write) is 0 for both ckpts — the start model is also near 0 there;
it's a shared capability hole, not damage from the new data.

## 4. Suggested fixes

1. **Label ratio**: consent:write = 94:6 → at least 5:5–7:3; additionally mine write labels at
   boundaries where "the prefix ends with a user confirmation of a *qualified* consent" (exactly the
   rows that currently teach the most harmful mapping).
2. **Domain ratio**: consent rows airline:retail:telecom = 4,138:2,734:262 — balance them to avoid
   domain-specific behavior overwrites.
3. **Mix with the large pool**: co-train with the tau1 96K multi-turn data (stateful at 10–20%), using
   complete "confirm → execute → wrap up" chains to counter the single-turn consent bias.
4. If staying pure-stateful: add a verifier penalty "after the user says Yes to a qualified consent,
   emitting another consent_message scores 0" to kill the confirmation loop directly.

## 5. Assets

- ckpts: `pivot_rl/ckpts/pstate_step{60,118}_hf`; voice results
  `pivot_rl/tvs_results/pstate_step{60,118}/SUCCESS.json`
- text sims: `$EVAL/tau2-bench/data/simulations/pstate{60,118}_*` (telecom splits: t1 + q3/q4; quarter
  task files `$EVAL/telecom_tasks_q{1..4}.txt` — this ckpt thinks long on telecom, half-splits hit the
  4h wall)
- verifier source: `RL-voicechat/3rdparty/Gym-workspace/Gym/resources_servers/tau_pivot_stateful/reference/recipes/tau_voice_rl_pivot/{pivot_verify.py,replay_five.py}`
- cross-model comparison sheet: `zh/tau_voice_simple_sheet.tsv`
- Chinese original: `zh/new_data_areal.md`
