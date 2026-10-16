# MiracleHarness

**Research question:** can smaller open-source LLMs, given better reasoning architecture, tool use, verification and adaptive
inference compute, match much larger models at lower total cost? The long-term goal is an *adaptive cognitive runtime* that picks
reasoning strategy, tools, model size, context, verification and compute budget per task.

First testbed: **SWE-bench Verified**, run through the **official** evaluation harness. The agent runtime is fully decoupled from the
benchmark infrastructure (`AgentRuntime` interface), so architectures can be swapped and compared on identical conditions.

## Status

| stage | what | state |
|---|---|---|
| infrastructure | frozen dataset + subsets, workspaces, official-harness evaluation, run records, comparison | done, tested |
| LLM client | DeepInfra (OpenAI-compatible) client with usage / latency / cost capture, tool-use probe | done, probed live |
| **S0 baseline** | **Qwen3.5-9B + tools, no reasoning architecture, 10-task dev set** | **done (below)** |
| S1 | S0 + continue (not stop) when a reply is cut off by the output-token cap | run on dev10: 7/10 vs S0's 6/10, **but the fix never triggered, so the difference is noise** (below) |
| S2 | S1 + external termination controller (`COMPLETION_CHECK`) | run on dev10: **−21% / −17% cost vs S0 / S1 at equal resolution (6/10)**; fired in 3/10 runs, 3/3 correct (below) |
| checker eval | offline measurement of the S2 checker on labelled patches (confusion matrix, precision/recall) | done: checker adds no discrimination beyond the trigger; false-COMPLETE rate on eligible states not estimable (1 negative) |
| next | more labelled negatives with test evidence; widen trigger coverage; replicate runs | not started |

## Experiment S0 — Qwen3.5-9B + tools, no reasoning architecture

The baseline every later architecture is compared against. Deliberately minimal and **not tuned from its failures**.

| setting | value |
|---|---|
| model | `Qwen/Qwen3.5-9B` via DeepInfra (the ~8B-class Qwen available there) |
| decoding | temperature 0, `reasoning_effort=none`, `max_tokens=4096` per step |
| budget | max 50 agent steps; no other limits |
| system prompt | "You are a software engineering agent. Solve the given issue using the available repository tools. Inspect the repository, make the necessary changes, and verify your solution with tests when possible." |
| tools | `list_files`, `read_file`, `search_code`, `edit_file` (exact unique string replace), `write_file`, `run_command` (task's conda env), `git_diff` |
| environment | container of the official SWE-bench instance image (same env as grading), network off |
| termination | the agent replies without a tool call, or the 50-step cap; patch = final `git diff` either way |
| dev set | `subsets/dev10.json`: seed-0 random sample of 10 Verified tasks, dataset revision + sha256 pinned, drawn before any agent existed |
| grading | official `swebench` 5.0.2 harness on each patch |

### Results

**6/10 resolved (60%; 95% Wilson CI 31%–83%)**  
Outcomes: 6 resolved, 3 unresolved, 1 empty_patch.

| By difficulty | resolved | instances |
|---|---|---|
| 1-4 hours | 0 | 2 |
| 15 min - 1 hour | 3 | 5 |
| <15 min fix | 3 | 3 |

| By repository | resolved | instances |
|---|---|---|
| astropy/astropy | 1 | 1 |
| django/django | 4 | 4 |
| matplotlib/matplotlib | 0 | 1 |
| sphinx-doc/sphinx | 0 | 1 |
| sympy/sympy | 1 | 3 |

| Resource | Total | Mean / task | Median / task |
|---|---|---|---|
| Prompt tokens | 9,835,231 | 983,523 | 1099107 |
| Completion tokens | 74,984 | 7,498 | 7781 |
| Inference cost (USD, provider-reported) | $0.995 | $0.099 | $0.111 |
| Agent wall time (s) | 4,109 | 411 | 337 |
| ↳ in LLM calls (s) | 3,175 | 318 | 282 |
| ↳ in tool execution (s) | 439 | 44 | 34 |
| Official-eval test runtime (s) | 120 | | |

- Cost per resolved task: **$0.166**; LLM calls: 444; prompt:completion token ratio 131:1 (context is re-sent every step).
- Mean LLM latency 7.2s/call; mean output 169 tokens/call.

**Trajectories**

- Stop reasons: 6 max_steps, 2 final_answer, 2 length. Steps mean 44.4, median 50.
- Tool calls: 440 total (run_command 153, read_file 116, search_code 80, list_files 40, write_file 23, edit_file 22, git_diff 6).
- Tool errors 53 (12%), invalid arguments 0, failed edits 0.
- Runs with a source edit: 9/10; first edit at step median 15. Test commands per run: median 3 (0 in 3 runs); tested after last edit in 3/10.

| instance | difficulty | outcome | F2P | P2P | steps | stop | first→last edit | tests | prompt tok | out tok | cost | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| astropy__astropy-8707 | 15 min - 1 hour | resolved | 1/1 | 11/11 | 50 | max_steps | 27→50 | 5 | 1,354,338 | 9,177 | $0.137 | 515s |
| django__django-13821 | <15 min fix | resolved | 1/1 | 14/14 | 50 | max_steps | 16→47 | 10 | 1,178,331 | 4,605 | $0.119 | 352s |
| django__django-15561 | 15 min - 1 hour | resolved | 1/1 | 141/141 | 50 | max_steps | 29→50 | 10 | 1,159,489 | 8,082 | $0.117 | 280s |
| django__django-16082 | 15 min - 1 hour | resolved | 2/2 | 170/170 | 50 | max_steps | 15→16 | 11 | 1,240,940 | 7,759 | $0.125 | 319s |
| django__django-17029 | <15 min fix | resolved | 1/1 | 43/43 | 37 | final_answer | 5→5 | 5 | 715,141 | 4,640 | $0.072 | 222s |
| matplotlib__matplotlib-22865 | 15 min - 1 hour | unresolved | 0/3 | 57/57 | 43 | length | none | 0 | 972,472 | 13,176 | $0.099 | 652s |
| sphinx-doc__sphinx-11510 | 1-4 hours | empty_patch | - | - | 41 | length | none | 0 | 700,301 | 9,162 | $0.071 | 675s |
| sympy__sympy-13372 | <15 min fix | resolved | 1/1 | 44/44 | 23 | final_answer | 11→11 | 1 | 200,130 | 3,942 | $0.021 | 239s |
| sympy__sympy-16597 | 1-4 hours | unresolved | 0/3 | 74/74 | 50 | max_steps | 31→47 | 1 | 1,275,364 | 6,638 | $0.129 | 534s |
| sympy__sympy-24066 | 15 min - 1 hour | unresolved | 0/1 | 29/30 | 50 | max_steps | 33→37 | 0 | 1,038,725 | 7,803 | $0.105 | 321s |

_Run `s0-dev10`; dataset `SWE-bench/SWE-bench_Verified` @ `78f471bf655a` (sha256 `87a3d67eb910`); runtime `s0-qwen3.5-9b`; swebench 5.0.2; arm64 host, Docker 29.4.0._

### What it did and where it failed

These are observations only; nothing in the agent was changed in response. "First edit" in the headline stats counts any successful
`edit_file`/`write_file` (scratch scripts included); the per-instance column counts `edit_file` source edits only.

1. **It does not reliably stop.** 6 of 10 runs hit the 50-step cap, 2 ended with a final answer, 2 were cut off by the output-token cap.
   In two resolved runs the fix was in place by step 5 (django-17029) and step 16 (django-16082) and the remaining 32–34 steps were repeated
   verification; in two other resolved runs (astropy-8707, django-15561) the last successful edit came **on step 50**, so the budget was binding.
2. **Late first edit.** Among the 8 runs that made a source edit, the first came at step 5–33 (median 21.5 of 50): much of the budget is exploration.
3. **Two runs ended by truncation, partly an artefact of S0's own design.** In sphinx-11510 and matplotlib-22865 the last step was the model narrating
   its analysis as plain text until the 4096-token cap. The loop treats any reply without a tool call as final, so the run ended mid-investigation,
   before any source edit. This reflects two S0 choices (`max_tokens`, "no tool call = done"), not only model weakness.
4. **Scratch files in patches.** matplotlib-22865's patch is only seven scratch scripts (no source change); sympy-24066's carries ten `test_*.py` scratch
   scripts beside the real edit. Grading tolerated them; a patch policy is a separate decision.
5. **Plausible-but-wrong fixes and thin verification.** sympy-24066 found the right function and idea but returned int `1` where the reference returns
   `Dimension(1)` (fails the target test and breaks `test_issue_20288`). sympy-16597 edited `assumptions.py` but fails all 3 target tests after a single test
   run. No test command was run at all in 3/10 runs, and only 3/10 ran one at or after their last edit.
6. **The tool interface was not the bottleneck.** 0 invalid-argument calls and 0 failed edits in 440 tool calls. The 53 "tool errors" are overwhelmingly
   commands that exited non-zero (failing tests, exploratory scripts), not interface problems.

### Caveats

- **n = 10, one sample.** The interval is very wide (31–83%) and DeepInfra output is not guaranteed deterministic at temperature 0.
  4/10 tasks are Django and all 4 resolved; the other 6 resolved 2/6. Do **not** read 60% as the model's SWE-bench Verified score.
- Wall times come from x86 containers emulated on an Apple-silicon host (slower than native); they affect time, not grading.
- Inference cost is DeepInfra's reported `usage.estimated_cost`; there is no prompt-cache discount in these numbers. Cost is ~99% prompt tokens
  because the full context is re-sent every step (no context management in S0).

### Known issues in the recorded data

- `astropy__astropy-8707` (S0) was generated before fixes to `DockerEnvironment` (see the S1 section for the root cause): a git stderr warning ("paths are ignored by one of your .gitignore files:
  .pytest_cache") leaked into its `patch.diff` (4 leading junk lines) and its `modified_files` list (the real change is 2 files). The official grade is unaffected
  (`git apply` skips leading text; it resolved). Fixed and verified afterwards; the record was left as generated rather than hand-edited.
- Evidence in `experiments/s0-dev10/` was exported with local absolute paths replaced by `<project>`; `instances.jsonl` is omitted (re-derivable from the pinned dataset revision).

### Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env                    # set DEEPINFRA_API_KEY (never committed)
# needs Docker with ~10+ GB free *inside Docker's virtual disk*, x86 images run under emulation on Apple silicon
.venv/bin/miracle run --runtime s0 --subset dev10 --name s0-dev10 --pipeline --rm-images -j 1
.venv/bin/miracle stats s0-dev10        # the report above
.venv/bin/miracle table s0-dev10        # per-instance outcome next to trajectory statistics
.venv/bin/miracle export s0-dev10 experiments/s0-dev10
```

## Experiment S1 — S0 + continue on `finish_reason == "length"`

**Change (only this):** a reply cut off by the output-token cap is no longer treated as "done"; it stays in the conversation and the model is told
"Your previous response was truncated by the output token limit. Continue the investigation from where you stopped. Use the available tools when appropriate."
Continuations count against the same 50-step budget. Everything else is identical to S0 (details: `docs/infrastructure.md`). Evidence: `experiments/s1-dev10/`.

| | S0 | S1 |
|---|---|---|
| resolved | 6/10 (60%) | **7/10 (70%)** (95% CI 40–89%) |
| steps ending in `finish_reason=length` | 2 (sphinx-11510, matplotlib-22865) | **0** |
| nudges sent | 0 (not implemented) | **0** |
| cost / per resolved | $0.995 / $0.166 | $0.945 / $0.135 |
| agent wall time | 4,109 s | 4,317 s (matplotlib alone 1,944 s: slow provider steps, not the nudge) |

| instance | S0 | S1 |
|---|---|---|
| sympy-24066 | unresolved (29/30 P2P) | **resolved** |
| sphinx-11510 | empty patch (ended by `length`) | unresolved (0/2; a real patch, 50 steps) |
| matplotlib-22865 | unresolved 0/3 (ended by `length`) | unresolved 1/3, 1 regression |
| the other 7 | same outcome (6 resolved, sympy-16597 unresolved) | same outcome |

**What this does and does not show.**
- **The S1 code path never ran.** No S1 step ended in `length`, so S1 behaved exactly like S0. The +1 (sympy-24066) and the changed outcomes on sphinx and matplotlib are
  run-to-run variation, not an effect of the fix. In particular, both instances that S0 truncated simply took a different path in S1 and were never truncated.
- **Truncation is rare:** 2 of ~440 S0 steps (0.5%) and 0 of ~440 in S1. A 10-task run cannot evaluate a fix for a rare event.
- **Useful by-product: a noise floor.** S1 is effectively a replicate of S0 (temperature 0, same model, same prompt). Across 10 tasks, 1 changed resolved-status, 2 more changed outcome class
  or test counts, and several trajectories changed length (e.g. django-13821 ran to the step cap in S0 but ended with a final answer in S1). Differences of one task between runs on this set should be read as noise.
- **The fix itself works when triggered:** unit-tested, and live against DeepInfra with forced truncations (the API accepts the truncated reply followed by the nudge and the model continues).
  A truncated *tool call* could not be provoked live. A causal test needs the intervention applied from an actual truncation state (see next steps).

**Bug found and fixed during this run.** Patch extraction skipped its binary-file filter whenever a gitignored `.pytest_cache` existed (`git add` exits 1 in that case and the filter was chained with `&&`),
so binary files an agent produced (matplotlib PNGs) leaked into patches as unappliable "Binary files differ" stubs. It was the root cause of the earlier astropy-8707 blemish too (S0: the
warning text, with the same trigger). Fixed in `DockerEnvironment` with a regression test (`tests/test_docker_env_staging.py`, fails on the old code). Impact: all S0 patches other than astropy-8707 were clean, so the S0 numbers stand.
S1's matplotlib patch had 11 leaked PNG stubs; I removed exactly those entries from the recorded patch (original kept as `patch.original.diff`, the original grade under `eval/superseded/`, a note in `result.json`) and re-graded:
identical test outcomes (unresolved, F2P 1/3, 1 regression), so the grade was unaffected.

## Experiment S2 — S1 + External Termination Controller

**Question:** can external executive control cut wasted agent steps and cost without hurting resolution? **Change (the only one vs S1):** a `COMPLETION_CHECK`
state. When there is evidence the agent may be done, the loop pauses and one small LLM call (same model, no tools, `max_tokens=300`, reasoning off) sees only the
issue, the current git diff, the modified files and the relevant test output, and answers `COMPLETE` or `INCOMPLETE: <unresolved requirement>`.
`COMPLETE` terminates the run; `INCOMPLETE` is recorded and the normal loop continues **unchanged** (nothing from the check is shown to the agent: no feedback, plan or
reflection). Unparseable/truncated replies and API errors mean "continue": the controller never terminates on uncertainty.
Model, system prompt, tools, step budget, dataset (`dev10`) are the same as S0/S1; check calls are not agent steps (not counted against the 50), but their cost is in the totals.

**Trigger (one predicate, shared by the live controller and the offline analysis, `agents/termination.py`):** a successful `edit_file` on a non-test path, followed by a test
command (pytest, runtests.py, ...) that succeeded (exit 0, tests actually ran, no failures). Rails: skipped if the diff doesn't modify an existing non-test file, if the diff is
unchanged since the last check, after 5 checks, and on the last step. Evidence: `experiments/s2-dev10/`.

### Result

| run | resolved | steps (total / mean) | prompt tok | completion tok | cost | $/resolved | agent time | check calls | check cost |
|---|---|---|---|---|---|---|---|---|---|
| s0-dev10 | 6/10 | 444 / 44.4 | 9,835,231 | 74,984 | $0.995 | $0.166 | 4,109s | 0 | $0.0000 |
| s1-dev10 | 7/10 | 442 / 44.2 | 9,343,709 | 69,231 | $0.945 | $0.135 | 4,317s | 0 | $0.0000 |
| s2-dev10 | 6/10 | 420 / 42.0 | 7,807,109 | 56,397 | $0.789 | $0.132 | 4,224s | 3 | $0.0004 |

| vs S0 / vs S1 | steps | prompt tokens | completion tokens | cost |
|---|---|---|---|---|
| S2 | 420 | 7.81M | 56.4K | $0.789 |
| change | −5.4% / −5.0% | −20.6% / −16.4% | −24.8% / −18.5% | **−20.7% / −16.5%** |

**Resolution is preserved.** S2 resolved 6/10, the same six tasks as S0; S1's extra task (sympy-24066) is one the controller never touched, i.e. baseline noise (S0 and S1 are
replicates and already differ by one). **Where the controller fired it was right every time: 3 checks, 3 `COMPLETE`, 3 terminated runs, all 3 resolved.**

**The saving is attributable to those three runs.** Using the 7 runs where no check fired as the control:

| | S0 | S1 | S2 |
|---|---|---|---|
| cost, 3 controller-terminated runs | $0.334 | $0.297 | **$0.135** (−60% / −54%) |
| steps, same 3 runs | 137 | 125 | **82** (−40% / −34%) |
| cost, 7 runs with no check (control) | $0.660 | $0.648 | $0.654 (±1%) |

The check itself is nearly free: 3 calls, 3,657 prompt + 6 completion tokens, **$0.0004** (0.05% of S2's cost). The offline analysis of the recorded S0/S1 traces predicted that 29–32% of cost
lies after the first candidate (`miracle candidates`); S2 captured ~17–21% overall, consistent with that bound.

Per instance (outcome · steps · cost · stop reason):

| instance | s0-dev10: outcome · steps · cost · stop | s1-dev10: outcome · steps · cost · stop | s2-dev10: outcome · steps · cost · stop |
|---|---|---|---|
| astropy__astropy-8707 | resolved · 50 · $0.137 · max_steps | resolved · 50 · $0.142 · max_steps | resolved · 34 · $0.071 · completion_check |
| django__django-13821 | resolved · 50 · $0.119 · max_steps | resolved · 40 · $0.068 · final_answer | resolved · 50 · $0.083 · max_steps |
| django__django-15561 | resolved · 50 · $0.117 · max_steps | resolved · 50 · $0.135 · max_steps | resolved · 50 · $0.117 · max_steps |
| django__django-16082 | resolved · 50 · $0.125 · max_steps | resolved · 50 · $0.125 · max_steps | resolved · 23 · $0.033 · completion_check |
| django__django-17029 | resolved · 37 · $0.072 · final_answer | resolved · 25 · $0.030 · final_answer | resolved · 25 · $0.031 · completion_check |
| matplotlib__matplotlib-22865 | unresolved · 43 · $0.099 · length | unresolved · 50 · $0.128 · max_steps | unresolved · 50 · $0.108 · max_steps |
| sphinx-doc__sphinx-11510 | empty_patch · 41 · $0.071 · length | unresolved · 50 · $0.096 · max_steps | unresolved · 50 · $0.098 · max_steps |
| sympy__sympy-13372 | resolved · 23 · $0.021 · final_answer | resolved · 27 · $0.025 · final_answer | resolved · 38 · $0.050 · final_answer |
| sympy__sympy-16597 | unresolved · 50 · $0.129 · max_steps | unresolved · 50 · $0.098 · max_steps | unresolved · 50 · $0.109 · max_steps |
| sympy__sympy-24066 | unresolved · 50 · $0.105 · max_steps | resolved · 50 · $0.098 · max_steps | unresolved · 50 · $0.088 · max_steps |

Controller activity:

| instance | outcome | agent steps | stop | checks | check steps | decisions | check tok (in/out) | check cost | tests that triggered |
|---|---|---|---|---|---|---|---|---|---|
| astropy__astropy-8707 | resolved | 34 | completion_check | 1 | [34] | COMPLETE | 1454/2 | $0.0001 | s34: `cd /testbed && python -m pytest astropy/io/fits/te` |
| django__django-13821 | resolved | 50 | max_steps | 0 | [] | - | 0/0 | $0.0000 | - |
| django__django-15561 | resolved | 50 | max_steps | 0 | [] | - | 0/0 | $0.0000 | - |
| django__django-16082 | resolved | 23 | completion_check | 1 | [23] | COMPLETE | 614/2 | $0.0001 | s23: `cd /testbed/tests && python runtests.py expression` |
| django__django-17029 | resolved | 25 | completion_check | 1 | [25] | COMPLETE | 1589/2 | $0.0002 | s25: `cd /testbed/tests && python runtests.py model_meta` |
| matplotlib__matplotlib-22865 | unresolved | 50 | max_steps | 0 | [] | - | 0/0 | $0.0000 | - |
| sphinx-doc__sphinx-11510 | unresolved | 50 | max_steps | 0 | [] | - | 0/0 | $0.0000 | - |
| sympy__sympy-13372 | resolved | 38 | final_answer | 0 | [] | - | 0/0 | $0.0000 | - |
| sympy__sympy-16597 | unresolved | 50 | max_steps | 0 | [] | - | 0/0 | $0.0000 | - |
| sympy__sympy-24066 | unresolved | 50 | max_steps | 0 | [] | - | 0/0 | $0.0000 | - |
| **total** | | | **3 terminated by controller** | **3** | | | **3,657/6** | **$0.0004** | |

### What this does and does not show

- **Coverage is the limit, not accuracy.** The controller fired in only 3/10 runs. In the other 7 the trigger never held: tests failed to run (`No module named pytest`, wrong runner path),
  the agent validated with ad-hoc scripts rather than a test runner, or it never edited a source file. Those runs behave exactly like S1. Savings are bounded by trigger coverage.
- **The `INCOMPLETE` path never ran in the experiment.** All 3 checks said `COMPLETE` on correct patches, so I cannot tell a discriminating checker from a rubber stamp, and there is no
  evidence yet on false-`COMPLETE` rate (stopping on an unfinished fix). That path is unit-tested only. This is the main open risk.
- **Wall time did not improve in this measurement, and it is not comparable across runs:** agent time was 4,224 s (S2) vs 4,109 / 4,317 s, but DeepInfra throughput during the S2 run was
  14.5 tok/s vs 23.6 / 21.3 (mean 9.3 s per call vs 7.2 / 7.4). A latency claim needs a rerun under similar provider speed.
- n = 10, a single S2 run; the baselines' spread (6 vs 7 resolved) is the noise floor for resolution comparisons.
- **Provider failure handled by rule.** The first S2 sphinx-11510 attempt ended with a network `ReadTimeout` at step 37 (all retries failed) and graded unresolved on a partial patch. That is an
  infrastructure failure, not an agent outcome (same principle as the full-disk incident), so provider failures now abort the instance and archive their evidence instead of being scored. The failed attempt
  and its grade are kept in `infra_failures/` and `eval/superseded/`; the instance was re-run (unresolved, 50 steps, no check, as in S0/S1). Only this one instance in any run was affected.

## Offline evaluation of the S2 completion checker (measurement only)

**Question:** when the checker says `COMPLETE`, how safe is it to terminate the agent? The checker, its prompt, the trigger and the agent loop are **unchanged** (the frozen S2 commit `cb88dbc`);
`git diff cb88dbc -- src/miracle/agents src/miracle/llm` is empty. Evidence: `experiments/checker-eval/` (`samples.jsonl`, `decisions.jsonl`, `report.md/json`, `dataset_summary.json`, official grading logs).

**Dataset (34 samples from the 30 recorded S0/S1/S2 runs).** *Final:* 29 non-empty final patches (19 resolved, 10 unresolved; label = the run's official grade). *Intermediate:* 5 diffs at steps where candidate
evidence appeared (source edit + successful test) and that differ from the run's final patch (4 resolved, 1 unresolved), reconstructed by replaying the trajectory's `edit_file`/`write_file` calls on a pristine
checkout through the production patch-extraction script. Reconstruction is trusted only if the replay reproduces the recorded final patch byte-for-byte: it did for **all 13** runs with candidates, and the prompts
rebuilt for the 3 real S2 checks are **identical strings** to the ones sent online. Intermediate labels: 1 by identity with a graded final patch, 4 graded with the **official harness**; **0 UNKNOWN**.
Only 5 intermediates exist because most candidate steps re-test the same code (distinct diffs are rare). Each sample gives the checker exactly its online inputs (issue, diff, test output since the last source edit)
through `CompletionController._check` (the live code path), 3 repeats each. A sample is *online-eligible* if the live controller could have invoked the checker on it (candidate evidence and a source change);
the rest (final patches with no passing test after the last edit, which includes **all 10 unresolved finals**) are a stress test on states the checker never sees online. Positive = `COMPLETE`; anything else continues the run.

| subset | n | TP | FP | FN | TN | **COMPLETE precision** (95% CI) | COMPLETE recall | false-COMPLETE rate | base rate resolved |
|---|---|---|---|---|---|---|---|---|---|
| online_eligible (primary) | 16 | 15 | 1 | 0 | 0 | **94%** (72%–99%) | 100% | 100% | 94% |
| all labelled | 34 | 23 | 7 | 0 | 4 | **77%** (59%–88%) | 100% | 64% | 68% |
| final, all | 29 | 19 | 6 | 0 | 4 | **76%** (57%–88%) | 100% | 60% | 66% |
| final, online-eligible | 11 | 11 | 0 | 0 | 0 | **100%** (74%–100%) | 100% | n/a | 100% |
| final, not eligible (stress test) | 18 | 8 | 6 | 0 | 4 | **57%** (33%–79%) | 100% | 60% | 44% |
| intermediate | 5 | 4 | 1 | 0 | 0 | **80%** (38%–96%) | 100% | 100% | 80% |

Decisions were identical across the 3 repeats for 34/34 samples (temperature 0: a stable decision rule). Cost: 102 calls, $0.0183 total ($0.00018 per call, ~1.8K prompt tokens, 1.5 s each).

**Findings.**
1. **The checker adds essentially no discrimination beyond the trigger.** On the 16 online-eligible samples it said `COMPLETE` every time: precision 94% (95% CI 72–99%) equals the 94% base rate of resolved states there. The single wrong one
   (s1 matplotlib-22865 @ step 38, graded unresolved) got `COMPLETE` in 3/3 repeats, so on the only eligible negative the false-COMPLETE rate is 1/1. The S2 run was safe because the trigger (a source edit followed by a passing
   test) already selects mostly-correct states (15/16 resolved), not because the checker judged well.
2. **On states without test evidence it rubber-stamps plausible patches.** Of the 9 unresolved diffs that actually change source code, the checker rejected only 2 (22%): sphinx-11510 (s1) and sympy-16597 (s2), both with a
   semantically relevant reason. The other 2 rejections were diffs with no source change at all (scratch/test files only), which the live controller already excludes before calling the checker. Stress-test precision is 57% vs a 44% base rate.
3. **Recall is 100% (FN = 0):** it never blocked a correct patch, so it never cost efficiency here. It behaves as a high-recall terminator that says `INCOMPLETE` only when the diff plainly lacks a source fix or contradicts the issue.
4. **No recoverable run was observed being cut short:** among the 16 eligible states the one unresolved state (matplotlib) also ended unresolved. But the checker also would not have prevented that loss.

**Caveats.** Small and correlated: the 16 eligible samples are 14 distinct diffs over 7 instances with exactly **one** unresolved diff; the whole set is 29 distinct diffs over 10 instances (11 unresolved). The precision interval on
the eligible set is wide and the false-COMPLETE rate there is not estimable from one negative. Labels are patch-level (a resolved intermediate is one that the official tests accept, not a proof of semantic completeness); the harness is
assumed deterministic. Negatives are concentrated in matplotlib/sympy/sphinx.

**Implication (no change made).** The safety measured for S2 so far is the trigger's, and the checker's own false-COMPLETE rate on eligible states is unmeasured because the data holds one negative. Estimating it needs more labelled negatives that
carry test evidence: more instances, graded intermediate edit-states, or synthetic partial fixes (e.g. gold patches with hunks removed, graded by the harness). Reproduce:
`miracle checker-eval build && miracle checker-eval grade && miracle checker-eval run --repeats 3 && miracle checker-eval report`.

## Repository layout

| path | contents |
|---|---|
| `src/miracle/` | package: dataset, selection, workspace, runtime interface, evaluator wrapper, runner, stats, export, LLM client (`llm/`), S0 agent (`agents/`) |
| `subsets/` | fixed, versioned instance subsets (`dev10` is the development set) |
| `experiments/s0-dev10/`, `experiments/s1-dev10/`, `experiments/s2-dev10/`, `experiments/checker-eval/` | committed evidence of the S0 / S1 / S2 runs and the checker evaluation: manifest, results, patches, full traces, official harness reports/logs |
| `tests/` | offline unit tests (`pytest`); real-harness integration tests (`pytest -m docker`) |
| `docs/infrastructure.md` | design and usage of the infrastructure, LLM client, tool-use probe and the S0 agent |
