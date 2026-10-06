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
| next | further interventions designed from S0's failure modes; larger-model reference | not started |

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

## Repository layout

| path | contents |
|---|---|
| `src/miracle/` | package: dataset, selection, workspace, runtime interface, evaluator wrapper, runner, stats, export, LLM client (`llm/`), S0 agent (`agents/`) |
| `subsets/` | fixed, versioned instance subsets (`dev10` is the development set) |
| `experiments/s0-dev10/`, `experiments/s1-dev10/` | committed evidence of the S0 / S1 runs: manifest, results, patches, full traces, official harness reports/logs |
| `tests/` | offline unit tests (`pytest`); real-harness integration tests (`pytest -m docker`) |
| `docs/infrastructure.md` | design and usage of the infrastructure, LLM client, tool-use probe and the S0 agent |
