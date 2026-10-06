# miracle — experiment infrastructure

Goal of this stage: given a SWE-bench Verified instance and *any* patch, reproducibly run the
**official** evaluation and record whether it resolved. No reasoning agent lives here.

```
instance ──► workspace ──► AgentRuntime ──► patch ──► official harness ──► result
 (frozen)    (src only,     (src/miracle/    (predictions   (swebench 5.0.2,    (+ our metadata,
              base_commit)   runtime.py)      .jsonl)        unmodified)         joined at the end)
```

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/miracle subset create dev10 --n 10 --seed 0           # frozen, versioned subset
.venv/bin/miracle run --runtime gold --subset dev10 -j 2 --rm-images   # pipeline self-test
.venv/bin/miracle compare runs/<a> runs/<b>   # (names)
.venv/bin/pytest            # fast unit tests;  pytest -m docker  = full pipeline vs real harness
```

Mock runtimes verify every outcome: `gold`→resolved, `noop`→unresolved, `broken`→patch_apply_failed,
`empty`→empty_patch. A real agent is any `AgentRuntime` subclass:
`miracle run --runtime mypkg.agent:MyAgent --runtime-config '{"model": "..."}' --subset dev10`.

## Layout

| module | role |
|---|---|
| `dataset.py` | HF dataset → content-addressed JSONL snapshot (revision + sha256) in `data/` |
| `selection.py` | deterministic subset selection (ids / repo / difficulty / n+seed) → `subsets/<name>.json` |
| `workspace.py` | single-commit, no-remote git checkout at `base_commit` (cached in `cache/`) |
| `runtime.py` | **the only agent-facing interface**: `Task` in, `AgentResult(patch, usage, metadata)` out |
| `mock.py` | gold / noop / broken / empty runtimes |
| `predictions.py` | official `{instance_id, model_name_or_path, model_patch}` JSONL, nothing else |
| `evaluator.py` | runs the official harness as a subprocess; maps its artefacts to a status |
| `runner.py` | run directory, manifest, generate/evaluate/collect, summary, compare |

### Run directory (`runs/<name>/`)
`manifest.json` (dataset rev+sha, instance ids, runtime config, versions, every evaluation attempt) ·
`instances.jsonl` (frozen records fed to the harness) · `agent/<id>/{result.json,patch.diff}` (**our**
metadata: usage, latency, strategy) · `predictions.jsonl` (official) · `eval/` (harness cwd: its `logs/`,
summary, stdout) · `results.jsonl` + `summary.json` (derived; `miracle report <run>` rebuilds them, no Docker).

## Design decisions

- **Agent decoupling.** `Task` is an explicit whitelist (id, repo, base_commit, problem_statement,
  workspace). Gold patch, test patch, FAIL_TO_PASS/PASS_TO_PASS and `hints_text` are never passed.
  Runtimes report `Usage` (tokens, calls, tool calls, cost) + free-form `metadata`; latency is measured by
  the runner, not self-reported.
- **No leakage via git.** A full clone at `base_commit` still contains the upstream fix
  (`git log --all`). Workspaces are a depth-1 fetch of just that commit with the remote removed.
  Tradeoff: no `git log`/`blame` history for the agent. Workspaces are source-only (no Python env).
- **Official tooling, unmodified.** Harness is pinned (`swebench==5.0.2`) and invoked as
  `python -m swebench.harness.run_evaluation` against the *frozen* `instances.jsonl`, with `cwd=runs/<n>/eval`
  so logs are per-run. We only add a status taxonomy on top of its artefacts and record the docker image id used.
- **Denominator.** `resolve_rate = resolved / all selected instances`; empty patches, apply failures,
  timeouts and eval errors all count as unresolved. Infra failures (harness' advisory classifier) are
  flagged in `summary.infra_failures` but not removed from the denominator.

## Things worth knowing (verified, not assumed)

1. **swebench 5.x differs from most docs/blog posts**: `swebench eval verified ...` CLI exists; there is
   no `--cache_level`/`--namespace`; the dataset rows themselves carry `image`, `eval_script`, `log_parser`;
   the harness only *pulls* images (never builds them) unless `--task-repo` is given.
2. **Apple Silicon**: images are published for x86_64 only, and the harness' own pull fails with
   `no matching manifest for linux/arm64`. `evaluator.ensure_images` pre-pulls with `--platform linux/amd64`;
   they then run under emulation (a Django instance takes ~35 s end to end).
3. **Disk**: one Django image ≈ 3 GB locally; the harness never deletes images. Use `--rm-images`
   (removes only images this run pulled). The full 500-task set needs on the order of 100+ GB.
4. **Harness caching** is keyed by `(run_id, model_name, instance_id)`; we use the run name as run_id and
   `--fresh` wipes `eval/` to force re-grading. Empty patches are skipped by the harness (not evaluated).
5. Harness nondeterminism (flaky tests, emulation timing) is possible; record repeated evaluations
   (`manifest.evaluation[]`) rather than assuming one grade is final.

## Not built yet (deliberately)
Parallel generation, agent-side execution inside the instance image, native-arm64 image builds
(`--task-repo`), cost/price tables, statistical comparison (CIs, paired tests) in `compare`.

## LLM client + tool-use capability probe (`src/miracle/llm/`)

Stage goal: verify that DeepInfra inference works, a small Qwen follows instructions, calls tools
correctly, handles multi-step interaction, and that usage/latency/traces are captured. No strategy yet.

```bash
cp .env.example .env     # set DEEPINFRA_API_KEY (or export it); .env is gitignored
.venv/bin/miracle llm models --grep qwen                         # public endpoint, no key needed
.venv/bin/miracle llm chat "Say hi" --raw                        # raw JSON + tokens/latency/cost
.venv/bin/miracle llm probe                                      # 5 capability cases on the toy repo
.venv/bin/miracle llm agent "What does apply_discount do?"       # ad-hoc tool loop; trace saved
```

Config (env or `.env`, overridable by `--model --base-url --temperature --max-tokens --reasoning-effort`):
`DEEPINFRA_API_KEY`, `MIRACLE_MODEL` (default `Qwen/Qwen3.5-9B`), `DEEPINFRA_BASE_URL`,
`MIRACLE_TEMPERATURE` (0.0), `MIRACLE_MAX_TOKENS` (2048), `MIRACLE_REASONING_EFFORT` (unset).

| file | role |
|---|---|
| `config.py` | `LLMConfig.from_env`; key excluded from `public()` so it never reaches traces/manifests |
| `client.py` | `LLMClient` ABC (the provider seam) + `OpenAICompatClient`: retries 429/5xx/network with backoff, returns raw body, tokens, `usage.estimated_cost`, latency, attempts |
| `tools.py` | `list_files` / `read_file` / `search_code`, sandboxed to a root; errors returned as `ERROR: ...` strings |
| `loop.py` | `ToolLoopRuntime(AgentRuntime)`: task → model → tool calls → results → model → answer; full trace in `metadata["trace"]` |
| `probe.py`, `toyrepo/` | cases `plain, list, read, chain (dependent calls), recover (tool error)` with automatic checks |

Output locations: `runs/llm-chat/chat-log.jsonl` (every chat call), `runs/llm-probe/<ts>-<runtime>/{case}.trace.json + report.json`,
`runs/llm-agent/<ts>.trace.json`. Trace schema v1: per step the raw response, content, reasoning text (if returned),
tool calls, usage, latency, retry attempts, and tool results (arguments, validity, output, is_error), plus the final message list.

Notes: Model choice — DeepInfra lists no 8B Qwen; `Qwen/Qwen3.5-9B` ($0.10/$0.15 per 1M tokens, thinking mode, tool calling)
is the closest, `Qwen/Qwen3-14B` the next size. DeepInfra advises avoiding system messages with tool calling, so none is sent by default
(`--system` to add one). Reasoning tokens bill as output; where the provider puts reasoning text/token counts is captured
if present in the response (`reasoning_content`, `completion_tokens_details.reasoning_tokens`) but unverified until the first live call.
The loop's tools are read-only, so it returns an empty patch; the answer is `metadata["final_answer"]`. A provider failure ends the loop with
`stop_reason="llm_error"` and the partial trace instead of raising. Use as a SWE-bench runtime: `miracle run --runtime toolloop --runtime-config '{"model": "..."}' ...`.

### Live findings (Qwen/Qwen3.5-9B on DeepInfra, 2026-10-05, temperature 0)

- **Default (provider) reasoning is on and can degenerate**: `chat "Say hi in five words."` produced a
  repetitive `reasoning_content` loop, hit `max_tokens=2048`, returned **empty `content`**, `finish_reason=length`,
  95 s, $0.0003. Always pass `--reasoning-effort` (or set `MIRACLE_REASONING_EFFORT`) and check `finish_reason`.
- Reasoning text arrives in `message.reasoning_content`; usage has **no** reasoning-token breakdown (they are inside
  `completion_tokens`); `usage.estimated_cost` is present.
- `reasoning_effort=none`: same prompt 1.1 s / 8 tokens. Probe: **5/5 on two runs** (identical token counts on 3 of 5
  cases), ~$0.001, ~36 s LLM time. `low` (max_tokens 4096): **5/5**, ~$0.0009, ~60 s, ~2x output tokens.
- Probe `recover` originally failed because the model silently fixed my typo'd path; prompt changed to name a nonexistent
  file (check unchanged). Tool use itself (valid JSON args, chaining 3 dependent calls, recovery from a tool error) worked.
- Caveat: five tiny cases on a 7-file repo say the plumbing and basic tool use work; they say nothing about SWE-bench difficulty.

## S0 baseline: Qwen3.5-9B + tools, no reasoning architecture (`src/miracle/agents/`)

```bash
.venv/bin/miracle run --runtime s0 --subset dev10 --name s0-dev10 --pipeline --rm-images -j 1   # agent -> official eval, one instance at a time
.venv/bin/miracle table s0-dev10        # per-instance outcome next to trajectory statistics
.venv/bin/miracle run ... --resume      # continue after a stop (infra failures leave no result, so they are retried)
```

**What S0 is** (fixed; do not tune from failures): one tool loop, native function calling, the single system instruction
"You are a software engineering agent. Solve the given issue using the available repository tools. Inspect the repository,
make the necessary changes, and verify your solution with tests when possible.", `Qwen/Qwen3.5-9B`, `reasoning_effort=none`,
temperature 0, max 50 steps, `max_tokens=4096` per step (room for a whole-file write). No planning, reflection, critics, retries
or context management; tool outputs are length-capped only. The agent ends when it replies without a tool call (or at 50 steps);
the patch is the final `git diff` either way.

**Tools:** `list_files`, `read_file` (numbered lines, ranges), `search_code` (git grep), `edit_file` (exact unique string
replace), `write_file`, `run_command` (fresh shell in the task's conda env; timeout), `git_diff`. No patch-application tool: a
9B model writing unified diffs is a known failure source, so edits are string replacements.

**Execution environment.** `AgentRuntime.environment = "container"` makes the runner give the agent an `Environment`
(`environment.py`: exec/read/write/diff; `docker_env.py`: implementation) backed by a container of the **official instance image**,
so tests run in exactly the environment grading uses. Details verified on real images:
- Image git history ends at `base_commit` (+ an empty "SWE-bench" commit): no future commits to find. Patch = diff against the
  tree snapshotted at container start (`git write-tree`), binaries and `__pycache__`/`.pyc`/`.pytest_cache` excluded.
- `--network none`: the agent cannot download the upstream fix (or packages; `pip install` fails by design).
- Each command: `timeout -k` + conda activation (~1.4 s overhead under x86 emulation), same locale setup as the official eval script.
- Infra failures (pull/start/disk) raise `InfraError`: the run stops and leaves no result for that instance, so it is never scored
  as an agent failure. Learned the hard way: Docker Desktop's virtual disk (62.7 GB here) is separate from the host disk and was full.

**Recorded per instance** (`runs/<run>/agent/<id>/`): `trace.json` (every raw LLM response, `reasoning_content`, `finish_reason`,
per-step usage/cost/latency/retries, `n_messages_sent` so each request is reconstructible from the final `messages`, tool calls and
results incl. exit codes, modified files, final diff), `result.json` (usage, latency, `modified_files`, `commands`, `tests` with parsed
pass/fail summaries, descriptive `stats`: edit errors, tests after last edit, repeated identical calls, finish-reason counts...),
`patch.diff`; `results.jsonl` joins the official evaluation outcome. `runs/s0-shakedown` is a one-off plumbing check on a non-dev10
instance and is **not** part of the baseline.

**Fixed development set:** `subsets/dev10.json` (seed-0 random sample of Verified, dataset revision + sha256 pinned; created before any
agent existed). `miracle run --subset dev10` refuses to run if the dataset snapshot no longer matches.
