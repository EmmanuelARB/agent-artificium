# Working on Artificium

Artificium is a **general-purpose** harness for long-running autonomous
agents: a continuous life-loop, self-managed context, durable memory, tool
building, and self-modification. It is not tied to any task, model, or
domain, and every change must keep it that way (see "Never overfit" below).

The user-facing guide is `README.md`; this file is for people and coding
agents changing the harness itself.

Artificium is **not a sandbox**: an instance can run any shell command its
Linux account allows. Never start a real instance on a machine or account
you care about while developing; the offline test suite needs no model and
no network.

## Repository layout

| Path | Contents |
|---|---|
| `artificium.py` | Launcher (`python3 artificium.py <command>`; `--help` lists commands). |
| `app/artificium/` | The Python runtime. Standard library only. |
| `app/prompts/` | The prompt pack ("promptgramming"), declared in `manifest.toml`. |
| `app/seed/` | The starting mind, mapped one-to-one onto `workspace/mind/`. |
| `app/tests/` | Offline tests (`unittest`, no network, no model). |
| `README.md` | User guide. Keep it accurate when behavior or defaults change. |

Created at runtime and ignored by Git: `config.json` (settings),
`.secrets.json` (API key, mode 0600), and `workspace/` (everything the agent
writes). Never commit them, and never read or print `.secrets.json`.

## Architecture map

| Area | Modules |
|---|---|
| Life-loop and turns | `runtime.py` (`Artificium`: `run_forever` → `run_turn` → `_collect_round_inputs` → `_run_round_request` → `_record_reply` → `_execute_tools`), `recovery.py`, `initialization.py` |
| Parsing model output | `life_loop.py` (`parse_life_loop_output`, tolerant tool-call JSON decoding and repair) |
| Model transport | `engine.py`, `engine_stream.py` (SSE streaming, stall watchdog), `engine_adapters.py`, `engine_providers.py`, `connection.py`, `discovery.py`, `vision.py` |
| Context budget | `context_budget.py`, `calibration.py`, `usage.py` |
| Tools | `tools.py` (registry and dispatch), `tool_files.py`, `tool_memory.py`, `tool_interactions.py`, `tool_attention.py` (Infinite Attention), `tool_loader.py` (loads agent-editable tools from `mind/tools/`) |
| Memory and continuity | `memory.py` (long-term memory, working memory, offloading) |
| Interactions and events | `interactions.py`, `events.py` |
| Records and logs | `records.py` (`emit` → lifetime log, `life` → life-loop log, feature logs, model exchanges) |
| Filesystem, seed, versions | `filesystem.py` (`Paths`, atomic writes, seed delivery, file-version backups, shrink guard) |
| Prompts | `prompts.py` (`PromptPack`: `always()`, `event(name, **values)`, overrides) |
| Configuration and setup | `config.py`, `setup.py`, `setup_ui.py` |
| CLI | `cli.py`, `cli_admin.py`, `cli_chat.py`, `cli_service.py`, `cli_watch.py`, `display.py`, `operator.py` |
| Code overlays | `_bootstrap.py`, `__init__.py` |
| Upgrade | `upgrade.py` (guarded `git pull`; moves a clone of the original project to this fork, `REPOSITORY_URL`) |
| Version | `version.py` (the only place the application version is written) |

## Core mechanisms to understand before changing them

- **Prompt pack.** `manifest.toml` lists `always/*` (sent on every model
  request), `tools/core_tools.md` (the tool catalog, also sent every time),
  `runtime/*` templates, and `events/*` (sent only when that event fires).
  Every word in `always/` and `tools/` is paid on every request, for every
  task, for the life of every instance: put detail in event prompts or seed
  memories instead, and measure the word delta of any prompt change.
- **Seed delivery.** `Paths.fill_from_seed` runs on every start. It copies
  each file under `app/seed/` into `workspace/mind/` if the workspace never
  received it, records a sha256 per delivered file in
  `logs/runtime/seed-deliveries.json`, and refreshes a mind file that is
  still byte-identical to what was shipped when the seed changes. A file the
  agent edited is never overwritten; `self.txt` and `meta_memory.md` are
  never refreshed. `__pycache__`/`.pyc` files are never delivered.
  `_EARLIER_SEED_RELEASES` holds hashes of seed files shipped before hashes
  were recorded, so older workspaces can be refreshed too.
- **Overrides.** An instance can shadow any shipped module with
  `workspace/overrides/code/<module>.py` and any prompt with
  `workspace/overrides/prompts/<path>`. An overlay written against an older
  module must keep working after an upgrade as far as reasonably possible:
  `tests/test_overlay_compat.py` imports historical copies from
  `tests/fixtures/overlay_baseline/`. When you add a new name that other
  modules import, resolve it defensively (see the `NEW_SETUP_DEFAULTS`
  `getattr` fallback in `setup.py`). Never edit the baseline fixtures to make
  a test pass; they are historical records.
- **Configuration.** `config.json` has a `harness` and a `model` section;
  model fields are those in `config.MODEL_FIELDS`, and putting a field in the
  wrong section is a load error. Every save writes every field, defaults
  included, so a saved file keeps its meaning when a later release changes a
  default; a key missing from an older, hand-edited, or not yet re-saved file
  means the current default, not "off". Defaults that should apply only to
  new installations (for example streaming) belong in the setup path, never
  silently applied to an existing `config.json`.
- **Tools.** A model-facing tool needs: its handler registered in
  `tools.py` (`self._functions`), a canonical example and accepted arguments
  there, and an entry in `prompts/tools/core_tools.md`. Tool results are
  dicts with a `status` and a `summary`; errors must be actionable (say what
  to do next). Agent-built tools live in `workspace/mind/tools/` and are
  invoked through `run_shell` or loaded by `tool_loader.py`; shipped
  standard-library scripts in `app/seed/tools/` (scheduler, search, fetch)
  reach instances through seed delivery.
- **File safety.** `write_file` and `save_memory` share the modes
  `create|overwrite|append`. Every overwrite backs up the previous content
  under `logs/runtime/file-versions/` (rotated), and an overwrite that would
  shrink a file over 4 KB to under half its size is refused unless
  `allow_shrink` is set. Keep these guarantees when touching file tools.
- **Tool-call robustness.** Calls in one model response run in order; calls
  before the first invalid one execute, and the invalid call and everything
  after it are withheld with a repair notice (mandatory offloading keeps
  whole-response withholding). Parser repairs in `life_loop.py` must stay
  conservative: recover only when the intended call is unambiguous, never
  guess between two readings.
- **Model requests.** `request_timeout_seconds` caps a whole request.
  With streaming on, `stall_timeout_seconds` aborts a stream that stops
  delivering data once it has started, and `first_token_timeout_seconds`
  (off by default) bounds the silent prompt-processing phase. A server that
  rejects streaming falls back to plain JSON for the session.
- **Logs** (in `workspace/logs/`): `lifetime.jsonl` (operational records via
  `Records.emit`), `life-loop.jsonl` (the life-loop trace via
  `Records.life`), `features/*.jsonl` and `summary.json`, `model/` (gzipped
  request/response exchanges), `context/` (offload checkpoints), `runtime/`
  (state, queues, scheduler, seed deliveries, file versions). The lifetime
  log is authoritative.

## Development

```bash
cd app && python3 -m unittest discover -s tests          # full suite, about a minute
cd app && python3 -m unittest tests.test_file_safety     # one module
```

- **Standard library only.** No third-party imports in `app/`, including
  seed tools.
- **Run Python from the repository root or `app/`, never from inside
  `app/artificium/`**: `app/artificium/operator.py` then shadows the
  standard-library `operator` module and imports fail.
- **Tests that import seed scripts** create `app/seed/tools/__pycache__/`.
  Seed delivery ignores it, but delete it before packaging or committing.
- Some tests size a context window against the real prompt pack (for example
  `fill_context` in `test_final_release.py`, the notice-backlog test in
  `test_revolution.py`). When prompts grow, adjust those fixtures in the
  spirit of the test rather than weakening the assertion.
- Add a test with every behavior change, next to the tests that already
  cover that area. Run the full suite before committing; it must be green.
- Match the surrounding code: its comment density, naming, and idiom.
  Comments explain why, not what.

## Checklists for common changes

- **Prompt change:** keep additions to `always/` and `tools/` minimal and
  measured; check tests that assert prompt text; bump the version (below).
- **New or changed tool:** handler + registry + canonical example + accepted
  arguments in `tools.py`, entry in `core_tools.md`, tests, README "Agent
  tools" table if user-visible.
- **Seed change:** edit files under `app/seed/`; unedited copies in existing
  minds refresh automatically; `tests/test_release.py` counts shipped seed
  memory files, so update it when adding one; document new seed tools in a
  seed memory under `app/seed/memory/tools/`.
- **New config field:** dataclass default and parser in `config.py`,
  `MODEL_FIELDS` if it belongs to the model section, setup and CLI plumbing
  (`setup.py`, `setup_ui.py`, `cli.py`, `cli_admin.py`), tests, README.

## Versioning

The application version (`RELEASE` in `app/artificium/version.py`) and the
prompt pack version (`version` in `app/prompts/manifest.toml`, of the form
`Artificium-<codename>-<release>`) are kept **in step**: any release bumps
both to the same number. Bump for every change to prompts or harness
behavior that instances would run with, so runs can be told apart in their
logs (`prompt_pack_loaded` records the pack version). Tests that assert the
version or User-Agent string must be updated with it; the overlay baseline
fixtures keep their historical versions.

## Git

- Committing directly to `main` is fine; keep the test suite green at every
  commit.
- Never commit `config.json`, `.secrets.json`, `workspace/`, or
  `__pycache__/`.
- Write commit messages that say what changed and why.

## Evaluating instances

Improvements come from watching real instances, and from reading their logs
critically:

- Compare like with like: the same task prompt, the same model, and windows
  of equal length. Early hours can look very different from day two.
- Know the log fields before counting: in `lifetime.jsonl`,
  `tool_executed.result.status` values such as `written`, `remembered`, or
  `reflection_required` are successes; only `failed` and `error` are
  failures. `reflection_required` on `sleep` and `offload_working_memory` is
  a deliberate two-step (the first call delivers the reflection prompt).
- Judge outputs, not only process: measure what the agent produced against
  what was asked (volume, quality, duplication, calibration of claims).
- Verify any number produced by an automated analysis before acting on it.

## Never overfit

Evaluation runs are samples used to find weaknesses. They are not targets.
Learn from them, but every change must make the harness better for tasks
nobody has run yet.

- Fix the general mechanism behind a failure, not the task where it showed
  up. Ask: "would this change be right for a completely different task?" If it
  only makes sense for the observed run, it does not belong in the harness.
- Keep run-specific vocabulary out of prompts, seed memories, and code: no
  motifs, names, domains, file layouts, or examples lifted from a test run.
  Use neutral, domain-agnostic wording and examples.
- Prefer mechanisms (tools, guards, measurements, feedback the agent sees)
  over ever more specific instructions.
- Watch for overcorrection: a rule written to stop one failure can create the
  opposite one elsewhere (pushing for pace produced filler; pushing for
  caution can produce stalling). State the balance, not only the fix.
- Judge a change on more than the run that motivated it; when in doubt,
  leave the harness as it is and note the observation instead.
