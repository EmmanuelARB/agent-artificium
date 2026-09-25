# Artificium

**A general agent harness for long term autonomous work, continual learning, and self-improvement.**

> [!NOTE]
> **This is a fork.** Artificium was created by [gr](https://gr.bio/), who built it as their own harness and published it at [officialgr/agent-artificium](https://github.com/officialgr/agent-artificium). The design and the original code are theirs. This fork is maintained by [EmmanuelARB](https://github.com/EmmanuelARB) and is not affiliated with or endorsed by the original author.

## Why this fork

This fork develops Artificium by running real instances on long, open-ended tasks, reading their logs critically, and fixing the general mechanism behind each weakness found, never the task where it showed up. The harness stays general-purpose: no change is made for one task, one model, or one domain. Releases from 1.10.1 on are published here and differ from the original project. The main additions so far:

- **One `app/` tree** for the runtime, prompt pack, seed, and tests, with a contributor guide in `AGENTS.md`.
- **File safety:** every overwrite by `write_file` or `save_memory` keeps a rotated backup, and an overwrite that would shrink a large file to under half its size is refused unless explicitly allowed.
- **Tool-call robustness:** valid calls before a malformed one still run; the rest are withheld with a repair notice.
- **Web search and fetch** as shipped standard-library tools.
- **Overrides that survive upgrades:** code overlays written against an older release keep working, checked against historical copies in the test suite.
- **Token accounting:** calibrated token counts, per-request and per-turn usage, and cache-miss warnings in `watch`.
- **Explicit configuration:** `config.json` lists every setting, defaults included, so its meaning does not change when a later release changes a default.
- **`watch` history** counts only records it displays, so a long wait for the model no longer hides it.
- **`upgrade`** moves a clone of the original project to this fork.

## What Artificium is

Artificium gives an agent full control over its own environment, lets it work indefinitely with or without outside interaction, and keeps its experience for later retrieval. The agent manages its own context window: it decides what stays loaded, what goes to long-term memory, and what to retrieve or revisit. Along that continuous life-loop it can build tools, revise its Self, and improve its methods, while its whole history stays available for learning and its active context stays on the work at hand.

[Quick start](#quick-start) · [Architecture and features](#architecture-and-features) · [Complete guide](#complete-guide) · [Future improvements](#future-improvements)

## Quick start

> [!WARNING]
> **Artificium is NOT A SAFE PRODUCT.**
>
> There is no built-in sandbox or human approval layer. The agent can run shell commands and access, modify, or delete anything available to its Linux account, including its own code and API key.
>
> **Run it in an isolated environment.** Prefer a disposable cloud instance, a VM with no host filesystem access, or a properly sandboxed container. Do not run it directly on your personal or work computer, or expose host drives, your home directory, the Docker socket, an SSH agent, or personal credentials.
>
> **Prefer a local model.** For a paid API, use a dedicated key and set a hard spending cap with a provider that can enforce it for that key or its dedicated project/account. Artificium can read and use its own key directly; a limit written in its prompt or editable configuration cannot contain that spending. Budget alerts alone do not stop charges. Local inference still requires the same isolation.

**Requirements:** Linux, Python 3.11+, and a local or remote model API. The runtime uses only the Python standard library; no `pip install` is needed.

### Install

Clone the repository and run it in place:

```bash
git clone https://github.com/EmmanuelARB/agent-artificium.git
cd agent-artificium
python3 artificium.py
```

```text
agent-artificium/
├── artificium.py      launcher
├── README.md
├── app/               code, prompts, starting mind (seed), tests
│
│   created by the first run, ignored by Git:
├── config.json        model connection and harness settings
├── .secrets.json      saved API key, mode 0600
└── workspace/         the agent's mind, logs, and self-modifications
```

Everything except `config.json`, `.secrets.json`, and `workspace/` stays unchanged at runtime. To start over with a fresh agent, stop it and delete `workspace/` (or run `reset`): the next start grows a new mind from `app/seed/` and keeps the model connection. One clone is one agent; for two agents, clone twice.

### Setup

The wizard asks for harness preferences (vision, working-memory target, mandatory offloading, automatic repair), then your model service, address or key, and model. It detects context capacity where possible and leaves reasoning and sampling to the server unless you customize them.

Setup verifies the connection with real, potentially billable model requests before saving. It checks the full harness prompt and, where applicable, image input, without executing tools or keeping the diagnostic response.

A fresh setup uses these recommended defaults, all changeable later:

| Setting | Fresh-setup default |
|---|---|
| Working-memory target | `auto`: 60% of the model's serving context |
| Mandatory offloading | On, at 80% of the working-memory target |
| Automatic repair | On, up to 3 attempts per incident |
| Model request timeout | 1,800 seconds |

An existing `config.json` keeps the values it already has.

### Run

**Chat with the agent.** This starts or reuses the background life-loop:

```bash
python3 artificium.py chat
```

*Tip: for a graphical chat, ask the agent to build a browser-based chat UI and send you the link.*

**Watch the life-loop.** In another terminal, follow its thoughts, tool use, token usage, and state changes:

```bash
python3 artificium.py watch
```

`watch` attaches without starting the agent. To run the life-loop without opening chat, use `python3 artificium.py start`.

**Closing chat or watch leaves the agent running.** To stop it:

```bash
python3 artificium.py stop
```

Supported connections: **llama.cpp, Ollama, vLLM, OpenRouter, OpenAI Responses, Gemini, Anthropic, OpenAI-compatible servers, and custom JSON APIs**. Local inference and custom setups are covered in the [complete guide](#complete-guide).

<details>
<summary><strong>Try Artificium on Runpod: setup and example prompts</strong></summary>

I usually chat with Artificium in one browser tab and keep its life-loop open in another.

**[Open the Runpod template](https://console.runpod.io/hub/template/x4bydgdt2h?ref=8knnycbq)**

The template downloads Artificium and the model, starts llama.cpp, and opens interactive setup in a browser terminal.

I recommend using an **RTX 3090 with 24 GB VRAM** and **Qwen3.8-27B UD-Q4_K_XL**. This is the configuration I use most often.

### Getting started

1. Before deploying, change the `ARTIFICIUM_WEB_PASSWORD` environment variable to a **strong, unique password**.
2. Open the Pod's **Connect → HTTP port 7860** link. Sign in with username **`artificium`** and your chosen password.
3. Wait for the model to download and load, then complete Artificium's interactive setup.
4. Choose terminal chat. Open the same terminal link in another tab to watch the life-loop. From the project folder, you can also run:

   ```bash
   python3 artificium.py watch
   ```

HTTP port **7861** is free for the agent to use. You can expose more HTTP ports if needed.

I **strongly recommend setting the temperature to 0.3** for this **Qwen3.8-27B configuration**. If you forgot to change it during Artificium's setup, you can easily change it afterward by running this from the project folder:

```bash
python3 artificium.py configure model --temperature 0.3 --yes && python3 artificium.py restart
```

Closing the browser tabs leaves Artificium running. Use `python3 artificium.py stop` to stop the agent, and stop the Pod separately in Runpod when you finish using its GPU.

### Example: give it a task

```text
Build me a web chat UI where I can talk to you, and send me a link I can open in my browser.
```

### Example: give it a long-term purpose

Ask Artificium to update its Self with an ongoing goal:

```text
Change your Self so your life purpose is to solve the Riemann hypothesis. Keep working on it autonomously until you solve it.
```

</details>

## Architecture and features

Each instance has **one model connection, one active inference loop, and one shared mind**. The runtime supplies context and executes tools; the model decides what to inspect, remember, build, pursue, or defer.

### Environment and agency

<details>
<summary><strong>1. Linux as the agent's body</strong></summary>

The agent acts through a Linux environment: it inspects files, executes commands, uses installed software, and builds programs. Its capabilities grow as it creates tools inside the same environment where it works.

The harness tells it how this arrangement works: operating instructions, filesystem conventions, available tools, current state, Self, and a map of its memory. Everything else it learns about the environment comes from inspection and experience.

Reusable apparatus lives in `mind/tools/`; projects and work products live in `mind/space/`. Operating knowledge belongs in memory, so a capability the agent builds stays discoverable after its original working context is offloaded. Actual access is defined by the Linux account's permissions.

</details>

<details>
<summary><strong>2. Model connections and textual tools</strong></summary>

The harness owns continuity on disk. Model adapters translate its requests into the selected API's format, so connecting a different model preserves Self, memory, interactions, and work products.

Tools use a small textual protocol inside ordinary model output:

```text
<tool_call>{"tool":"read_file","path":"mind/self.txt"}</tool_call>
```

Provider-native function calling is not needed. Each request is read by decoding what follows its opening tag, so the common slips of real models are tolerated when their meaning is unambiguous: a missing or repeated tag, raw newlines, lone backslashes, or unescaped quotes from code inside strings, a tag quoted inside an argument, a draft call inside a thought, or Qwen's native `<function=NAME><parameter=KEY>` form. A repair is kept only when the repaired request ends exactly where the request does; ambiguous cases, and a request left unclosed by a reply cut off at the output limit, are still rejected. Before executing a response's batch of calls, the runtime checks tool names, JSON, and argument signatures. A protocol error withholds the whole batch and returns repair guidance. Accepted calls run in order; an operational error does not undo earlier actions. A successful offload, Self revision, or sleep ends the batch so later work sees the changed state.

This keeps interaction and memory consistent across backends. Reasoning controls, vision, capacity, and task performance still depend on the model and server.

</details>

<details>
<summary><strong>3. Promptgramming</strong></summary>

**Promptgramming** expresses the agent's operating model as inspectable instructions. Files in `app/prompts/` define the environment, tool protocol, memory lifecycle, interaction rules, and the meaning of runtime events; Python implements the matching mechanics.

Stable operating contracts go into every inference. Event and guidance prompts add context when something happens: an input arrives, a tool request needs repair, memory needs organizing.

Strategies for using these mechanisms live in editable operating memories under `mind/memory/harness/`, so the agent can refine its methods through experience while keeping the same interfaces. It can also change the prompts and code themselves, through overrides (see [feature 16](#architecture-and-features)).

</details>

<details>
<summary><strong>4. A dynamic Self</strong></summary>

`mind/self.txt` describes lasting identity, purpose, priorities, and initiative. It is part of every inference, so it survives working-memory offloads and restarts.

Self can describe an agent that responds and sleeps, one that explores a subject, or one dedicated to a continuing task. A research goal can live there for the life of the instance: the agent keeps investigating, recording results, and resuming work without an open chat or anyone to reply to.

`revise_self` uses two stages: request reflection, then confirm a complete replacement. The previous Self is archived. Detailed knowledge belongs in long-term memory; Self stays focused on what guides activity. It is intentionally mutable, so a standing goal remains something the agent can reconsider.

</details>

<details>
<summary><strong>5. A continuous life-loop</strong></summary>

The life-loop gives the agent opportunities to act on startup, on incoming events, and on unfinished work. A completed generation with no tool calls automatically continues unless the agent is sleeping or the runtime is stopped or paused for error handling. Continuation has no interval and never interrupts generation or tool execution. Continuous work can span any number of inferences and tool calls, with no round-count or elapsed-time limit.

When no message needs attention, Self supplies direction. The agent can continue a project, investigate, build a tool, organize memory, or sleep. The `sleep` tool asks for reflection first, then waits until an event or a timer wakes it.

Sleep uses no inference tokens. The process must keep running to notice wake conditions, and continued work requires an available model backend and compute budget. The scheduler can arrange a future wake; offloading preserves a useful continuation for the next phase of work.

</details>

### Memory and learning

<details>
<summary><strong>6. Long-term memory as files and folders</strong></summary>

Knowledge lives under `mind/memory/`: facts, methods, decisions, failed approaches, entity and project history, and continuation checkpoints. Files suit an agent that already works through Linux: it can inspect, search, reorganize, and reuse them with the same tools it uses everywhere else.

The agent chooses descriptive filenames and semantic folders. Retrieval uses indexes, text search, direct reads, or Infinite Attention for larger sources. There is no vector database or separate skill store.

`save_memory` writes the supplied content and returns organization guidance. The agent is responsible for putting useful evidence and retrieval conditions into that content, and for maintaining the indexes that make it discoverable. Later retrieval brings the knowledge back into working context, where it can be tested, corrected, or extended.

</details>

<details>
<summary><strong>7. Meta-memory and retrieval indexes</strong></summary>

`mind/meta_memory.md` is the root map of the mind. It holds a small amount of essential knowledge plus routes to deeper memory and important tools. The complete file is part of every inference.

Folder-level `index.txt` files explain what their memories contain, how they relate, and when to retrieve them. The agent follows them from broad subjects down to specific evidence. As memory grows, the root points to meaningful branches instead of listing every file.

The agent authors this navigation; saving a memory does not update it automatically. A tool created in `mind/tools/` also needs an entry explaining what it does and how to use it. Keeping the root concise matters because it costs context on every request; past its 8,000-token guidance threshold, the harness reminds the agent to reorganize it.

</details>

<details>
<summary><strong>8. Working memory and offloading</strong></summary>

**Working memory** is the context the agent actually works in. Every request combines recent history and observations with the operating contracts, Self, meta-memory, tool catalog, runtime state, and active images.

The **working-memory target** is a budget the harness sets for that context, at or below the model's real serving capacity. It drives the context percentage the agent sees, the mandatory-offload threshold, and Infinite Attention chunk sizes. By default it is 60% of the serving context. For example, with a 200,000-token model, the target is 120,000 tokens and mandatory offloading triggers around 96,000.

Why not use the whole context?

- **Room to offload.** Offloading happens while the full detail is still loaded: the agent reflects, saves memories, and writes a checkpoint, which all takes tokens. A threshold at the edge of the real context leaves no room to do it well.
- **Room for surprises.** A single large tool result can add tens of thousands of tokens in one step.
- **Room to answer.** Reasoning and output must also fit in the serving context.
- **Focus and cost.** Resolved debugging and stale tool output interfere with reasoning, and every loaded token is paid again on every request.

Nothing is lost by aiming lower: offloaded history is archived, not deleted.

**Offloading keeps the learning and the continuation, then releases the detail:**

1. The agent calls `offload_working_memory` and receives reflection guidance while the details are still present.
2. It saves reusable lessons, updates retrieval paths, and writes a checkpoint with active goals, findings, evidence, obligations, and next actions.
3. It confirms the offload. The runtime saves the checkpoint in `mind/memory/`, archives the old working history, and replaces it with the checkpoint.
4. Work continues with a freshly rebuilt context. Active images are released; their source files remain.

The model decides what the checkpoint preserves. Archived context and durable interactions are there to recover anything it left out.

**Mandatory offloading**, on by default for new setups, adds a runtime gate at 80% of the target. At the threshold, or when generation needs the remaining context, ordinary actions and sleep are withheld until an offload succeeds. Memory preparation stays available, and the requirement survives restart. Context is never silently truncated.

**Automatic repair**, also on by default for new setups, recovers from a rejected input or an empty answer by retrying from an earlier successful context, with up to three attempts per incident (see [Configuration](#complete-guide)).

</details>

<details>
<summary><strong>9. Infinite Attention</strong></summary>

Infinite Attention reads text sources larger than one context window. The agent works through a file or directory in bounded chunks while carrying forward an understanding specific to its objective.

1. Open a stream with a source and an objective.
2. Read a chunk with the current carry: findings, evidence locations, open questions, and what to examine next.
3. Checkpoint the chunk with an updated carry. The runtime removes consumed stream material from working history.
4. Advance to the next chunk, then repeat or save the final result.

The objective, source manifest, cursor, chunk number, carry, and status persist on disk. A chunk must be checkpointed before the next one is delivered; asking early returns the pending chunk again. Streams can pause and resume, read coarsely or finely, and refine byte ranges of a single-file source.

This connects large-source analysis to memory: the agent can investigate a corpus, recover a fact from archived context, pause for an interaction, then resume from the saved cursor and carry.

Each inference is still finite, and the carry is lossy: exact claims need checking against source passages. Keep source files unchanged during a pass, since the manifest records initial sizes, not copies. Binary sources need conversion first.

</details>

<details>
<summary><strong>10. Continual learning and self-improvement</strong></summary>

Learning happens through a cycle of experience, verification, memory, and reuse. A solved problem can become a procedure; a failure can become a condition to recognize; a recurring need can become a program.

The operating guides ask the agent to test a method, record evidence and limitations, and check whether it transfers to later work. Improvements can live in ordinary memory, a harness operating guide, a tool, Self, or a prompt or code override. Offloading is a natural point to extract lessons, and meta-memory is the route back to them.

This is adaptation through stored knowledge, instructions, and executable tools. Model weights stay unchanged. The harness provides the mechanisms; the model authors the changes, and whether they help has to be established through evaluation. Fine-tuning from accumulated memory is one of the [future directions](#future-improvements).

</details>

### Inputs, tools, and continuity

<details>
<summary><strong>11. Non-blocking interactions</strong></summary>

People, applications, sensors, scheduled tasks, and external agents all use the same durable event format: sender, recipient, interaction ID, timestamp, content, attachments, and an optional reply target.

Publishing an input does not wait for an answer. A compact notification tells the agent where to look; reading the body and handling the request are separate decisions. The agent can prioritize several inputs, postpone one, combine related work, or continue its standing objective. Delivery, reading, and handling each have their own receipt.

New input can arrive while inference or a tool is running. It reaches the agent at the next inference boundary; it does not preempt the current request or run in parallel with the shell.

Replies are outbound events created with `send_interaction`; ordinary model text stays in the internal trace. A client or bridge delivers replies to their destination. All interactions within one instance share its mind. Built-in spawning and swarm coordination are future work.

</details>

<details>
<summary><strong>12. Deliberate perception and image context</strong></summary>

Attachments are copied into durable storage before their event is published. Their paths appear in event metadata; their contents do not enter model context on their own. The agent decides what to read, view, or convert.

For a vision-capable model, `load_images` creates an explicit visual working set. Images can be kept for one successful inference or across several. `release_images` removes them from model input without deleting their files. One-shot images survive a failed inference; offloading releases the whole active set.

Text attachments use file reads or Infinite Attention. Audio, video, PDFs, and other binaries need conversion apparatus such as transcription, frame extraction, or text extraction. The agent can keep observations and original paths in memory, then reload the source when exact reinspection matters.

</details>

<details>
<summary><strong>13. Reusable tools and an evolving workspace</strong></summary>

The agent can turn a missing capability into a program under `mind/tools/`, test it, and record how to use it. Projects, experiments, and user-facing work products belong in `mind/space/`.

Custom apparatus runs through the shell. Adding a file does not register a native harness tool; meta-memory and operating memories tell the agent that the program exists, when it helps, and how to call it.

A fresh mind ships with two such tools: the scheduler (feature 14) and `mind/tools/search.py`, a standard-library web search that tries Brave, then Wikipedia, then DuckDuckGo, and says which engine answered or why none did. Pages found this way can be read with an installed text browser such as `lynx -dump URL`.

These primitives compose into workflows: a sensor publishes observations as interactions; a converter makes an attachment readable; an analyzer produces evidence for a memory; a client connects another agent or service. Browsers, remote UIs, sensors, and service bridges are things to build or install, not bundled integrations.

</details>

<details>
<summary><strong>14. Persistent scheduling</strong></summary>

The scheduler is editable apparatus in `mind/tools/scheduler.py`, loaded at startup and supervised by the runtime. It stores one-time or recurring tasks and publishes due work as `scheduled_task` interaction events.

Polling continues while inference is busy. Waiting costs no model calls, and a due event wakes a sleeping agent. The task text carries the instructions, relevant memory or artifact paths, and where results belong; the same life-loop decides how to carry it out.

Recurrence uses intervals in seconds. A task marked `completed` has emitted its event; the work itself may still be pending. Scheduler code changes take effect after restart.

</details>

<details>
<summary><strong>15. Durable state, logs, and recovery</strong></summary>

The instance persists working history, interactions, attention streams, scheduling and sleep state, and pending control requirements. Logs keep model requests and responses, tool activity, feature events, and archived contexts. The 50 most recent model exchanges stay as plain JSON; older ones are gzip-compressed, which cuts their size roughly tenfold. `status`, `watch`, and `logs` expose this state to the operator.

Notification delivery uses a claim-and-commit process around inference. Failed requests release claimed notifications for retry; startup recovers interrupted claims and reconciles inbound events that lack receipts. External side effects still need care around retries.

The application ships Self, meta-memory, ten operating memories, the scheduler, and the search tool as a **seed** in `app/seed/`, not as a live mind. Each seed file is delivered to a workspace once, and deliveries are recorded in `workspace/logs/runtime/seed-deliveries.json`. Three things follow:

- A workspace with no record is built whole, so a deleted workspace comes back factory-fresh.
- A memory the agent has revised or discarded stays as the agent left it.
- A memory added by a newer application version is delivered, because this workspace never received it.

An existing file is never overwritten. `self.txt` and `meta_memory.md` are the exception: the harness cannot start without them, so a missing one is restored. The first wake orients the agent to its environment.

The mind holds selected learning; the logs hold raw evidence and resumable state. Back up both.

</details>

<details>
<summary><strong>16. An immutable application, changed through overrides</strong></summary>

The application does not change at runtime. Everything the agent writes (mind, logs, and self-modifications) lives in the workspace next to it, so upgrading or resetting the application never touches what the agent has learned.

The agent can still change its own operating prompts and code, through overlays instead of in-place edits:

- `workspace/overrides/prompts/<path>` shadows `app/prompts/<path>`, using the same relative path as the prompt manifest.
- `workspace/overrides/code/<module>.py` shadows `app/artificium/<module>.py` as a complete module replacement, not a patch.

Both take effect at the next start. Every code override is syntax-checked first; if any fails to parse, the whole code overlay is skipped for that start instead of blocking it. After three consecutive failed starts, the overlay moves to `workspace/overrides.quarantined/` automatically. The agent can revise its own harness while a known-good application stays underneath.

The file tools refuse writes anywhere in the project outside `workspace/`, `config.json`, and `.secrets.json`, and point the agent at the overlay instead. That is a guardrail, not a security boundary: `run_shell` is unrestricted. The only hardening that actually holds is making the project files read-only to the Linux account that runs the agent, leaving only those three paths writable.

</details>

<details>
<summary><strong>17. Prompt caching and token accounting</strong></summary>

Every request resends the whole working context, so the harness keeps its layout cache-friendly:

- **Stable prefix.** The tool catalog comes before the pinned Self and meta-memory in the system prompt, and the per-request state header is a separate message.
- **Pinned-mind snapshot.** Self and meta-memory stay frozen in the system prompt between context rebuilds (process start, a successful offload, or an Infinite Attention compression). An edit to either file reaches the agent at once as a change notice, and the pinned copy catches up at the next rebuild, so an ordinary edit does not invalidate the provider cache for the whole context. Set `harness.pinned_mind_snapshot` to `false` to read them live on every request instead.
- **Cache markers.** For Anthropic, the harness marks cache breakpoints on the system prompt, the end of persisted history, and pending input (`model.prompt_cache`, on by default). Providers that cache automatically ignore it. OpenAI Responses and OpenAI-compatible servers can also receive a `prompt_cache_key` routing hint: set `model.prompt_cache_key` to `"auto"` or a string. It is off by default because some compatible servers reject it.

Token usage from every provider is normalized into one reading. `watch` shows input, cached, and output tokens and generation speed for each request, a running total per turn, and a warning when a large request misses the cache, with a likely cause: expiry after a long idle gap, or re-routing by the backend.

Where a provider cannot count tokens before sending, the harness estimates from characters and **calibrates** that estimate against the real input counts reported afterwards. The ratio is persisted per provider and model and reset when the model changes. Once calibrated, a request that is comfortably within budget and has no images can skip the provider's preflight count.

</details>

## Complete guide

Setup, configuration, operation, integrations, and maintenance.

<details>
<summary><strong>Model connections: local inference, credentials, and custom APIs</strong></summary>

### Local inference

Run the model server inside the isolated environment, or use a remote inference service you intend it to reach. WSL2 works, but its usual shared Windows drives do not isolate it from your main computer.

For an installed llama.cpp server:

```bash
llama-server -m /path/to/model.gguf --alias local-model --ctx-size 32768 --parallel 1 --port 8080
```

In a second terminal, from the project folder:

```bash
python3 artificium.py setup --provider llamacpp --no-launch
python3 artificium.py chat
```

A single served model is selected automatically, and no placeholder API key is needed. `--url` accepts a server root, a `/v1` base, or a complete chat endpoint. Choose a model and context allocation that fit your hardware: Artificium neither installs the model server nor enlarges its allocation. A local vision model also needs the server's vision/projector configuration.

| Provider flag | Endpoint | Default local address |
|---|---|---|
| `llamacpp` | `/v1/chat/completions` | `http://127.0.0.1:8080/v1` |
| `ollama` | Native `/api/chat` | `http://127.0.0.1:11434` |
| `vllm` | `/v1/chat/completions` | `http://127.0.0.1:8000/v1` |
| `openrouter` | `/api/v1/chat/completions` | — |
| `openai` | `/v1/responses` | — |
| `gemini` | `models/{model}:generateContent` | — |
| `anthropic` | `/v1/messages` | — |
| `custom` | Your OpenAI-compatible endpoint, or a custom JSON contract | Supplied by you |

For another OpenAI-compatible service:

```bash
python3 artificium.py setup --provider custom --url http://MODEL_SERVER:8000/v1 --model MODEL_ID --context-window 32768 --no-launch
```

For unattended setup, add `--yes` and supply every required value and credential; setup still verifies inference. Use `--model ID` when discovery finds several models. If capacity cannot be discovered, pass `--context-window TOKENS` with the real serving limit. Interactive setup asks once; unattended setup otherwise falls back to a labelled 32,768-token budget. A successful test does not verify an unknown limit in full.

Context discovery uses the server's real allocation where available: llama.cpp's serving `n_ctx`, vLLM's `max_model_len`, or provider metadata. A GGUF's training capacity is not what llama.cpp serves. Ollama receives the configured window as `options.num_ctx` (32,768 by default, or a lower advertised maximum); a smaller slot already loaded does not cap it. Detected limits refresh on reconnection; manual limits persist.

For llama.cpp, discovery reads `/v1/models` and `/props?model=ID`: `default_generation_settings.n_ctx` gives the serving capacity and `modalities.vision` the image support. Its optional `/apply-template` and `/tokenize` endpoints give exact prompt counts before inference. The adapter maps repetition penalty to `repeat_penalty`.

### Credentials and connection checks

Use the masked setup prompt, `--api-key-file /path/to/key`, `ARTIFICIUM_API_KEY`, or the provider variable: `LLAMA_API_KEY`, `OLLAMA_API_KEY`, `VLLM_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`. Avoid typing a secret directly into a shell command.

Saved keys live in `.secrets.json` in the project folder (mode 0600). Git ignores it and it sits outside `workspace/`, so it survives upgrades and resets. An explicitly saved key takes precedence, so runtime uses the key that passed verification. Otherwise `ARTIFICIUM_API_KEY` comes before the provider variable and the saved fallback. Saved keys are bound to provider and address. `key` verifies a replacement before saving it, and a running agent reloads it. The agent can read its key; apply the spending controls from Quick start.

`doctor` shows the adapter, exact URL, safe header names, and request parameters without inference. `check` (same as `doctor --live`) sends a real diagnostic request with the complete prompt pack, pinned mind, and saved working context, plus an image check where applicable. It does not start the agent, import mutable tools, consume pending events, or keep the response. Setup and reconnection use the same check before saving.

### Custom JSON APIs

For a service that is not OpenAI-compatible, describe its API in a contract file such as `my-engine.json`: a complete URL, optional static headers, authentication mapping, a JSON request body, and JSON Pointer paths into the response. For example:

```json
{
  "url": "https://api.example.com/v2/generate",
  "headers": {"X-API-Version": "2026-08-29"},
  "auth": {"header": "X-API-Key", "prefix": "", "required": true},
  "body": {
    "model_name": "$artificium.model",
    "conversation": "$artificium.messages",
    "generation": {
      "temperature": "$artificium.temperature",
      "max_new_tokens": "$artificium.max_output_tokens",
      "reasoning_level": "$artificium.reasoning_effort"
    }
  },
  "response": {
    "content": "/result/text",
    "reasoning": "/result/reasoning",
    "usage": "/usage",
    "finish_reason": "/result/status"
  }
}
```

Then connect with it:

```bash
python3 artificium.py setup --provider custom --model MODEL_ID --custom-contract my-engine.json --context-window 64000 --no-launch
```

Keep the actual key in the credential system. `auth: false` means no authentication; otherwise a key defaults to bearer authentication unless `header` and `prefix` override it.

Exact `$artificium.NAME` strings keep the substituted JSON type. Available names: `model`, `messages`, `prompt`, `system`, `last_user`, `context_window_tokens`, `reasoning_effort`, `reasoning_budget_tokens`, `reasoning_mode`, `temperature`, `max_output_tokens`, `top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, and `stop_sequences`. Unset optional values are omitted.

`response.content` is required; `reasoning`, `usage`, and `finish_reason` are optional. A list of paths selects the first that exists. The contract covers one non-streaming JSON POST with an object-shaped JSON response; multipart, streaming, websockets, and executable adapters are out of scope.

To return to ordinary OpenAI compatibility:

```bash
python3 artificium.py configure model --custom-contract none --url http://localhost:8000/v1 --context-window 32768
```

</details>

<details>
<summary><strong>Configuration: context, offloading, vision, and generation controls</strong></summary>

Settings live in `config.json` in the project folder (ignored by Git), saved atomically, with a `harness` and a `model` section. Every save writes every setting, defaults included, so a saved file keeps its meaning when a later release changes a default; a setting missing from the file takes the current default. Inspect them with `config`, `config harness`, or `config model`. Change harness preferences offline with `configure harness`; use `connect` or `configure model` for a verified model change. Configuration changes take effect after `restart`; a replaced API key reloads on its own.

```bash
python3 artificium.py configure harness --vision auto
python3 artificium.py configure harness --mandatory-offload on --offload-threshold 80
python3 artificium.py configure harness --working-memory-tokens auto --auto-repair on
python3 artificium.py connect --reasoning auto
python3 artificium.py restart
```

| Setting | Meaning |
|---|---|
| `--vision auto`, `yes`, or `no` | Image preference, reconciled with discovered and tested model capabilities. |
| `--working-memory-tokens auto`, `same`, or `TOKENS` | Working-memory target. `auto` (fresh-setup default) is 60% of the serving context; `same` follows the whole context; a number must be at least 4,000 and fit inside the context. |
| `--mandatory-offload on` or `off` | Require a successful offload at the threshold. On for fresh setups. |
| `--offload-threshold PERCENT` | Threshold as a percentage of the working-memory target, 1–95; default 80. |
| `--auto-repair on` or `off` | Retry from an earlier context after an input failure, up to three attempts. On for fresh setups. |
| `--context-window TOKENS` | Actual serving capacity; changing this number cannot enlarge a server. |
| `--request-timeout SECONDS` or `off` | Inference timeout, 1–86,400 seconds. 1,800 for fresh setups; `off` waits indefinitely. Shell timeouts are separate. |

Older configurations keep the values they were saved with, including a missing timeout (off) or disabled offloading and repair. The removed `heartbeat_seconds` setting is ignored and dropped on the next save.

A few advanced settings have no CLI flag. Edit them in `config.json`, then restart:

| Key | Default | Meaning |
|---|---|---|
| `model.stream_responses` | `false` | Stream replies from llama.cpp, vLLM, OpenRouter, and OpenAI-compatible servers, so `watch` shows progress during long generations. |
| `model.prompt_cache` | `true` | Anthropic cache breakpoints (see feature 17). |
| `model.prompt_cache_key` | `null` | `"auto"` or a string sends a `prompt_cache_key` to OpenAI Responses and OpenAI-compatible servers. |
| `harness.pinned_mind_snapshot` | `true` | Freeze Self and meta-memory between context rebuilds for cacheability. |
| `model.request_options` | `{}` | Extra request fields verified against your server, such as `chat_template_kwargs`. They cannot replace protected prompt/model fields or conflict with normalized controls. Never put credentials here. |

### Context budget and offloading

Mandatory offloading is evaluated at inference boundaries. Reaching the threshold withholds ordinary tool actions and sleep until the two-stage offload succeeds; memory preparation stays available, and the requirement survives restart.

Token counts come from the provider where possible: llama.cpp's native input-count endpoint (older builds render and tokenize text-only requests), and the counting APIs of vLLM, OpenAI Responses, Anthropic, and Gemini. Otherwise the harness uses a calibrated character estimate (see feature 17). Counts are labelled `provider`, `calibrated`, or `estimate` in the life-loop.

The input check leaves a counting margin (1% for provider counts, 3% for calibrated estimates, 5% for plain estimates, at least 256 tokens) plus room for any explicitly configured output or reasoning budget. **It does not rewrite generation settings or impose an output cap.** Unset optional settings stay with the provider. Anthropic requires an output maximum: the harness uses the model's reported maximum, or `min(context, 32,000)` when that is unknown. Reasoning and the answer still have to fit the serving context.

A large tool result can still cross the threshold in one step. A threshold below the size of the pinned prompt, or a checkpoint that barely shrinks the context, can demand offloading repeatedly. Keep the working-memory target large enough for the full harness prompt with room to spare.

### Automatic repair

With automatic repair on, context errors, tokenization or template errors, image errors, and empty or malformed responses trigger recovery. The harness archives the working history and retries from the context before an earlier successful response, with a short explanation to the agent. A second failure goes further back. For a context error, the last attempt uses an isolated summary helper, sooner if no earlier request is left. Three attempts are allowed per incident, across restarts, until a normal request succeeds. Authentication, network, quota, and configuration failures are handled separately.

The helper uses the same model, API, and serving context, with its own summary prompt, no tools, no Self, and server-default reasoning. Its input is reduced to fit half the context; its output uses the remaining room, capped by any configured or reported model limit. The summary and the rebuilt request are both checked before the working history is replaced. Custom JSON adapters support the earlier-context retries; the summary helper requires a built-in adapter.

Recovery changes conversation context only: it does not undo files or replay actions. The notice points to the original request and archive so the agent can check later tool results before acting again. User messages stay queued until a successful request accepts them. Helper requests are logged as `logs/model/emergency_summary_*.json`. After three failures the harness pauses with every record kept. Oversized pinned instructions, oversized pending input, or an unavailable provider can still need operator intervention.

### Self and meta-memory size

Meta-memory is pinned in full, never truncated. Its 8,000-token guidance threshold triggers a reminder to reorganize it, not a hard cap. Self has a 50,000-character limit. Keep Self about lasting purpose and meta-memory about essential knowledge and navigation; everything detailed belongs in ordinary memory.

### Vision

`no` blocks loading and sending images; images already retained are suspended, not consumed, and come back when vision is re-enabled. With `auto`, setup tests an image unless the server reports text-only input; a rejected test keeps the verified text connection and disables images. `yes` requires an image check when capability is unknown; a model reported as text-only stays text-only.

`status` shows both the preference and the effective mode. After changing the model behind the same endpoint, reconnect and restart. If a running model rejects an image request in auto mode, the harness releases active images and retries text-only with the same pending notifications.

### Reasoning and sampling

`--reasoning auto` clears the override and leaves the choice to the server. `off` requests disabled reasoning where supported; `on` exists for APIs with a real toggle. Models with effort levels expose those instead (`minimal` through `max`). `--reasoning-budget-tokens` sets an exact budget where supported. More reasoning may need a larger output allowance and a longer timeout.

| Provider | Mapping |
|---|---|
| llama.cpp | `reasoning_effort`; `on` uses `chat_template_kwargs.enable_thinking`. Effort needs a supporting template; exact budgets and OpenAI reasoning mode are rejected. |
| Ollama | Native `think`; GPT-OSS uses low, medium, or high. |
| vLLM | `reasoning_effort` and optional `thinking_token_budget`. |
| OpenRouter | `reasoning.effort` or `reasoning.max_tokens`. |
| OpenAI | Responses `reasoning.effort`. |
| Gemini | Model-specific `thinkingLevel` or `thinkingBudget`. |
| Anthropic | `output_config.effort` or a model-specific thinking budget. |
| Custom | Compatible fields or the JSON contract; behavior depends on the server. |

Optional generation flags: `--temperature`, `--max-output-tokens`, `--top-p`, `--top-k`, `--min-p`, `--frequency-penalty`, `--presence-penalty`, `--repetition-penalty`, `--seed`, and repeatable `--stop-sequence`. `--reasoning-mode standard|pro` is available for supported OpenAI configurations. Support varies by transport and model, and setup validates what applies. An accepted parameter does not prove the model uses it.

Switching models keeps harness preferences, Self, memory, and work products. A connection change clears explicit generation controls; a provider or address change also clears custom headers and request options. Supply new values with the change, or clear old ones explicitly:

```bash
python3 artificium.py connect --reset-generation-settings --reasoning auto
```

To pin one OpenRouter inference provider, use `configure model --openrouter-provider cerebras`, which sends `provider.only: ["cerebras"]`; `automatic` removes the constraint. A pinned route fails if that provider cannot serve the model and controls requested.

</details>

<details>
<summary><strong>Daily operation and autonomous work</strong></summary>

Run commands as `python3 artificium.py COMMAND`, from any directory; the launcher always works on its own clone. Add `--help` for options; `--version` prints the application version.

| Command | Purpose |
|---|---|
| `setup` | Connect the model and prepare the instance; `--no-launch` skips the launcher menu. |
| `chat` | Start or reuse the background agent and open a terminal interaction client. |
| `start` / `stop` / `restart` | Manage the background life-loop; `restart` applies saved configuration. `stop` escalates to a forced stop if needed. |
| `run` / `run --once` | Run continuously in the foreground, or do a bounded amount of work and return. |
| `watch` | Follow the life-loop trace without starting the agent. |
| `status` / `status --json` | Read-only process, context, event, vision, and offload status. |
| `send` / `show` | Publish an interaction event, or print an interaction's events. |
| `notify` | Queue a generic notification. |
| `attention` | Queue an Infinite Attention request for a source and an objective. |
| `configure harness` / `configure model` / `connect` | Change harness preferences or the model connection. |
| `config` | Show saved settings; add `harness` or `model` for one section. |
| `models` | Discover served models without inference. |
| `doctor` / `check` | Inspect the request mapping, or test the complete connection. |
| `key` | Verify and save a replacement API key. |
| `logs` | Recent trace entries; `--lifetime`, `--feature NAME`, and `--summary` give other views. |
| `upgrade` / `upgrade --check` | Fast-forward the clone with `git pull`, or preview the change. |
| `reset` | Delete `workspace/` and grow a fresh mind from the seed; the model connection is kept. |
| `overrides` / `overrides diff [NAME]` / `overrides clear [NAME]` | List, compare, or remove the agent's prompt and code overrides. |

Aliases: `init` for `setup`, `reconfigure` for `configure`, `stream` for `attention`. `send`, `notify`, and `attention` queue input without starting the process. In foreground `run`, the first Ctrl-C requests shutdown and the second forces exit.

Continuous `run` and background `start` have no turn-count or turn-duration limit; incoming events, stop requests, and API-key changes are checked between rounds. Only `run --once` uses `harness.max_life_loop_rounds` and `harness.max_turn_seconds` (defaults: 64 rounds, 900 seconds). The time budget is checked between rounds and never interrupts inference.

### Watching the life-loop

`watch` renders the same view as a foreground `run`, with local timestamps, and survives log rotation. Useful options:

| Option | Effect |
|---|---|
| `--tail N` | Show the last N displayed records first (default 100); records the view skips, such as per-second progress during a long request, do not count. |
| `--only CATEGORIES` | Comma-separated subset of `turns`, `engine`, `tools`, `thoughts`, `context`, `memory`, `notifications`, `recovery`. |
| `--no-thoughts` | Hide the model's thoughts and outputs. |
| `--reasoning` | Also show provider reasoning traces (hidden by default). |
| `--full` / `--max-lines N` | Show long blocks in full, or truncate them after N lines (default 6). |
| `--since HH:MM` | Only show records at or after a local time or ISO timestamp. |
| `--json` / `--no-color` | Raw JSON records, or plain uncolored text. |

Each model response line shows input, cached, and output tokens, generation speed, and the count source. Tool lines show the shell command, return code, and duration. The context line shows a bar against the working-memory target.

### Chat, attachments, and large sources

Chat supports `/history`, `/attach PATH MESSAGE`, `/status`, `/help`, and `/quit`. For paths with spaces, use the CLI's quoted attachment argument:

```bash
python3 artificium.py send --interaction main --sender user_1 --attachment "my image.png" "Inspect this."
python3 artificium.py show main
python3 artificium.py attention ./large-source.txt "Extract the decisions and supporting evidence" --granularity fine --output mind/space/decisions.txt
```

`attention` accepts a file or directory; `--granularity` is `auto`, `coarse`, or `fine`; `--output` names where the result belongs. These commands hand work to the agent, which must be running to act on it. A queued request is not a completed result.

### Give an instance a standing purpose

To prepare a fresh instance without opening chat:

```bash
python3 artificium.py setup --no-launch
```

This seeds `workspace/mind/self.txt`. Before starting, edit it to describe the instance's purpose, priorities, and use of idle time. For example:

```text
My continuing purpose is to investigate the research question described in
mind/space/research-question.md. Continue useful work without waiting for chat.
Preserve evidence, failed approaches, reproducible experiments, and open
questions in memory. Verify claims against sources or executable checks.
Keep a progress record and use checkpoints so the investigation can resume.
Sleep when progress requires an external event or a scheduled opportunity.
```

Write the referenced brief with a concrete question and success criteria, then start and observe:

```bash
python3 artificium.py start
python3 artificium.py watch
```

Initial setup also accepts `--self-file /path/to/self.txt` or `--self "TEXT"`. Self is mutable, so a standing purpose is guidance the agent can revise, not an enforced policy.

Continuous operation has no lifetime work budget. Repeated identical no-action output triggers backoff, and engine failures keep their recovery and pause behavior. Use provider spending controls and `status` to manage an unattended experiment.

</details>

<details>
<summary><strong>Files, agent tools, and interaction integrations</strong></summary>

### Project layout

| Path | Contents |
|---|---|
| `artificium.py` | Launcher. |
| `README.md` | This guide. |
| `app/artificium/` | The Python runtime. |
| `app/prompts/` | The prompt pack (promptgramming). |
| `app/seed/` | The starting mind, copied into `workspace/mind/` on first start. |
| `app/tests/` | Offline tests. |
| `config.json` | Model connection and harness settings. Ignored by Git; survives upgrades and resets. |
| `.secrets.json` | Saved API key, mode 0600. Ignored by Git; survives upgrades and resets. |
| `workspace/` | Everything the agent writes. Ignored by Git; deleting it resets the agent. |
| `workspace/mind/self.txt` | Mutable identity, purpose, and initiative. |
| `workspace/mind/meta_memory.md` | Pinned essential knowledge, memory map, and tool registry. |
| `workspace/mind/memory/` | Learned knowledge, operating guides, agent-authored indexes, and checkpoints. |
| `workspace/mind/tools/` | Reusable programs and clients, including the scheduler and search tool. |
| `workspace/mind/space/` | Projects, experiments, and work products. |
| `workspace/mind/interactions/` | Interaction metadata, durable events, and copied attachments. |
| `workspace/logs/` | Model exchanges, traces, working history, archived contexts, attention state, queues, and runtime state. |
| `workspace/overrides/` | The agent's prompt and code overlays. |

The agent's own paths are written relative to `workspace/`, so `mind/self.txt` and `logs/` read the same wherever the clone sits; the project around it is `../`. General tool paths are relative to the workspace unless absolute. Memory-tool paths are relative to `mind/memory/`, for example `projects/research/verified-method.txt`; memory references returned by the harness begin with `memory/`. Moving a memory or tool also means repairing its index and meta-memory references.

### Managing overrides

`overrides` lists active overlays, `overrides diff [NAME]` compares one with the shipped file, and `overrides clear [NAME]` removes it (for example `overrides diff code/runtime.py`). Starting with `--no-overrides`, or with `ARTIFICIUM_NO_OVERRIDES=1`, runs stock code for diagnosis. The failed-start counter resets once a start succeeds. Because overrides live in the workspace, `reset` also restores the factory harness.

### Agent tools

These are model-facing tools, not CLI commands. The [tool contract](app/prompts/tools/core_tools.md) gives their arguments and behavior.

| Capability | Tools |
|---|---|
| Files and shell | `list_directory`, `read_file`, `write_file`, `run_shell` |
| Active images | `load_images`, `list_loaded_images`, `release_images` |
| Durable memory | `save_memory`, `search_memory`, `remove_memory` |
| Continuity and identity | `offload_working_memory`, `revise_self`, `finish_initialization`, `sleep` |
| Interactions | `list_interactions`, `read_interaction_event`, `set_interaction_event_status`, `send_interaction` |
| Scheduling | `schedule_task`, `list_scheduled_tasks`, `cancel_scheduled_task` |
| Infinite Attention | `open_attention`, `checkpoint_attention`, `next_attention_chunk`, `refine_attention`, `complete_attention`, `list_attention_streams` |

`load_attachment` and `compact_context` remain as compatibility operations; `load_images`, text reads, and `offload_working_memory` replace them.

Ordinary reads are bounded; oversized output is reported with source and output paths instead of passing as a complete reading. Use Infinite Attention for large text.

`run_shell` is synchronous, with a 120-second default timeout and a per-call maximum of 3,600 seconds. Each command runs in its own process session with stdin closed, so a command waiting for input returns immediately and a timeout kills the whole process group, background children included. Large output keeps the head and tail of stdout and stderr separately. A timeout returns guidance to check for surviving processes and saved output, and to use background execution for long work; the harness does not relaunch the command.

### Python clients

Add the project's `app/` folder to Python's import path, then use the bundled client:

```python
from artificium import ArtificiumClient

client = ArtificiumClient('/absolute/path/to/agent-artificium')
event, path = client.send(
    'project-room',
    sender='user_1',
    recipient='artificium',
    content='Inspect this image.',
    attachments=['/absolute/path/image.png'],
    interaction_name='Project room',
    kind='message',
)
for event in client.events('project-room'):
    print(event['sender'], event['content'])
```

The client copies attachments before publishing the event, then writes a receipt and a compact notification. It does not start the agent. Applications, sensors, and other agents use the same interface with their own sender IDs and event kinds. Connecting separate instances or external services needs a client or bridge; there is no bundled remote transport or swarm manager.

### Filesystem event contract

| Path (relative to `workspace/`) | Purpose |
|---|---|
| `mind/interactions/ID/interaction.json` | Interaction metadata and participants. |
| `mind/interactions/ID/events/EVENT_ID.json` | Immutable inbound or outbound event. |
| `mind/interactions/ID/attachments/` | Durable attachment copies. |
| `logs/runtime/interaction_receipts/` | Delivery, reading, and handling state; update through runtime APIs. |

An inbound event looks like this:

```json
{
  "id": "event_example",
  "interaction_id": "project-room",
  "interaction_name": "Project room",
  "created_at": "2026-09-07T12:00:00.000000Z",
  "sender": "user_1",
  "recipient": "artificium",
  "direction": "inbound",
  "kind": "message",
  "content": "Hello.",
  "attachments": [],
  "in_reply_to": null
}
```

IDs contain 1–128 letters, digits, dots, dashes, or underscores. Use UTC ISO 8601 timestamps and event filenames that match their IDs. A writer outside Python can publish metadata, attachments, and events by writing each file to a temporary sibling and renaming it; finish attachment copies first. The runtime reconciles inbound event files that lack receipts; it does not watch arbitrary files.

A reply has `direction: "outbound"`. Preserve sender, recipient, interaction ID, and `in_reply_to`, and sort events by timestamp and ID. Delivery does not imply handling: the agent may reply, postpone, or ignore. Consumers should tolerate retries, since recovery does not promise exactly-once external side effects. Remote clients must authenticate and sanitize rendered content.

</details>

<details>
<summary><strong>Security, troubleshooting, backups, upgrades, and development</strong></summary>

### Security and privacy

The deployment environment defines the agent's permissions. Restrict host mounts, network access, and credentials outside the environment the agent controls. Self, memory, tools, and overrides are writable wherever the agent's Linux account allows. The file tools' refusal to write outside `workspace/` is a guardrail, not a boundary, and reflection gates and prompt guidance neither isolate the agent nor reliably contain prompt injection.

The agent can read its own API key, so enforce paid-API limits at the provider with a dedicated key or account, and keep account administration credentials out of the instance. A local model removes metered usage but not filesystem or shell access.

Sender IDs do not authenticate anyone. All interactions share the instance's mind; there are no private per-entity memory boundaries.

Memory, attachments, backups, logs, and raw model requests may contain private data. With a remote model, the assembled context (Self, meta-memory, retrieved material, loaded images) goes to that provider. Treat the whole project folder as private once it has run, and never publish it as is; Git already ignores the private parts.

### Troubleshooting

| Symptom | Next step |
|---|---|
| Setup or model requests fail | `connect` to repair the connection, or `check` to test it without changing settings. |
| HTTP 401 or 403 | 401: check credentials, and use `key` to replace one. 403: check endpoint, permissions, and any proxy. |
| Local server reports insufficient context | Increase the server's real allocation if the hardware allows, then reconnect; otherwise shrink the prompt and context footprint. |
| Repeated mandatory offloads | Check `status`, pinned prompt size, and checkpoint size. Leave room for real compression, and adjust the threshold, the working-memory target, or the server allocation. |
| Frequent cache-miss warnings in `watch` | Long idle gaps let provider caches expire; misses after short gaps usually mean backend re-routing. For OpenAI-style servers, try `model.prompt_cache_key: "auto"`. |
| Model changed behind the same endpoint | `configure model` to refresh discovery and verification, then `restart`. |
| A message gets no reply | Check the process is running, then inspect `show`, receipts, and logs. Queued or delivered input is not a response; trace output is internal. |
| Background process fails to start | Read `workspace/logs/daemon.stderr.log` and `logs --lifetime --limit 20`. A missing or broken `workspace/mind/tools/scheduler.py` must be repaired or restored. |
| A fact is missing after offloading | Look in its memory branch, the interactions, or the archived context; recover the evidence and repair the memory and indexes. |
| Odd behavior after a self-edit | `overrides` and `overrides diff NAME` show what the agent changed; `overrides clear NAME` removes it, and `--no-overrides` confirms whether stock code behaves. |

Do not delete Self, memory, or runtime state to fix an API setting. Timed-out inference is not blindly replayed; explicit transient HTTP failures get bounded retries, and the life-loop backs off before continuing.

### Back up

Back up `config.json`, `.secrets.json` (securely), and all of `workspace/`; the rest comes back with `git clone`. The logs are not just diagnostics; they hold resumable state and the raw evidence Infinite Attention can revisit.

### Upgrade

From the project folder:

```bash
python3 artificium.py stop &&
python3 artificium.py upgrade &&
python3 artificium.py start
```

`upgrade` is a guarded `git pull`: it refuses while the agent runs, refuses if tracked files were edited locally, then fetches the tracked branch and fast-forwards to it. Git ignores `config.json`, `.secrets.json`, and `workspace/`, so they are never touched. If the clone has commits of its own that are not upstream, nothing changes and you reconcile them with Git yourself. A clone that still tracks the original project (`officialgr/agent-artificium`) is moved to this fork first, and `upgrade` says so; `upgrade --check` previews the fork without moving anything. A plain `git pull` with the agent stopped does the same, without the checks below.

An override keeps winning after an upgrade even if the file it shadows changed. `upgrade` lists every override that hides a changed file; review each with `overrides diff NAME` and drop stale ones with `overrides clear NAME`. New seed memories are delivered on the next start without touching existing ones.

`upgrade --check` fetches and previews the change without applying it, even while the agent runs.

### Reset

```bash
python3 artificium.py reset
```

Run it with the agent stopped. It deletes `workspace/`, after confirmation (`--yes` skips it), and grows a fresh one from the seed. The model connection survives because `config.json` and `.secrets.json` sit outside the workspace. Deleting `workspace/` by hand has the same effect; to keep an old workspace, rename it instead.

### Moving an existing installation

An installation with the earlier `app/` + `workspace/` layout moves into a fresh clone with its settings and mind intact:

```bash
git clone https://github.com/EmmanuelARB/agent-artificium.git ~/agent-artificium
mv OLD_INSTALL/config.json OLD_INSTALL/.secrets.json OLD_INSTALL/workspace ~/agent-artificium/
```

Installations older than that layout have no migration path; start a new one.

### Development

```bash
python3 -m unittest discover -s app/tests -q
```

The starting mind is `app/seed/`, laid out one-to-one onto `workspace/mind/`: `self.txt`, `meta_memory.md`, `memory/harness/*.txt`, `memory/tools/*.txt`, and `tools/` (scheduler and search). Edit those files to change what a fresh agent starts with. The version is set in `app/artificium/version.py`. A release is a Git tag; `git archive` or GitHub's source download packages it, and never includes the ignored instance files.

Offline tests cover runtime mechanics and request contracts. Evaluate model performance and continual improvement separately, with stated tasks, models, budgets, and success criteria.

</details>

## Future improvements

Some ideas I want to explore next:

- **Built-in sub-agents and agent swarms:** spawning, delegation, coordination, and shared or separate memory.
- **Stronger security:** sandboxing, scoped filesystem and network access, credential isolation, and authenticated integrations.
- **Better life-loop inspectability:** clearer views of context changes and memory updates, beyond what `watch` shows today.
- **Continual learning through weight updates:** exploring QLoRA fine-tuning on long-term memories to incorporate accumulated experience into the model's weights.
- **Benchmarks and tests:** broader test suites and repeatable benchmarks to measure Artificium's capabilities, reliability, and progress over time.

Experiments, criticism, and reports of what worked or failed are welcome.
