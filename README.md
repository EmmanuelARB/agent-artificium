# Artificium

**A general agent harness for long term autonomous work, continual learning, and self-improvement.**

> [!NOTE]
> **From the creator, [gr.bio](https://gr.bio/)**
>
> I built Artificium for myself and have been using it as my own harness. This is its first public release: I am sharing it because I think the design could be useful to others, and I want to develop it into a general harness for continual learning and self-improvement.

Artificium is designed to give an agent **full control over its own environment**, let it **work indefinitely with or without external interactions**, and **retain its experience for future retrieval**. The agent **manages its own context window**: it chooses what stays always loaded, what to offload into long-term memory, and what to retrieve or revisit. Across that continuous life-loop, it can build tools, revise its Self, and improve its own methods. The aim is to make its entire history available for learning while keeping its active context focused on the work at hand.

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

Extract the complete source archive. From the extracted project folder, run:

```bash
python3 artificium.py
```

The wizard asks for harness preferences—vision, working-memory target, and optional recovery—then your model service, address or key, and model. It detects context capacity where possible and uses server defaults for reasoning and sampling unless you customize them. Choose terminal chat after setup.

Setup verifies the connection with real, potentially billable model requests before saving. It checks the full harness prompt and image input where applicable, without executing tools or retaining the diagnostic response.

**Chat — interact with the agent.** This starts or reuses the background life-loop:

```bash
python3 artificium.py chat
```

**Watch — inspect the agent's life-loop.** Open another terminal to follow its activity, tool use, and state changes:

```bash
python3 artificium.py watch
```

`watch` attaches without starting the agent. To run the life-loop without opening chat, use `python3 artificium.py start` first.

**Closing either chat or watch leaves the agent running.** To stop it:

```bash
python3 artificium.py stop
```

Supported connections: **llama.cpp, Ollama, vLLM, OpenRouter, OpenAI Responses, Gemini, Anthropic, OpenAI-compatible servers, and custom JSON APIs**. Local inference and custom setup are covered in the complete guide below.

<details>
<summary><strong>Try Artificium on Runpod — setup and example prompts</strong></summary>

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

The agent acts through a Linux environment: it can inspect files, execute commands, use installed software, and build programs. Its capabilities can grow by creating tools inside the same environment where it works.

The harness gives it explicit context about this arrangement: operating instructions, filesystem conventions, available tools, current state, Self, and a map of its memory. Further knowledge of the environment comes from inspection and experience.

Reusable apparatus lives in `mind/tools/`; projects and work products live in `mind/space/`. Operating knowledge belongs in memory, so a capability the agent builds can remain discoverable after its original working context is offloaded. Actual access is defined by the Linux account's permissions.

</details>

<details>
<summary><strong>2. Model connections and textual tools</strong></summary>

The harness owns continuity on disk. Model adapters translate its requests into the selected API's format, so reconnecting a different model preserves Self, memory, interactions, and work products.

Tools use a small textual protocol inside ordinary model output:

```text
<tool_call>{"tool":"read_file","path":"mind/self.txt"}</tool_call>
```

Provider-native function calling is unnecessary. Before executing a response's batch, the runtime checks tool names, JSON, and argument signatures. A protocol error withholds the batch and returns repair guidance. Accepted calls execute sequentially; an operational error does not undo earlier actions. Successful offloading, Self revision, or sleep ends the batch so subsequent work can use the changed state.

This keeps the interaction and memory model consistent across backends. Reasoning controls, vision, capacity, and task performance still depend on the selected model and server.

</details>

<details>
<summary><strong>3. Promptgramming</strong></summary>

**Promptgramming** expresses the agent's operating model as inspectable instructions. Files in `artificium-code/prompts/` define the environment, tool protocol, memory lifecycle, interaction rules, and the meaning of runtime events. Python implements the corresponding mechanics.

Stable operating contracts are included in every inference. Event and guidance prompts provide context when something happens—for example, an input arrives, a tool request needs repair, or memory needs organizing.

Strategies for using these mechanisms live in editable operating memories under `mind/memory/harness/`. The agent can improve its methods through experience while retaining the same underlying interfaces. Foundational prompt or code changes are also possible within its filesystem permissions; the operating guidance calls for recoverable changes and verification.

</details>

<details>
<summary><strong>4. A dynamic Self</strong></summary>

`mind/self.txt` describes lasting identity, purpose, priorities, and initiative. It is loaded afresh for every inference, making it available through working-memory offloads and restarts.

Self can describe an agent that responds and sleeps, one that explores a subject, or one dedicated to a continuing task. A research goal can live there for the life of the instance: the agent can keep investigating, recording results, and resuming work without an open chat or anyone to reply to.

The `revise_self` tool uses two stages: request reflection, then confirm a complete replacement. The previous Self is archived. Detailed knowledge belongs in long-term memory; Self stays focused on what guides the agent's activity. It is intentionally mutable, so a standing goal remains something the agent can reconsider.

</details>

<details>
<summary><strong>5. A continuous life-loop</strong></summary>

The life-loop gives the agent opportunities to act on startup, incoming events, and unfinished work. During continuous operation, a completed generation with no tool calls automatically continues unless the agent is sleeping or the runtime is stopped or paused for error handling. Continuation has no interval or off switch and never interrupts generation or tool execution. Continuous work can span any number of inferences and tool calls, without a round-count or elapsed-time turn limit.

When no message needs attention, Self supplies direction. The agent can continue a project, investigate, build a tool, organize memory, or sleep. The sleep tool requests reflection before sleeping until an event or a timer wakes it.

Sleep uses no inference tokens. The process must stay running to notice wake conditions, and continued work requires an available model backend and compute budget. The scheduler can arrange a future wake, while offloading preserves a useful continuation for the next phase of work.

</details>

### Memory and learning

<details>
<summary><strong>6. Long-term memory as files and folders</strong></summary>

Knowledge lives under `mind/memory/`: facts, methods, decisions, failed approaches, entity and project history, and continuation checkpoints. Files suit an agent that already works through Linux: it can inspect, search, reorganize, and reuse them with the same tools it uses elsewhere.

The agent chooses descriptive filenames and semantic folders. Retrieval uses indexes, text search, direct reads, or Infinite Attention for larger sources. There is no required vector database or separate skill store.

`save_memory` writes the supplied content and returns organization guidance. The agent must preserve useful evidence and retrieval conditions in that content, then maintain the indexes that make the memory discoverable. Later retrieval brings relevant knowledge into working context, where it can be tested, corrected, or extended.

</details>

<details>
<summary><strong>7. Meta-memory and retrieval indexes</strong></summary>

`mind/meta_memory.md` is the root map of the mind. It holds a small amount of essential knowledge plus routes to deeper memory and important reusable tools. The complete file is loaded afresh in every inference.

Folder-level `index.txt` files explain what their memories contain, how they relate, and when to retrieve them. The agent follows these indexes from broad subjects to specific evidence. As memory grows, the root can point to meaningful branches instead of listing every file.

The agent authors this navigation; saving a memory does not automatically update it. A tool created in `mind/tools/` also needs an entry explaining its purpose and use. This makes learned knowledge and executable capabilities available again after offloading. Keeping the root concise matters because it occupies context on every request.

</details>

<details>
<summary><strong>8. Working memory and offloading</strong></summary>

Working memory is the finite context used for the current inference. The request combines recent history and observations with the operating contracts, current Self, complete meta-memory, tool catalog, runtime state, and active images.

**Offloading preserves learning and continuation before releasing detailed history:**

1. The agent requests an offload and receives reflection guidance while the details are still present.
2. It saves reusable lessons, updates retrieval paths, and writes a checkpoint containing active goals, findings, evidence, obligations, and next actions.
3. It confirms the offload. The runtime saves the checkpoint in `mind/memory/`, archives the old working history, and replaces that history with the checkpoint.
4. Work continues with freshly loaded Self and meta-memory. Active images are released; their source files remain available.

The model determines what the checkpoint preserves. Archived context and durable interactions provide recovery sources if something was omitted.

Optional **mandatory offloading** adds a runtime gate. It is off by default; when enabled, its default threshold is 80% of the working-memory target. At the threshold, or when generation needs the remaining context, ordinary actions and sleep are withheld until offloading succeeds. Memory preparation remains available, and the pending requirement survives restart. The harness uses provider token counts where available and estimates otherwise.

The working-memory target defaults to the model's serving context, but can be smaller. For example, a model actually serving 1,000,000 tokens can use a 100,000-token working-memory target, leaving room to finish an offload. This target applies to the complete input, including pinned instructions and active images; it does not silently truncate history.

Optional **automatic repair**, off by default, can recover from rejected input or an empty answer. It archives the history, tries an earlier successful request's context, and tells the agent what happened. There are at most three attempts. Persistent context exhaustion can use an isolated summary from the same configured model and API. Files, completed actions, and incoming messages are retained.

</details>

<details>
<summary><strong>9. Infinite Attention</strong></summary>

Infinite Attention is a high-level reading process for text sources larger than one context window. The agent works through a file or directory in bounded chunks while carrying forward an objective-specific understanding.

1. Open a stream with a source and an objective.
2. Read a chunk with the current carry: findings, evidence locations, unresolved questions, and what to examine next.
3. Checkpoint the chunk with an updated carry. The runtime removes eligible consumed stream material from working history.
4. Advance to the next chunk, then repeat or save the final result.

The objective, source manifest, cursor, chunk number, carry, and status persist on disk. A chunk must be checkpointed before the next one is delivered; requesting it early returns the pending chunk again. Streams can pause and resume, use coarse or fine reading, and refine byte ranges of a single-file source.

This connects large-source analysis to memory: the agent can investigate a corpus, recover a fact from archived context, pause for an interaction, then resume from the saved cursor and carry.

Each inference remains finite, and the carry is lossy. Exact claims need checking against source passages. Keep source files unchanged during a pass: the manifest records initial sizes, not immutable copies. Other working history can still require offloading, and binary sources need conversion first.

</details>

<details>
<summary><strong>10. Continual learning and self-improvement</strong></summary>

Learning happens through a cycle of experience, verification, memory, and reuse. A solved problem can become a procedure; a failure can become a condition to recognize; a recurring need can become a reusable program.

The operating guides ask the agent to test a method, record evidence and limitations, and check whether it transfers to later work. Improvements can live in ordinary memory, a harness operating guide, a tool, or Self. Offloading is a useful point for extracting those lessons, and meta-memory provides the route back to them.

This is adaptation through stored knowledge, instructions, and executable tools. Model weights stay unchanged. The harness provides the mechanisms; the model authors the changes, and useful improvement has to be established through evaluation. Fine-tuning from accumulated memory is one of the future directions below.

</details>

### Inputs, tools, and continuity

<details>
<summary><strong>11. Non-blocking interactions</strong></summary>

People, applications, sensors, scheduled tasks, and external agents use the same durable event format. Each event records sender, recipient, interaction ID, timestamp, content, attachments, and an optional reply target.

Publishing an input does not wait for an answer. A compact notification tells the agent where to inspect it; reading the body and handling the request are separate decisions. The agent can prioritize several inputs, postpone one, combine related work, or continue its standing objective. Delivery, reading, and handling have separate receipts.

New input can arrive while inference or a tool is busy. It becomes available to the agent at an inference boundary; it does not preempt the current model request or make shell execution parallel.

Replies are outbound events created with `send_interaction`; ordinary model text stays in the internal trace. A client or bridge delivers replies to their destination. This gives separate agents a communication primitive and lets a long investigation coexist with incoming questions. Built-in spawning and swarm coordination are future work. All interactions within one instance share its mind.

</details>

<details>
<summary><strong>12. Deliberate perception and image context</strong></summary>

Attachments are copied into durable storage before their event is published. Their paths appear in event metadata; their contents do not automatically enter model context. The agent decides what to read, view, or convert.

For a vision-capable model, `load_images` creates an explicit visual working set. Images can be retained for one successful inference or across several. `release_images` removes them from model input without deleting their sources. One-shot images survive a failed inference; offloading releases the full active set.

Text attachments use file reads or Infinite Attention. Audio, video, PDFs, and other binaries need conversion apparatus, such as transcription, frame extraction, or text extraction. The agent can preserve observations and original paths in memory, then reload the source when exact reinspection matters.

</details>

<details>
<summary><strong>13. Reusable tools and an evolving workspace</strong></summary>

The agent can turn a missing capability into a program under `mind/tools/`, test it, and record how to use it. Projects, experiments, and user-facing work products belong in `mind/space/`.

Custom apparatus normally runs through the shell. Adding a file does not automatically register a native harness tool; meta-memory and operating memories tell the agent that the program exists, when it helps, and how to invoke it.

These primitives can compose into workflows: a sensor publishes observations as interactions; a converter makes an attachment readable; an analyzer produces evidence for a memory; a client connects another agent or service. Browsers, remote UIs, sensors, and service bridges are extensions to build or install, rather than bundled integrations.

</details>

<details>
<summary><strong>14. Persistent scheduling</strong></summary>

The scheduler is editable apparatus in `mind/tools/scheduler.py`, loaded at startup and supervised by the runtime. It stores one-time or recurring tasks and publishes due work as `scheduled_task` interaction events.

Polling continues while inference is busy. Waiting requires no model calls, and due events can wake a sleeping agent. The task text carries the instructions, relevant memory or artifact paths, and where results belong; the same life-loop decides how to execute it.

Recurrence uses intervals in seconds. A task marked `completed` has emitted its event; the agent may still need to carry out the work. This connects time-based activity to ordinary interactions, memory, and Self without introducing a second agent. Scheduler code changes take effect after restart.

</details>

<details>
<summary><strong>15. Durable state, logs, and recovery</strong></summary>

The instance persists working history, interactions, attention streams, scheduling and sleep state, and pending control requirements. Logs retain model requests and responses, tool activity, feature events, and archived contexts. `status`, `watch`, and `logs` expose this state to the operator.

Notification delivery uses a claim-and-commit process around inference. Failed requests release claimed notifications for retry; startup recovers interrupted claims and reconciles inbound events that lack receipts. These mechanisms preserve continuity, but external side effects still need care around retries.

The release includes Self, meta-memory, ten operating memories, and the scheduler. The pre-built mind is already present; the first wake orients the agent to its environment. Startup preserves the instance's memories and does not recreate guides it has deleted. The scheduler executable must remain present and valid.

Together, the mind and logs preserve both selected learning and raw evidence. Back up both: logs contain resumable state as well as diagnostics, and Infinite Attention can revisit their contents when ordinary memory is incomplete.

</details>

## Complete guide

Setup, configuration, operation, integrations, and maintenance. The [API and client reference](artificium-code/REFERENCE.md) also documents the model transports and event contract.

<details>
<summary><strong>Model connections: local inference, credentials, and custom APIs</strong></summary>

### Local inference

Run the model server inside the isolated environment or use an intentionally accessible remote inference service. WSL2 is supported, but its usual shared Windows drives do not provide isolation from your main computer.

For an installed llama.cpp server:

```bash
llama-server -m /path/to/model.gguf --alias local-model --ctx-size 32768 --parallel 1 --port 8080
```

In a second terminal, from the project folder:

```bash
python3 artificium.py setup --provider llamacpp --no-launch
python3 artificium.py chat
```

A single served model is selected automatically. No fake API key is required. `--url` accepts a server root, `/v1` base, or complete chat endpoint. Use a model and context allocation that fit your hardware. Artificium does not install the model server or enlarge its allocation itself. A local vision model also needs the server's appropriate vision/projector configuration.

| Provider flag | Transport | Default local address, if applicable |
|---|---|---|
| `llamacpp` | OpenAI-compatible chat completions | `http://127.0.0.1:8080/v1` |
| `ollama` | Native `/api/chat` | `http://127.0.0.1:11434` |
| `vllm` | OpenAI-compatible chat completions | `http://127.0.0.1:8000/v1` |
| `openrouter` | OpenRouter chat completions | — |
| `openai` | OpenAI Responses | — |
| `gemini` | Gemini `generateContent` | — |
| `anthropic` | Anthropic Messages | — |
| `custom` | OpenAI-compatible endpoint or declarative custom JSON | Supplied by you |

For another OpenAI-compatible service:

```bash
python3 artificium.py setup --provider custom --url http://MODEL_SERVER:8000/v1 --model MODEL_ID --context-window 32768 --no-launch
```

Replace the placeholders with your server and model. For unattended setup, add `--yes` and supply all required values and credentials. Setup still verifies inference. Use `--model ID` when discovery offers several models. If capacity cannot be discovered, specify `--context-window TOKENS` using the actual serving limit. Interactive setup asks once; unattended setup otherwise uses a labelled 32,768-token working budget. A successful test does not verify the entire unknown limit.

Context discovery uses the server's allocation where available: llama.cpp's serving `n_ctx`, vLLM's `max_model_len`, or provider metadata. GGUF training capacity is not a llama.cpp serving allocation. Ollama receives the configured window as `options.num_ctx`. Automatically detected limits refresh on reconnection; explicit manual limits persist. Connection checks and runtime use the same provider counting and output budgeting path.

### Credentials and connection checks

Use the masked setup prompt, `--api-key-file /path/to/key`, `ARTIFICIUM_API_KEY`, or the provider variable: `LLAMA_API_KEY`, `OLLAMA_API_KEY`, `VLLM_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`. Avoid placing a secret directly in a shell command.

Saved credentials live in `artificium-code/.secrets.json`. Explicitly saved keys take precedence so runtime uses the connection that passed verification. Otherwise, `ARTIFICIUM_API_KEY` precedes the provider variable and saved fallback. New saved keys are bound to provider and address. `python3 artificium.py key` verifies a replacement before saving it; a running agent reloads it. The key remains accessible to the agent; apply the spending controls in Quick start.

`doctor` inspects request mapping and available metadata without inference. `check`, also available as `doctor --live`, sends a real diagnostic request with the complete prompt pack, pinned mind, and saved working context, plus an image check when applicable. It does not start the agent, import mutable tools, consume pending events, or retain the diagnostic response. Setup and model reconnection use this same verification before saving.

### Custom JSON APIs

For a service that is not OpenAI-compatible, copy [the example contract](artificium-code/examples/custom-json-contract.example.json) to a file such as `my-engine.json`, adapt it to your API, then run:

```bash
python3 artificium.py setup --provider custom --model MODEL_ID --custom-contract my-engine.json --context-window 64000 --no-launch
```

A contract provides a complete URL, optional static headers, authentication mapping, a JSON request body, and JSON Pointer paths into the response. For example:

```json
{
  "url": "https://api.example.com/generate",
  "auth": {"header": "X-API-Key", "prefix": "", "required": true},
  "body": {
    "model": "$artificium.model",
    "messages": "$artificium.messages",
    "max_tokens": "$artificium.max_output_tokens"
  },
  "response": {"content": "/result/text", "usage": "/usage"}
}
```

Keep the actual key in the credential system. `auth: false` means no authentication; otherwise a supplied key defaults to bearer authentication unless `header` and `prefix` override it.

Exact `$artificium.NAME` strings preserve the substituted JSON type. Available names are `model`, `messages`, `prompt`, `system`, `last_user`, `context_window_tokens`, `reasoning_effort`, `reasoning_budget_tokens`, `reasoning_mode`, `temperature`, `max_output_tokens`, `top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, and `stop_sequences`. Unset optional values are omitted from objects and lists.

`response.content` is required; `reasoning`, `usage`, and `finish_reason` are optional. A list of JSON Pointer paths selects the first existing value. The contract handles one non-streaming JSON POST and an object-shaped JSON response; multipart, streaming, websockets, and executable adapters are outside this interface.

To return to ordinary OpenAI compatibility:

```bash
python3 artificium.py configure model --custom-contract none --url http://localhost:8000/v1 --context-window 32768
```

</details>

<details>
<summary><strong>Configuration: context, offloading, vision, and generation controls</strong></summary>

Settings live in an atomically saved `artificium-code/config.json`, with separate `harness` and `model` objects. Inspect them with `config`, `config harness`, or `config model`. Edit harness preferences offline with `configure harness`; use `connect` or `configure model` for a verified model change. Configuration changes normally require `restart`; API-key replacement reloads automatically.

Automatic continuation replaces the heartbeat setting. Older `heartbeat_seconds` config values are ignored and removed on the next save; the `--heartbeat` flag is no longer accepted.

```bash
python3 artificium.py configure harness --vision auto
python3 artificium.py configure harness --mandatory-offload on --offload-threshold 80
python3 artificium.py configure harness --working-memory-tokens 100000 --auto-repair on
python3 artificium.py connect --reasoning auto
python3 artificium.py restart
```

| Setting | Meaning |
|---|---|
| `--vision auto`, `yes`, or `no` | Image preference, reconciled with discovered and tested model capabilities. |
| `--mandatory-offload on` or `off` | Require a successful offload at the configured context threshold; **off by default**. |
| `--offload-threshold PERCENT` | Working-memory threshold, from 1–95%; default 80%. |
| `--working-memory-tokens TOKENS` or `same` | Target for offloading and attention chunk sizes; default `same` follows model context. Explicit targets must fit inside model context. |
| `--auto-repair on` or `off` | Try earlier context after an input failure; at most three attempts. **Off by default**. |
| `--context-window TOKENS` | Actual model serving capacity; changing this value alone cannot enlarge a server. |
| `--request-timeout SECONDS` or `--request-timeout off` | Inference request timeout; **off by default**. A number enables a timeout of 1–86,400 seconds. Shell timeouts are separate. |

Model settings → **Request timeout** accepts a number or `off`. For example,
`python3 artificium.py configure model --request-timeout off --yes` disables it;
use `--request-timeout 1800` for a 30-minute timeout. Restart a running agent
after saving. Existing saved numeric timeouts are preserved when upgrading.
With `off`, Artificium waits indefinitely for the model's response. Reported
errors still follow the existing retry/repair/pause handling, but a request
that never replies can require operator intervention. Provider or network
limits can still end a request.

### Context and mandatory offloading

When enabled, mandatory offloading is evaluated at inference boundaries. Reaching the threshold withholds ordinary tool actions and sleep until the two-stage offload succeeds. Memory preparation remains available, and the pending requirement survives restart. Context is not silently truncated to get past it.

Token counting uses llama.cpp's native input-count endpoint when available; older builds can render and tokenize text-only requests. vLLM, OpenAI Responses, Anthropic, and Gemini use their input-count APIs. Unsupported endpoints retain the existing estimate. Counts are labelled `provider` or `estimate` in the life-loop and used to check capacity and trigger mandatory offloading.

The input check leaves a counting margin (1% for provider counts, 5% for estimates, at least 256 tokens) and room for any explicitly configured output/reasoning budget. **It does not rewrite normal generation settings or impose a fixed output cap.** Unset optional settings remain with the provider; Anthropic requires an output maximum and uses the model's reported maximum when available. Thinking and the answer still need to fit the serving context. A provider can reject or truncate generation when that room is exhausted; automatic repair can handle that failure when enabled.

A large tool result can still cross a threshold in one step. A threshold below the pinned prompt size—or a checkpoint that barely reduces context—can repeatedly demand offloading. Keep the working-memory target large enough for the full harness prompt, and the serving context larger when possible. To resync a separate target after changing models, use `configure harness --working-memory-tokens same`.

With `--auto-repair on`, context errors, tokenization/template errors, image errors, and empty or malformed responses can trigger recovery. The harness archives the original working history and tries the context before an earlier successful response, with a brief explanation. A second failure goes back further. For a context error, the final attempt uses an isolated summary helper; it may use that helper sooner if no earlier request remains. Three recovery attempts are allowed per incident, including across restarts, until a normal request succeeds. Authentication, network, quota, and configuration failures retain their existing handling.

The helper uses the same model, API, and serving context with its own summary prompt, no tools or Self instructions, and server-default reasoning. Its input is reduced to fit within half the model context; its output limit uses the remaining room and any smaller configured or reported model limit. **There is no special 2K summary cap.** A complete summary and the rebuilt request are checked before replacing working history. Arbitrary custom JSON adapters support the earlier-context retries; the isolated helper requires a built-in adapter.

Recovery changes conversation context only: it does not undo files or replay completed actions. The notice points to the original request and archive so the agent can inspect later tool results before acting again. User messages stay queued until accepted by a successful request; loaded images may be released from context, while their files remain. Helper requests are logged in `logs/model/emergency_summary_*.json`. If three attempts fail, the harness pauses with the records retained. Oversized pinned instructions, oversized pending input, or an unavailable provider can still require operator intervention.

The complete meta-memory is pinned without silent truncation; its default 8,000-token guidance threshold is a reminder to reorganize it, not a hard cap. Self has a 50,000-character prompt limit. Keep Self focused on lasting purpose and meta-memory focused on essential knowledge and navigation; put detailed material in ordinary memory.

### Vision

`no` blocks loading and sending images. Already retained images are suspended rather than consumed and can be used after re-enabling vision. With `auto`, setup tests an image unless the server explicitly reports text-only input. A rejected image test keeps the verified text connection and disables images. `yes` requires an image check when capability is unknown; explicitly reported text-only models remain text-only.

Status reports both preference and effective mode. Reconnect after changing the model behind the same endpoint, then restart. If a running model rejects an image request in auto mode, the harness releases active visual context and retries text-only with the same pending notifications. One-shot images are consumed only after a successful inference. Offloading releases the entire active image set without deleting source files.

### Reasoning and sampling

`--reasoning auto` clears the override and leaves the choice to the server. `off` requests disabled reasoning where supported; `on` is available for APIs with a true toggle. Models that use effort levels expose those instead. `--reasoning-budget-tokens` is an alternative exact budget where supported, and `--reasoning-effort` is also accepted. More reasoning may require a larger output allowance and longer timeout.

| Provider | Mapping in this release |
|---|---|
| llama.cpp | `reasoning_effort`; `on` uses `chat_template_kwargs.enable_thinking`. Effort requires a supporting template; exact reasoning budgets and OpenAI reasoning mode are rejected. |
| Ollama | Native `think`; GPT-OSS uses low, medium, or high. |
| vLLM | `reasoning_effort` and optional `thinking_token_budget`. |
| OpenRouter | `reasoning.effort` or `reasoning.max_tokens`. |
| OpenAI | Responses `reasoning.effort`. |
| Gemini | Model-specific `thinkingLevel` or `thinkingBudget`. |
| Anthropic | `output_config.effort` or a model-specific thinking budget. |
| Custom | Compatible fields or the JSON contract; behavior depends on the server. |

The optional generation flags are `--temperature`, `--max-output-tokens`, `--top-p`, `--top-k`, `--min-p`, `--frequency-penalty`, `--presence-penalty`, `--repetition-penalty`, `--seed`, and repeatable `--stop-sequence`. `--reasoning-mode` is available for supported OpenAI configurations. Support varies by transport and model; setup validates the applicable controls. An accepted parameter does not prove a model or template makes use of it.

Switching models preserves harness preferences, Self, memory, and artifacts. Connection changes clear explicit generation controls; provider or address changes also clear custom headers and request options. Supply desired new values with the change. To explicitly clear old overrides:

```bash
python3 artificium.py connect --reset-generation-settings --reasoning auto
```

For a fixed OpenRouter inference provider, use `configure model --openrouter-provider cerebras`; use `automatic` to clear that constraint. A fixed route may fail if that provider cannot serve the requested model and controls.

For a verified server extension, edit `model.request_options` in the saved configuration, then restart. These options cannot replace protected prompt/model fields or conflict with normalized controls. Do not put credentials in request bodies or static headers. Combined `configure --FLAG` commands are also accepted.

</details>

<details>
<summary><strong>Daily operation and autonomous work</strong></summary>

Run these commands from the project folder as `python3 artificium.py COMMAND`. Add `--help` to a command for its options. The global `--root /absolute/instance/path` option, placed before the command, selects another instance.

| Command | Purpose |
|---|---|
| `setup` | Connect the model and initialize the instance; `--no-launch` finishes without opening the launcher. |
| `chat` | Start or reuse the background agent and open a terminal interaction client. |
| `start` / `stop` / `restart` | Manage the background life-loop; restart applies saved configuration. |
| `run` / `run --once` | Run continuously in the foreground, or perform a bounded amount of work and return. |
| `watch` | Follow the life-loop trace without starting the agent. |
| `status` / `status --json` | Read-only process, context, event, vision, and offload status. |
| `send` / `show` | Publish an interaction event or inspect an interaction's events. |
| `notify` | Queue a generic notification. |
| `attention` | Queue an Infinite Attention request for a source and objective. |
| `configure harness` / `connect` | Change harness preferences or the model connection. `configure model` is also available. |
| `config` | Display saved settings; append `harness` or `model` for one group. |
| `models` | Discover served models without inference; arbitrary JSON contracts have no discovery endpoint. |
| `doctor` / `check` | Inspect request mapping and metadata, or test the complete connection. `doctor --live` equals `check`. |
| `key` | Verify and save a replacement API key. |
| `logs` | Recent trace entries; `--lifetime`, `--feature NAME`, and `--summary` select other views. |

Continuous `run` and background `start` have no turn-count or turn-duration limit. Model actions continue across inference requests, with incoming events, stop requests, and API-key changes checked between rounds. Working-memory offloading, sleep, and error handling retain their own behavior.

Only `run --once` uses the saved `harness.max_life_loop_rounds` and `harness.max_turn_seconds` budgets (defaults: 64 rounds and 900 seconds). The time budget is checked between rounds; it does not interrupt inference. Reaching either budget returns to the operator with working context preserved and adds no turn-limit notification. Model-request and shell timeouts are separate settings.

Aliases: `init` for `setup`, `reconfigure` for `configure`, and `stream` for `attention`. `send`, `notify`, and `attention` queue input without starting the process. Closing chat or watch leaves a background agent running. In foreground `run`, the first Ctrl-C requests shutdown; the second forces exit. Use `stop` when you want the background life-loop to end.

### Chat, attachments, and large sources

Chat supports `/history`, `/attach PATH MESSAGE`, `/status`, `/help`, and `/quit`. For paths containing spaces, use the CLI's quoted attachment argument:

```bash
python3 artificium.py send --interaction main --sender user_1 --attachment "my image.png" "Inspect this."
python3 artificium.py show main
python3 artificium.py attention ./large-source.txt "Extract the decisions and supporting evidence" --granularity fine --output mind/space/decisions.txt
```

`attention` accepts a file or directory. `--granularity` accepts `auto`, `coarse`, or `fine`; `--output` specifies a result destination. These commands submit work to the agent, which must be running to act on it. A queued request is not a completed result.

### Give an instance a standing purpose

To prepare a fresh instance without opening chat:

```bash
python3 artificium.py setup --no-launch
```

Before starting it, edit `mind/self.txt` to describe its purpose, priorities, and use of idle time. For example:

```text
My continuing purpose is to investigate the research question described in
mind/space/research-question.md. Continue useful work without waiting for chat.
Preserve evidence, failed approaches, reproducible experiments, and open
questions in memory. Verify claims against sources or executable checks.
Keep a progress record and use checkpoints so the investigation can resume.
Sleep when progress requires an external event or a scheduled opportunity.
```

Create the referenced research brief with a concrete question and useful success criteria, then start and observe the life-loop:

```bash
python3 artificium.py start
python3 artificium.py watch
```

Initial setup also accepts `--self-file /path/to/self.txt` or `--self "TEXT"`. Self is mutable, so a standing purpose is guidance the agent can revise, not an immutable enforcement policy. No open chat is required for continued work.

Continuous operation has no lifetime work budget. Repeated identical no-action output triggers backoff, and engine failures retain their recovery and pause behavior. Use provider spending controls and runtime status to manage an unattended experiment.

</details>

<details>
<summary><strong>Files, agent tools, and interaction integrations</strong></summary>

### Instance layout

| Path | Contents |
|---|---|
| `artificium.py` | Repository-root launcher. |
| `artificium-code/` | Python runtime, prompt pack, tests, configuration, and saved credentials. |
| `mind/self.txt` | Mutable identity, purpose, and initiative. |
| `mind/meta_memory.md` | Pinned essential knowledge, memory map, and important apparatus registry. |
| `mind/memory/` | Learned knowledge, operating guides, agent-authored indexes, and continuation checkpoints. |
| `mind/tools/` | Reusable programs and clients, including the scheduler. |
| `mind/space/` | Projects, experiments, and work products. |
| `mind/interactions/` | Interaction metadata, durable events, and copied attachments. |
| `logs/` | Model exchanges, tool and feature traces, working history, archives, attention state, queues, and runtime control state. |

General tool paths are relative to the project root unless absolute. Memory-tool paths are relative to `mind/memory/`; for example, `projects/research/verified-method.txt`. Memory references returned by the harness begin with `memory/`. Moving a memory or tool also requires repairing its index and meta-memory references.

### Agent tool families

These are model-facing tool names, not additional CLI subcommands. The [tool contract](artificium-code/prompts/tools/core_tools.md) gives their arguments and behavior.

| Capability | Tools |
|---|---|
| Files and shell | `list_directory`, `read_file`, `write_file`, `run_shell` |
| Active images | `load_images`, `list_loaded_images`, `release_images` |
| Durable memory | `save_memory`, `search_memory`, `remove_memory` |
| Continuity and identity | `offload_working_memory`, `revise_self`, `finish_initialization`, `sleep` |
| Interactions | `list_interactions`, `read_interaction_event`, `set_interaction_event_status`, `send_interaction` |
| Scheduling | `schedule_task`, `list_scheduled_tasks`, `cancel_scheduled_task` |
| Infinite Attention | `open_attention`, `checkpoint_attention`, `next_attention_chunk`, `refine_attention`, `complete_attention`, `list_attention_streams` |

`load_attachment` and `compact_context` remain compatibility operations. Current usage favors explicit image loading or text reads, and `offload_working_memory`.

Ordinary reads are bounded; oversized output is reported with source/output paths rather than silently treated as a complete reading. Use Infinite Attention for large text. Shell execution is synchronous, with a default 120-second timeout and a configurable per-call maximum of 3,600 seconds. Custom apparatus generally runs through the shell; placing an arbitrary Python file in `mind/tools/` does not automatically register a new native tool. The bundled scheduler has a specific runtime loading contract.

A shell timeout returns an error to the agent with guidance to check surviving
processes and saved output, then use background execution for long work. The
harness does not automatically relaunch the command.

### Python clients

Add the instance's `artificium-code/` directory to Python's import path, then use the bundled client:

```python
from artificium import ArtificiumClient

client = ArtificiumClient('/absolute/path/to/Artificium')
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

The client copies attachments before publishing the event, then creates a receipt and compact notification. It does not start the agent. Applications, sensors, and other agents can use the same interface with their own sender IDs and event kinds. To connect separate instances or external services, supply a client or bridge; there is no bundled remote transport or swarm manager.

### Filesystem event contract

| Relative path | Purpose |
|---|---|
| `mind/interactions/ID/interaction.json` | Interaction metadata and participants. |
| `mind/interactions/ID/events/EVENT_ID.json` | Immutable inbound or outbound event. |
| `mind/interactions/ID/attachments/` | Durable attachment copies. |
| `logs/runtime/interaction_receipts/` | Delivery, reading, and handling state; update through runtime APIs. |

An inbound event has this shape:

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

IDs contain 1–128 letters, numbers, dots, dashes, or underscores. Use UTC ISO 8601 timestamps and event filenames matching their IDs. A writer that does not use Python can atomically publish interaction metadata, attachments, and event files with a temporary sibling plus rename. Finish attachment copies first. The runtime reconciles inbound event files that lack receipts; it does not watch arbitrary workspace files.

A reply has `direction: "outbound"`. Preserve sender, recipient, interaction ID, and `in_reply_to`, and sort events by timestamp and ID. Delivery does not imply handling: an agent may reply, postpone, or ignore. Consumers should account for retries; filesystem recovery does not promise exactly-once execution of external side effects. Remote clients must supply authentication and sanitize rendered content. Keep reusable clients in `mind/tools/` and entity-specific projects in `mind/space/`.

</details>

<details>
<summary><strong>Security, troubleshooting, backups, upgrades, and development</strong></summary>

### Security and privacy

The deployment environment defines the agent's permissions. Restrict host mounts, network access, and credentials outside the agent-controlled environment. Self, prompts, tools, and runtime code are writable wherever its Linux account allows them. Reflection gates and prompt guidance do not provide isolation or reliably contain prompt injection.

The agent can access its own API key, so enforce paid-API limits through the provider using a dedicated key or account. Keep account administration credentials outside the instance. Choosing a local model removes metered API usage, but does not reduce filesystem or shell access.

Sender IDs do not authenticate participants. All interactions share the instance's mind; there are no private per-entity memory boundaries. Remote clients need their own authentication and sanitized rendering.

Memory, attachments, backups, logs, and raw model requests may contain private data. With a remote model, the assembled context—including pinned Self, meta-memory, retrieved material, and loaded images—is sent to that provider. Treat the entire used instance as private, and do not publish its directory as a release.

### Troubleshooting

| Symptom | Next step |
|---|---|
| Setup or model requests fail | Run `python3 artificium.py connect` to repair the connection, or `check` to inspect it without changing settings. |
| HTTP 401 or 403 | Check credentials for 401; inspect endpoint, permissions, and any proxy for 403. Use `key` for a verified key replacement. |
| Local server reports insufficient context | Increase the server's real allocation if hardware permits, then reconnect; otherwise reduce the instance's prompt/context footprint. Changing a client number alone cannot enlarge the server. |
| Repeated mandatory offloads | Inspect `status`, pinned prompt size, and checkpoint size. Leave room for useful compression and adjust the threshold or actual context allocation. |
| Model changed behind the same endpoint | Run `configure model` to refresh discovery and verification, then `restart`. |
| A message has no reply | Check that the process is running and inspect `show`, receipts, and logs. Queued or delivered input is not proof of a completed response; plain trace output is internal. |
| Background process fails to start | Inspect `logs/daemon.stderr.log` and `logs --lifetime --limit 20`. A missing or invalid `mind/tools/scheduler.py` must be repaired or restored from a known-good backup. |
| A fact is missing after offloading | Inspect its memory branch, durable interactions, or archived context; recover the evidence and repair the memory and indexes. |

Do not delete Self, memory, or runtime state to repair an API setting. Timed-out inference is not blindly replayed; explicit transient HTTP failures receive bounded retries, and the life-loop can back off before continuing.

### Upgrade Artificium

From the existing project folder:

```bash
python3 artificium.py stop &&
python3 artificium.py upgrade &&
python3 artificium.py start
```

`upgrade` downloads the latest `main` into a temporary checkout and updates the launcher, runtime, promptgramming, tests, examples, scripts, and documentation. It preserves the entire `mind/` (including agent-built tools and the editable scheduler), configuration, credentials, environment files, custom connection files at the code root, and existing logs. The agent must be stopped before changes are applied.

Replaced or removed code is backed up under `logs/upgrades/`. If applying an update fails, the command attempts to restore that code and reports any restoration failure. Local edits inside the managed code directories are backed up and replaced, not merged. Keep a separate backup of the complete instance as well.

Preview without stopping the agent or applying changes:

```bash
python3 artificium.py upgrade --check
```

Use `--ref BRANCH_OR_TAG` to select a version, or `--repo GIT_URL` for a fork. The command uses Git to download source; it does not pull, reset, or advance the Git branch inside the running instance's folder. Use `upgrade` for subsequent instance updates. Use a development checkout for Git work and release builds.

For an older installation that does not have the `upgrade` command yet, stop it and run the new launcher against the existing folder. Run these commands **from your existing instance folder**:

```bash
python3 artificium.py stop &&
ARTIFICIUM_UPGRADE_SOURCE=$(mktemp -d) &&
git clone --depth 1 https://github.com/officialgr/agent-artificium.git "$ARTIFICIUM_UPGRADE_SOURCE" &&
python3 "$ARTIFICIUM_UPGRADE_SOURCE/artificium.py" --root "$PWD" upgrade &&
python3 artificium.py start
```

This preserves the existing mind and removes obsolete code, including the former `artificium-code/prompts/mind-seed/` copies. Updated operating memories in a release apply to new installations; an upgrade leaves the existing agent's learned versions alone.

### Development and release packaging

From a development Git checkout, commit the intended release changes first:

```bash
python3 -m unittest discover -s artificium-code/tests -q
python3 scripts/build_release.py
```

The only pre-built Self, meta-memory, and operating memories are the files under `mind/`; there is no separate seed tree. Edit those files in the development checkout to change the starting mind.

The builder packages committed files from `HEAD` (or `--ref COMMIT_OR_TAG`), excluding instance configuration, credentials, logs, and uncommitted learned changes. Review the committed mind and scheduler before publishing. Uncommitted code edits are also excluded, so a release always corresponds to the selected Git revision.

Offline tests cover runtime mechanics and request contracts. Evaluate model performance and continual improvement separately, with stated tasks, models, budgets, and success criteria.

</details>

## Future improvements

Some ideas I want to explore next:

- **Built-in sub-agents and agent swarms:** spawning, delegation, coordination, and shared or separate memory.
- **Stronger security:** sandboxing, scoped filesystem and network access, credential isolation, and authenticated integrations.
- **Better life-loop inspectability:** clearer views of the agent's activity, tool use, context changes, and memory updates.
- **Continual learning through weight updates:** exploring QLoRA fine-tuning on long-term memories to incorporate accumulated experience into the model's weights.
- **Benchmarks and tests:** building broader test suites and repeatable benchmarks to measure Artificium's capabilities, reliability, and progress over time.

Experiments, criticism, and reports of what worked or failed are welcome.
