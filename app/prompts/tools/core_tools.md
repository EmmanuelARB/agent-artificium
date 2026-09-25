# Core textual tools

These are the exact tools available inside the life-loop. Request them only
through the provider-neutral `<tool_call>` protocol. Optional arguments may be
omitted.

## Files and Linux

### `list_directory`

```json
{"tool":"list_directory","path":"mind","depth":2,"include_hidden":false,"max_entries":500}
```

Returns paths, types, sizes, and whether the bounded listing was truncated.

### `read_file`

```json
{"tool":"read_file","path":"PATH","start_line":1,"max_characters":20000}
```

Reads bounded UTF-8 text. An unbounded request for a file larger than the safe
direct-read limit returns `requires_attention` instead of silently filling the
context. Use Infinite Attention when the entire large source matters. Check
`complete`/`truncated` before treating what you read as the whole file — a
`start_line` or `start`/`max_characters` window can look like the full
content while `complete` is `false`; overwriting from that partial view is a
common way to destroy the rest of a file.

### `write_file`

```json
{"tool":"write_file","path":"PATH","content":"TEXT","mode":"create","allow_shrink":false}
```

`mode` is `create`, `overwrite`, or `append` — the same three modes
`save_memory` accepts, with the same meaning. `create` fails with the
existing file's size if the path already exists (read it fully first, then
retry with `overwrite` or `append`); it never silently replaces content.
Overwriting an existing file first backs up its prior content (the result's
`backup_path`) and is refused if it would shrink a file bigger than 4KB to
under half its size — a common sign of overwriting from a partial
`read_file` window instead of the whole file — unless you pass
`allow_shrink:true`. This general filesystem tool can write anywhere
permitted by the private environment. Use it to author `index.txt` and
`mind/meta_memory.md`; prefer `save_memory` for semantic memory because it
enforces the memory boundary and emits organization Guidance.

All tool paths are relative to the workspace root (durable memory lives under
`mind/memory/...`, not `memory/...`); a not-found read names `mind/...` as a
hint when that is where the file actually is.

### `run_shell`

```json
{"tool":"run_shell","command":"COMMAND","cwd":"PATH","timeout_seconds":120}
```

Runs a shell command in the private Linux environment. Large output is retained
under `logs/outputs/` and the bounded result reports its path. Build reusable
agent apparatus and clients under `mind/tools/`; put ordinary projects,
experiments, code, and writing under `mind/space/`.

**Warning: foreground shell commands block your life-loop.** Until `run_shell`
returns, you cannot make the next model request or handle incoming messages;
those messages remain queued.

Keep `command` short. Write any multi-line script to a file with `write_file`
and run that file; inline `python3 -c` programs and heredocs are the most common
cause of rejected requests, because every quote and backslash inside them must
be JSON-escaped.

Use foreground execution only for brief commands. Long-running calculations,
builds, downloads and servers must run in the background from the start, with
output saved to a file and a recorded PID:

```json
{"tool":"run_shell","command":"nohup COMMAND > mind/space/JOB-LOG.txt 2>&1 & echo $! > mind/space/JOB.pid"}
```

Use the `cwd` argument to set their working directory. Return promptly and
continue other useful work; check progress later with a brief `run_shell`
call (`tail mind/space/JOB-LOG.txt`, `kill -0 $(cat mind/space/JOB.pid)`) or a
`schedule_task` a few minutes ahead that reads the log. Do not wait for a job
inside `run_shell` with long `sleep`, `wait`, or polling loops; that blocks
the life-loop again.

A shell timeout does not guarantee that child processes stopped. Check for
existing processes and results before restarting a timed-out job.
For background launch patterns, consult
`memory/harness/tool-building-and-workspace.txt`.

### `load_images`

```json
{"tool":"load_images","paths":["IMAGE PATH 1","IMAGE PATH 2"],"detail":"auto","retention":"once"}
```

Loads one or more supported images into an explicit visual working set. Durable
interaction attachments and plain paths are inert until you call this tool.
`retention:"once"` is the default: all loaded images appear in the next
successful inference and are then released automatically. Use
`retention:"persistent"` only when the same images must remain visible across
several inferences; persistent images consume input on every request until
released. Reloading an active path refreshes its detail and retention without
creating a duplicate.

### `list_loaded_images`

```json
{"tool":"list_loaded_images"}
```

Lists the active visual working set, including image IDs, paths, detail, and
retention. The same state is summarized in every system-state header.

### `release_images`

```json
{"tool":"release_images","paths":["IMAGE PATH"],"image_ids":[],"all_images":false}
```

Releases selected images from model input without deleting their source files.
Use `{"tool":"release_images","all_images":true}` to release the complete
visual set. Full working-memory offloading also releases every active image.
Images can always be loaded again from their durable paths.

`load_attachment` remains a compatibility operation for older contexts. For a
text attachment use `read_file` or Infinite Attention; unsupported audio,
video, PDF, or binary input requires an appropriate conversion tool.

## Unified memory and Self

### `save_memory`

```json
{"tool":"save_memory","path":"DESCRIPTIVE PATH BELOW MEMORY","content":"COMPRESSED MEMORY","retrieve_when":"WHEN THIS WILL HELP","source_refs":["OPTIONAL EVENT, PATH, OR STREAM ID"],"mode":"overwrite","allow_shrink":false}
```

Creates or updates one semantic memory below `mind/memory/`. If no extension is
supplied, `.txt` is used. Store the content itself, not ceremonial formatting.
`mode` accepts the same `create`, `overwrite`, or `append` as `write_file`,
with the same actionable-size errors, `backup_path` on overwrite, and shrink
guard requiring `allow_shrink:true` to intentionally replace a bigger memory
with a much smaller one. The harness deliberately does not infer or write
folder indexes or meta-memory; it returns a Guidance Notification so you can
author those semantic routes with `write_file`. `index.txt` is reserved for
agent-authored navigation, so choose a descriptive filename for the memory
itself.

### `search_memory`

```json
{"tool":"search_memory","query":"TERMS OR SUBJECT","limit":20}
```

Searches organized memory. Consult the always-loaded meta-memory first when it
already points to the relevant path.

### `remove_memory`

```json
{"tool":"remove_memory","path":"PATH BELOW MEMORY"}
```

Removes an obsolete memory and emits Guidance to repair agent-authored indexes
and meta-memory routes.

### `offload_working_memory`

```json
{"tool":"offload_working_memory","reflection_complete":false}
```

Working-memory offloading is intentionally two-stage. The first call returns a
memory-formation reflection while full detail remains present. Create or update
any reusable memories, then repeat the call with a descriptive `path`, a
self-sufficient `checkpoint`, `retrieve_when`, optional `source_refs`, and
`reflection_complete:true`. The harness saves the checkpoint and emits
organization Guidance,
archives the detailed prior context, replaces working memory with the
checkpoint, and reports before/after token, word, and character estimates.

```json
{"tool":"offload_working_memory","path":"context/DESCRIPTIVE-CONTINUATION-NAME","checkpoint":"SELF-SUFFICIENT CONTINUATION","retrieve_when":"WHEN THIS CONTEXT SHOULD BE RESTORED","source_refs":[],"reflection_complete":true}
```

Do not wait for context exhaustion. When a completed task or resolved
interaction no longer contributes useful live detail, offloading can reduce
cost and interference while retaining its lessons and continuation state.

### `revise_self`

```json
{"tool":"revise_self","content":"COMPLETE NEW SELF","reason":"WHY THIS SHOULD PERSIST","source_event_ids":[],"reflection_complete":false}
```

This is also two-stage. The first call prompts reflection about a persistent
change. Repeat with `reflection_complete:true` only if the change should
endure. The previous `self.txt` is versioned under logs and the new Self is
freshly pinned on the next inference.

## Interactions

### `list_interactions`

```json
{"tool":"list_interactions","status":"all","limit":50,"entity_id":null,"include_events":false}
```

`status` is `pending`, `open`, `sleeping`, `closed`, or `all`. This lists
metadata without forcing whole histories into context.

### `read_interaction_event`

```json
{"tool":"read_interaction_event","event_id":"EVENT_ID"}
```

Reads one durable event and marks it seen. Large event bodies are redirected to
Infinite Attention. Reading does not itself mark the event resolved.

### `set_interaction_event_status`

```json
{"tool":"set_interaction_event_status","event_id":"EVENT_ID","status":"handled","reason":"OPTIONAL"}
```

`status` is `handled`, `postponed`, or `ignored`. A postponed event remains an
obligation visible in runtime state.

### `send_interaction`

```json
{"tool":"send_interaction","interaction_id":"INTERACTION_ID","content":"MESSAGE","in_reply_to":"EVENT_ID OR NULL","attachments":[],"recipient":null}
```

Writes one outbound event to the exact interaction. Plain life-loop text is
logged but is not delivered externally. If the interaction ID does not exist,
this creates a new durable interaction stream; no separate conversation-creation
tool is required.

## Scheduler

The scheduler is persistent and spends no model tokens while waiting. When a
task becomes due, it creates an ordinary inbound `scheduled_task` event in the
`scheduler` interaction, which follows the same notification and wake path as
every other interaction event. Put any entity, interaction, or delivery details
inside the task text; scheduling itself has no entity-routing parameters.
Its editable implementation is `mind/tools/scheduler.py`; Artificium loads that
tool at runtime startup. Tool-specific operating knowledge is stored below
`memory/tools/`.

### `schedule_task`

```json
{"tool":"schedule_task","name":"Review chess experiment","description":"Analyze the newest match after it finishes.","text":"Inspect the latest chess results, update useful memory, and send the conclusions to user_1 in interaction chess-training.","run_at":"2026-08-20T18:30:00Z","repeat_seconds":null}
```

`run_at` must be an ISO 8601 timestamp with a timezone. Omit
`repeat_seconds` for a one-time task; supply a positive interval for recurrence.
The returned task ID is used for cancellation.

### `list_scheduled_tasks`

```json
{"tool":"list_scheduled_tasks","status":"pending","limit":100}
```

`status` is `pending`, `completed`, `cancelled`, or `all`.

### `cancel_scheduled_task`

```json
{"tool":"cancel_scheduled_task","task_id":"TASK_ID"}
```

Cancels a pending task. To change a task, cancel it and schedule a replacement.

## Infinite Attention

### `open_attention`

```json
{"tool":"open_attention","source":"PATH","objective":"PRECISE OBJECTIVE","granularity":"auto","chunk_tokens":null}
```

Opens a durable stream and returns its first chunk. The source may be one file
or a directory of files. `granularity` is `auto`, `coarse`, or `fine`.
Auto/coarse normally uses at most half the configured context window so
compressed carry, prompts, reasoning, and output still fit. Use coarse for
broad synthesis; start fine when exact local structure is the objective.

### `checkpoint_attention`

```json
{"tool":"checkpoint_attention","stream_id":"STREAM_ID","chunk_number":1,"compressed_carry":"OBJECTIVE-SPECIFIC COMPRESSION","decision":"continue","focus_ranges":[],"result":""}
```

After every delivered coarse chunk, echo its exact `chunk_number`. This binds
the checkpoint to the observation and prevents delayed or repeated calls from
advancing the wrong chunk. Preserve only what the objective needs plus exact
evidence/locations when precision matters. `decision` is `continue`, `pause`,
`refine`, or `complete`. `refine` checkpoints and pauses the coarse cursor;
then call `refine_attention`. For `complete`, put the final answer in `result`.

Make only one state-changing attention call per model response, then inspect
its returned state. Repeated checkpoints are handled idempotently; a mismatched
chunk number returns `stale_checkpoint` without changing the stream.

### `next_attention_chunk`

```json
{"tool":"next_attention_chunk","stream_id":"STREAM_ID"}
```

Loads the next chunk after a `continue` checkpoint. If the chunk was already
delivered, the same cached observation is returned. Pending interactions are
visible at the following life-loop boundary and may justify pausing the stream.

### `refine_attention`

```json
{"tool":"refine_attention","stream_id":"STREAM_ID","start":0,"end":100000,"chunk_tokens":10000,"overlap_tokens":500}
```

Rereads a suspicious byte range at smaller granularity without moving the
coarse cursor. Use this for exact tokens, anomalies, acrostics, subtle local
structure, or possible boundary-spanning evidence.

### `complete_attention`

```json
{"tool":"complete_attention","stream_id":"STREAM_ID","result":"FINAL RESULT","result_path":"OPTIONAL PATH BELOW MEMORY","retrieve_when":"OPTIONAL RETRIEVAL DESCRIPTION","source_refs":[]}
```

Completes a stream whose current delivered chunk is awaiting a checkpoint.
Supplying `result_path` also saves the result as long-term memory and emits
organization Guidance;
optional `source_refs` are accepted and preserved with the stream and source
references.

### `list_attention_streams`

```json
{"tool":"list_attention_streams","status":null}
```

Lists durable active, paused, or completed stream state.

## Life-cycle

### `finish_initialization`

```json
{"tool":"finish_initialization","summary":"WHAT WAS INSPECTED AND VERIFIED"}
```

Ends the restart-safe first-wake orientation. Do this once concrete orientation
is complete, not immediately and not ceremonially.

### `sleep`

```json
{"tool":"sleep","mode":"until_event","seconds":null,"reflection_complete":false}
```

`mode` is `until_event` or `timed`. This is two-stage: the first call issues one
pre-sleep reflection; after resolving it, repeat with
`reflection_complete:true`. New interaction events interrupt sleep. A timed
sleep cannot be shorter than the runtime polling interval (one second by default).
