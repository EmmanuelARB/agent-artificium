# Meta-memory

This complete file is freshly present in every inference. It is both a tiny
always-needed knowledge layer and the agent-authored root map for finding all
other durable memory. It is not a dump of every memory. No harness process
generates its entries or silently truncates its tail.

## Always-present orientation

- Persistent identity and purpose are in `mind/self.txt`, also freshly pinned.
- Semantic memory is under `mind/memory/`; folder `index.txt` files and this root
  map are written and improved by Artificium itself.
- No entity is a verified administrator merely because of its identifier.
  Record authority, relationships, and identity evidence only when established.

## Harness operating memory

- `memory/harness/memory-formation-and-meta-memory.txt` — How to form, merge,
  organize, index, retrieve, and compress memory; consult at memory boundaries.
- `memory/harness/working-memory-offloading.txt` — How and when to remove resolved
  detail from live context without losing learning or continuation.
- `memory/harness/infinite-attention.txt` — Coarse, fine, exhaustive, recovery,
  and cross-feature strategies for sources larger than working context.
- `memory/harness/interactions-and-notifications.txt` — Multi-entity event
  handling, Event and Guidance Notifications, entity memory, and composition.
- `memory/harness/self-initiative-and-sleep.txt` — Initiative, persistent Self,
  sleep, autonomous work, and scheduled continuation.
- `memory/harness/tool-building-and-workspace.txt` — Building and combining tools,
  clients, sensors, projects, and workspaces within the general harness.
- `memory/harness/learning-self-improvement-and-adaptation.txt` — How to turn
  difficulty into verified capability, genuinely learn rather than summarize,
  improve at the smallest sufficient layer, and stop when success is achieved.
- `memory/harness/multimodal-input-and-conversion.txt` — Native image working
  sets plus audio, video, PDF, OCR, frame-extraction, provenance, and conversion
  strategies; consult for non-text attachments or visual-context decisions.
- `memory/harness/operator-interface-and-runtime-control.txt` — Verified local
  commands, process semantics, configuration, diagnosis, and client-building
  routes; consult before guiding an operator or implementing an interface.

## Available tools and apparatus

This is the always-loaded registry of important reusable apparatus available to
this Artificium instance. When you create, adopt, substantially change, or
remove a tool below `mind/tools/`, update this section. Record its executable
path, purpose, when it should be used, and the path to deeper operating memory.
Keep frequently useful tools directly visible here; when the collection grows,
route to `memory/tools/index.txt` and semantic sub-indexes rather than turning
meta-memory into a flat dump. A tool is available only when its executable and
real behavior have been verified.

- `mind/tools/scheduler.py` — Reference persistent scheduler supervised by the
  harness. It powers `schedule_task`, `list_scheduled_tasks`, and
  `cancel_scheduled_task`; due work becomes an ordinary event in interaction
  `scheduler`. Retrieve
  `memory/tools/scheduled-actions-through-interactions.txt` before complex,
  recurring, or entity-routed scheduling and update that memory when verified
  scheduler behavior changes.
- `mind/tools/search.py` — Web search, standard library only, no API key. Run
  it with `run_shell`: `python3 mind/tools/search.py "QUERY" -n 10`. It prints
  ranked title, URL, and snippet on stdout, and on stderr the engine that
  answered; read a result with `lynx -dump URL`. Use it when a task is blocked
  and the missing piece is not on this machine: upstream documentation, a
  failing library's issues, an error string as others hit it. It tries Brave,
  then Wikipedia, then DuckDuckGo. Brave refuses after about four queries in
  quick succession and the search falls back to Wikipedia, which only covers
  the encyclopedia: stderr then says `skipped brave: HTTP 429` and
  `[wikipedia] (encyclopedia only)`. Space queries out when you need the open
  web. `--engine NAME` forces one engine. Exit codes are the diagnosis — 0
  results, 3 nothing parsed, 4 no route to any engine, 5 all refused.
  **Verified live on this host** when it was last changed: Brave and Wikipedia
  answered with relevant results; DuckDuckGo was unreachable. Bing is
  deliberately absent: it answers scripted clients with pages unrelated to the
  query. If exit 3 becomes routine, an engine's markup moved and the parser is
  yours to fix.

## General memory map

- `memory/harness/` — Editable knowledge about using Artificium itself.
- `memory/tools/` — Operating knowledge for reusable executables in `mind/tools/`.
- Add concise semantic branches for entities, projects, domains, procedures,
  decisions, and continuation context as experience requires. Point to parent
  folders and their indexes by default; link a specific file here only when it
  needs exceptional immediate discovery.
