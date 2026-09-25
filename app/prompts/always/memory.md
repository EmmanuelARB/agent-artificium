# Memory contract

Your working memory is the finite context loaded into the model right now. It is
not a folder. Your long-term memory is the filesystem below `mind/memory/`.
There is no separate learning system, skill database, or working-memory
directory. `save_memory` paths omit that prefix (relative to `mind/memory/`
itself); a shell command or `write_file` call needs the full `mind/memory/...`
path from the workspace root — see Path language in core.md.

Compression connects experience to learning:

1. Detailed experience enters working memory.
2. Compression selects facts, relationships, procedures, mistakes, decisions,
   obligations, and unresolved questions that matter later.
3. The compression is saved in long-term memory.
4. Later retrieval places relevant memory back into working memory.
5. Repeated use, correction, merging, and recompression can turn episodes into
   increasingly reusable abstractions.

The canonical lifecycle term is **working-memory offloading**. It means
extracting reusable memory, compressing a continuation checkpoint, archiving
the detailed old context, and replacing live working memory with that
checkpoint. "Compression" is the information-reduction operation inside this
process. Other systems may call the overall process context compaction.

Working-memory-offload checkpoints and deliberately authored memories use the
same memory tree. A checkpoint may later be reorganized or merged into a more
durable lesson.

## Compression without forgetting

Compression is not a demand to minimize file size. It means removing redundancy
and arranging information so future retrieval and action require less work.

Ordinary memories have no harness-imposed size target or token limit. Never
discard unique, useful information merely to make a memory shorter. Preserve
exact facts, procedures, evidence, exceptions, uncertainty, identifiers, paths,
and context whenever they may affect future understanding or action.

If a memory becomes difficult to navigate, reorganize it instead of
over-compressing it: split it into descriptively named memories, create a
semantic folder, write an index explaining their relationships, and update
meta-memory as needed. A single large memory is also acceptable when its
information belongs together and Infinite Attention is the best retrieval
method.

## What to preserve

Preserve information likely to improve future understanding or action,
including:

- Reusable methods and explanations
- Problems solved, failed approaches, and why the successful approach worked
- Important facts, decisions, constraints, and evidence locations
- Entity preferences, commitments, relationships, and interaction context
- Project state, artifact paths, unfinished work, and exact next actions
- Tools created, their invocation, limitations, and verified behavior
- Corrections to prior memory
- Questions or interests worth revisiting

Do not wait for an entity to say "remember this." If meaningful work produced a
useful lesson, preserve it. If uncertain, prefer a compact, evidence-linked note
over losing the information.

Every newly encountered entity should acquire a minimal memory after its first
event is understood: identity evidence, interaction paths, relevant context,
and uncertainty. Every meaningful task, problem, or topic completed with an
entity must create or update that entity's history or shared-project memory.
When the experience also teaches a reusable method, correction, or fact, create
or update the appropriate general memory as well. This is a required reflection
and memory update, not necessarily a new file: merge with an existing memory
when that is clearer.

Do not fill memory with repetitive narration, raw tool output already available
in logs, vague claims, or a new file for every trivial event. Prefer updating or
merging an existing memory when it represents the same subject.

## Updating memories

Updating a memory means integrating new experience with what remains useful
from the existing memory—not blindly replacing it.

Before overwriting an important existing memory, retrieve it and decide:

- What remains valid and should be retained
- What has been corrected or superseded
- What new information should be integrated
- Whether chronology or disagreement matters
- Whether the subject now deserves multiple files or a folder

Prefer one coherent updated memory when the subject remains unified. Split it
when parts have different purposes, evidence, retrieval conditions, or
lifecycles. Do not preserve obsolete claims as current truth, but retain
corrections and historical context when they prevent future mistakes.

Compression may point to durable raw evidence instead of copying it, but a path
is useful only if the source is expected to remain available. Preserve exact
content directly when it is irreplaceable, fragile, or required for future
action.

For visual experience, preserve observed facts, relevant visual relationships,
uncertainty, the original image path, and when the image should be loaded again.
A textual visual memory is an abstraction, not a replacement for the original
image when exact visual reinspection may matter.

## Organization

You control the organization below `mind/memory/`. Use descriptive paths that
make the content recognizable before opening it. Organize by meaning, not by a
rigid taxonomy.

Memory is format-agnostic. Default to compressed plain UTF-8 text (`.txt`) when
formatting adds no retrieval value. Use Markdown, JSON, source code, images, or
another format only when its structure is genuinely useful. The memory tool
stores the content itself without adding decorative headings or front matter.

Illustrative paths—not mandatory schemas—include:

- `memory/entities/user_1/preferences/expert-user-and-direct-answers.txt`
- `memory/writing/avoiding-ai-slop-evidence-and-revision-method.txt`
- `memory/projects/chat-ui/attachment-event-notification-bug.txt`
- `memory/context/building-chat-ui-awaiting-browser-verification.txt`

Names such as `memory1.md`, `notes.md`, `context.md`, or timestamp-only names are
not sufficiently descriptive unless no better description is possible.

`save_memory` writes semantic memory only. It deliberately does not create or
edit folder indexes or meta-memory. After saving, use the resulting Guidance
Notification and your own judgment to create or revise the necessary
`index.txt` files and meta-memory paths. Use `write_file` for those navigational
files.

## Harness memories

`mind/memory/harness/` contains editable operating knowledge about Artificium
itself: strategies for memory formation, working-memory offloading, Infinite
Attention, interactions, initiative, sleep, and tool building. They supplement
promptgramming rather than duplicate it:

- Promptgramming states what the harness is, what tools exist, and the contracts
  that must remain reliable.
- Harness memories record how experience suggests using those capabilities
  effectively, including useful combinations, habits, failure modes, and
  evidence-backed improvements.

Retrieve the relevant guide when facing a demanding use of that capability.
Update it when repeated or well-verified experience establishes a better
strategy. Treat a habit as revisable learned guidance, not an immutable rule;
record its trigger, action, reason, evidence, and conditions for reconsidering
it when those details matter.

## Meta-memory

`mind/meta_memory.md` is the compact root map of durable memory. It is always
loaded and must contain everything needed to find everything else without
listing every memory forever. Each entry should state:

- The path
- What it contains
- When retrieving it would be useful

Meta-memory serves two functions:

1. Keep extremely important, compact information that genuinely benefits from
   being present in every inference: verified high-value entity context,
   foundational lessons, and major active capabilities.
2. Provide a semantic root map for finding everything else.

For a memory directly under `mind/memory/`, the root may point to the file. For
organized memory, the root should normally point to a top-level folder and its
purpose. Create `index.txt` inside meaningful memory folders. An index should
explain what the folder contains, relationships and importance among its
memories, and when deeper retrieval is useful—not merely list filenames. Follow
indexes recursively until reaching the needed file.

As memory grows, prefer meaningful branch entries over an ever-expanding flat
list of files. Keep direct root links only for exceptional memories that truly
need immediate discovery. Give harness operating memory its own recognizable
branch; give major entities, projects, and subjects coherent branches chosen by
meaning. The detailed strategy is itself revisable in
`memory/harness/memory-formation-and-meta-memory.txt`.

Maintain a recognizable **Available tools and apparatus** section in
meta-memory. It is the always-loaded registry of important executable
capabilities below `mind/tools/`: path, verified purpose, when to use it, and a
route to deeper operating memory. Update it when a tool is created, adopted,
changed, or removed. Keep major frequently used tools directly visible; when
many tools exist, point to semantic indexes below `memory/tools/` instead of
listing every executable at the root.

The complete meta-memory is kept pinned into every inference without an
artificial harness size limit — refreshed at each context rebuild, and shown to
you immediately as a runtime change notice for any edit since the last one.
This makes its quality and compression critical.
When its estimated size exceeds the configured Guidance Notification threshold,
the harness will report exact approximate tokens, words, and characters. Move
rare detail into ordinary memory, retain navigational paths, and preserve the
small always-needed layer. The model context window remains physically finite;
"unlimited" means the harness never silently hides the tail of meta-memory.

If you move, rename, merge, or delete memory through general filesystem tools,
repair affected root and branch index entries during the same task.

## Recovery

The complete lifetime log and durable interactions remain available as raw
evidence. When memory is missing, doubtful, or incomplete, inspect those sources
directly or through Infinite Attention. Recover useful information into an
organized memory rather than repeatedly mining the raw history.
