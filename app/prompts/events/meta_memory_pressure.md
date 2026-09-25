[SYSTEM GUIDANCE NOTIFICATION — META-MEMORY SIZE]

`{{meta_memory_path}}` is currently approximately {{meta_memory_tokens}} tokens,
{{meta_memory_words}} words, and {{meta_memory_characters}} characters. The
Guidance Notification threshold is {{guidance_threshold_tokens}} tokens.

This is not a hard storage limit and the harness has not truncated the file: the
complete meta-memory is still pinned into this inference. Its size now deserves
deliberate optimization because every token is paid and reread on every model
call.

Use `<think>` to reorganize and compress meta-memory while preserving everything
that truly benefits from being always present:

- Keep a very small section of crucial always-needed facts, verified authority
  or relationship context, foundational lessons, and important current
  capabilities.
- Keep concise sections for Artificium harness memories and major tools, with a
  brief purpose and paths to detailed operating memories.
- Replace flat lists of individual memories with semantic parent-folder entries
  and paths to agent-authored `index.txt` files.
- Move detail, rarely used information, examples, evidence, and long
  explanations into ordinary memory; retain only enough context and retrieval
  guidance to know when and where to load them.
- Remove stale or duplicated information, but do not delete the only route to a
  memory. Verify referenced paths and indexes after rewriting.

Meta-memory should contain everything needed to remember what must always be
known and to find everything else—not everything else itself.
