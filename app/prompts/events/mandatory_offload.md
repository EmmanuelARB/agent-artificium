[SYSTEM NOTIFICATION — MANDATORY WORKING-MEMORY OFFLOADING]

The operator enabled mandatory offloading at {{threshold_percent}}% of the
working-memory target. That threshold was reached, or remaining model context
is needed for generation. Ordinary actions are now withheld
until the existing offload_working_memory process completes successfully. This
is an operator requirement, beyond the ordinary advisory context milestones.

Preflight input count before this notice: {{input_tokens}} tokens
({{token_count_source}}); working-memory target: {{working_memory_tokens}} tokens.
The harness recounts the complete request with this notice before sending it.

Call offload_working_memory without reflection_complete first. While detail is
still present, form reusable memories and a self-sufficient continuation. Then
confirm with reflection_complete:true, a descriptive memory path, checkpoint,
and retrieve_when. Preserve pending interactions, active attention stream IDs
and carry, image paths/observations, unfinished work, and the exact next action.
The harness will archive and replace context through its usual offloading path.

During this phase you may use save_memory, search_memory, remove_memory,
read_file, list_directory, list_loaded_images, release_images, and write_file
within mind/memory/ or to mind/meta_memory.md. The compact_context compatibility
alias also remains available. Ordinary tools, outbound replies, and sleep resume
after a successful offload. A batch containing a disallowed call is withheld
in full; do not combine continuation work with unrelated actions.
