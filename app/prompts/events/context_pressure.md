[SYSTEM NOTIFICATION — WORKING-MEMORY PRESSURE]

Working context is approximately {{context_tokens}} tokens against a
{{context_window_tokens}}-token working-memory target ({{context_percent}}%).
The soft working-memory-offload threshold is
{{soft_threshold_percent}}%; the hard offload threshold is
{{hard_threshold_percent}}%.

Decide whether the current detailed context still earns its cost or should be
compressed and offloaded into long-term memory. Completed or superseded work is
a strong reason to offload even when some context capacity remains.

If you choose offloading, call `offload_working_memory`. Its first stage issues
a memory-formation reflection before replacing context. Preserve:

- Current objective and why it matters
- Interaction and event IDs that still matter
- Verified findings, decisions, and relevant uncertainty
- Entity commitments and promised replies
- Artifact, tool, memory, and evidence paths
- Failed approaches that should not be repeated
- Unfinished work and the exact next action

Use a descriptive memory path and retrieval description. At or above the hard
threshold, working-memory offloading is required before substantial additional
work. Do not delete obligations or guess what can be recovered.
