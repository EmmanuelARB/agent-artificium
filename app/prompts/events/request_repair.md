[SYSTEM NOTIFICATION — AUTOMATIC REQUEST REPAIR]

The harness received this error: {{error}}
Recovery attempt: {{attempt}} of 3. Method: {{method}}.
Original failed request: {{original_log}}
Original working-history archive: {{archive}}

This changed only the conversation supplied to the model. Files, Self, tools,
background jobs, and completed actions were NOT rolled back. Incoming messages
remain queued until a request succeeds. Later tool results are in the archive
(tools: {{later_tools}}). Inspect them before repeating any action.
Active images released from context: {{images_released}}. Original image files remain.

{{continuation}}

If the method above is an isolated summary, the same configured model and API
prepared it in a separate conversation with no tools or Self instructions.
Material omitted from the helper's input: {{omitted}}. The note may omit details
or contain mistakes; consult the original records as needed. You did not perform
this recovery yourself. Continue from the current files and actual task state.
