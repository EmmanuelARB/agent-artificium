[SYSTEM NOTIFICATION — RECOVERY]

The previous life-loop execution did not end cleanly.

Failure or restart reason: {{recovery_reason}}
Last confirmed action: {{last_confirmed_action_or_none}}
Uncertain action: {{uncertain_action_or_none}}
Last durable checkpoint: {{last_checkpoint_or_none}}
Pending event count: {{pending_event_count}}
Recovery evidence paths: {{recovery_paths}}

Reconstruct state from durable evidence before repeating an external side
effect. Check interaction delivery records, tool results, created artifacts,
working-memory-offload checkpoints, and relevant life-loop entries. An unconfirmed action is
not automatically a failed action.

If the failure came from the engine, do not enter an immediate unbounded retry
loop. Preserve the obligation, follow the configured backoff, and communicate a
delay only when useful. If ordinary records are insufficient, use Infinite
Attention over the relevant logs.

Once state is established, continue from the last verified point and save any
recovery lesson likely to prevent recurrence.
