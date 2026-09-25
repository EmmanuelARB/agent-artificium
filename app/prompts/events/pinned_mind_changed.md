[SYSTEM NOTIFICATION — PINNED CONTEXT CHANGED]

`{{path}}` changed on disk since the pinned system-prompt snapshot was last
refreshed. Pinned-mind snapshotting keeps that block stable between context
rebuilds (process start, a successful working-memory offload, `compact_context`,
or an Infinite Attention checkpoint compression) so the shared prompt prefix
stays cacheable across requests. This notice carries the change so you see it
immediately; the pinned block itself catches up at the next rebuild.

Representation: {{mode}}

{{content}}

[END PINNED CONTEXT CHANGE]
