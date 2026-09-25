[SYSTEM GUIDANCE NOTIFICATION — TOOL REQUEST REPAIR]

The preceding response contained one or more invalid textual tool requests.
Artificium reads each response in order: calls before the first invalid one
are a safe sequential prefix and **already executed with real side effects**;
the invalid call and everything after it in that response did not run. The
malformed raw response is preserved for diagnosis at `{{model_log_path}}`,
but its broken syntax was not retained in active working context.

Each repair case below identifies the 1-based call position, requested tool,
concrete error, exact accepted arguments, received arguments, and a canonical
flat request for that specific tool. A case whose error says the call
"already executed" ran normally — do not repeat it. If a tool name was
misspelled, `suggested_tool` and its example identify the nearest available
operation.

{{repair_cases}}

For every call you still need, emit a complete request in this exact envelope:

```text
<tool_call>
{"tool":"TOOL_NAME","argument":"value"}
</tool_call>
```

`tool` is the operation name; all other top-level fields are its arguments.
Think again, repair only the calls that remain useful, and retry. If calls
depend on one another, request only the first, inspect its result, and construct
the next call afterward. Do not repeat the malformed response unchanged.

If a case says the reply was cut off before the call closed, its content was
too large for one response. Split it into several `write_file` calls instead
of one: the first with `mode:"create"`, each following one with `mode:"append"`,
none larger than comfortably fits in a single response.
