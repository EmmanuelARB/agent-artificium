# Textual tool protocol

All life-loop tools use Artificium's provider-neutral textual protocol. Do not
depend on a provider's native function-calling state or invent tool-call IDs.

A tool request has this form:

```text
<tool_call>
{"tool":"TOOL_NAME","ARGUMENT":"VALUE"}
</tool_call>
```

Use one flat JSON object: `tool` names the operation and every other top-level
field is an argument, and close every request with `</tool_call>`. Arguments
must be valid JSON: inside a string, escape each `"` as `\"` and each literal
backslash as `\\`. Multi-line code is where escaping fails, so do not pass a
script through `run_shell` with `python3 -c "..."` or a heredoc: write it with
`write_file`, then run the file with a short `run_shell` command. The older nested
`{"name":"...","arguments":{...}}` form remains readable for recovery, but do
not generate it. You may emit multiple tool calls in one response
only when they are independent and remain valid regardless of earlier results.
When one action depends on another's result, request the first action, inspect
its result, think again, and then request the dependent action.

The harness returns results as structured textual records containing the call
ID, tool name, status, important paths, bounded output, and truncation or error
information. Never claim a tool succeeded before its result says so. Never
manufacture missing output.

A response is read in order. If a request has malformed JSON, an unknown
tool, or an invalid argument set, or was cut off by the output limit before
closing, the calls before it already ran (a valid sequential prefix is safe),
and that call and everything after it in the same response are withheld. The
malformed response is kept in complete logs but omitted from active working
context, and a repair notification says exactly which calls ran and which
were withheld, with the concrete error and accepted arguments for each.
Correct only the withheld calls that remain useful; do not repeat one that
already ran. A syntactically valid tool that later returns an operational
error remains evidence; change the condition, arguments, or method before
retrying.

When a `write_file` call is large enough that a response might be cut off
before it closes, write it in chunks instead: the first call with
`mode:"create"`, each following one with `mode:"append"`, so a mid-write cutoff
loses at most one chunk instead of the whole file.

Use precise paths and commands. For a file operation, identify the path in the
preceding `<think>` note. For a shell operation, state the concrete purpose.

Ordinary file reading is intentionally bounded. If a file cannot safely fit in
working context, `read_file` must refuse or return bounded metadata rather than
silently truncating it as though it were complete. Use Infinite Attention for
large sources or whenever the entire source matters.

Use the memory-specific tools for semantic memories and working-memory
offloading so the meta-memory index and context lifecycle remain consistent. General
filesystem tools remain available for the wider Linux environment and for tools
you create under `mind/tools/`.

Use the interaction-send tool for external communication. Use the sleep tool
only after the pre-sleep reflection has been resolved for the current wake
cycle.
