# pi-reduce

`pi-reduce` is a Pi package that rebuilds the current branch into a smaller session.

It keeps only the message types you want, drops the rest, and writes the result into a brand new Pi session that you continue from immediately.

No summaries are generated in v1. The reduction is deterministic: keep, drop, and truncate only.

## Install

```bash
pi install npm:pi-reduce
```

Or from git:

```bash
pi install git:github.com/toorusr/pi-reduce
```

## Commands

- `/reduce [chat|reasoning|tools|no-tools|advanced|last]`
- `/reduce-advanced`

## Presets

- `chat` — keep user messages and final assistant output only
- `reasoning` — keep user messages, assistant thinking, assistant comments, and final output
- `tools` — keep user messages plus tool calls and tool results
- `no-tools` — keep user messages plus assistant comments and final output
- `last` — rerun the last reduce config
- `advanced` — open the full reducer UI

## What gets reduced

`/reduce` always starts from the **current active branch context**. It does not rewrite your source session. Instead it:

1. reads the current branch context
2. filters messages by type
3. optionally filters tools by name
4. optionally truncates tool arguments and tool results
5. creates a new session
6. switches into the new session immediately

The current model and thinking level are copied to the new session.

## Advanced options

`/reduce-advanced` lets you configure:

### Message categories

- user messages
- assistant thinking
- assistant comments from tool-using turns
- assistant final messages
- assistant status / aborted / error messages
- tool calls
- tool results
- user bash executions (`!` / `!!`)
- custom / extension messages
- existing branch summaries
- existing compaction summaries

### Tool filters

- keep all tools
- keep only selected tools
- exclude selected tools

### Truncation

- tool call approx-token budget
- tool call char budget
- tool call line budget
- tool result approx-token budget
- tool result char budget
- tool result line budget
- head or tail truncation for tool results

## Output

The final notification shows a compact before/after summary with:

- context reduction bar
- approximate context tokens before vs after
- kept vs removed messages
- key block reductions (thinking, tool calls, tool results)
- recorded source usage totals

The new reduced session also stores provenance in a `reduce-source` custom entry.

## Notes

- Copied messages do not preserve historical assistant usage tags, so the new session starts with `0` recorded usage until you continue chatting.
- Approximate token counts use Pi's own conservative `chars / 4` heuristic.
- Tool-result `details` are replaced with compact reduced metadata so the new session stays light.

## Local development

```bash
npm install
npm run check
```

For a quick local test without installing the package:

```bash
pi -e /absolute/path/to/pi-reduce
```
