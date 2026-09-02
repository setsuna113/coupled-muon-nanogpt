# CLAUDE.md

## Sub-agent model rule (hard rule, enforced by a hook)

发起子代理时不能使用 fable，只能使用 opus 或 sonnet。

Every sub-agent launched from this repo must run on **opus** or **sonnet**. Never
fable, never haiku, and never "whatever the parent session is using". This applies
to every surface that spawns a sub-agent:

- **`Agent` tool** — always pass `model: "opus"` or `model: "sonnet"` explicitly.
  An omitted `model` inherits the session model, which is a violation when the
  session runs on fable. `subagent_type: "fork"` is never allowed (a fork always
  inherits the parent model).
- **`Workflow` tool** — every `agent(...)` call in the script must carry
  `{model: 'opus'}` or `{model: 'sonnet'}` as a plain string literal (or an
  identifier bound to one via `const`). No nested `workflow()` calls, no named
  workflows: the guard cannot inspect those.
- **Remote child sessions** (`create_session`) — pass an explicit opus or sonnet
  model id.

Enforcement lives in `.claude/settings.json`, which runs
`.claude/hooks/subagent-model-guard.sh` → `subagent-model-guard.py` as a
`PreToolUse` hook on `Agent`, `Task`, `Workflow` and `mcp__*__create_session`.
The hook denies violations and fails closed (any hook crash also denies). If a
call is denied, fix the `model` and retry; do not route around the hook.

Tests for the guard: `tests/test_subagent_model_guard.py`.
