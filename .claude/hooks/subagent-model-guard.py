#!/usr/bin/env python3
"""PreToolUse guard: sub-agents must run on opus or sonnet, never fable.

Project rule (see CLAUDE.md):

    发起子代理时不能使用 fable，只能使用 opus 或 sonnet。

Registered in .claude/settings.json for every tool that launches a sub-agent:

    Agent / Task              tool_input.model must be opus or sonnet
    Workflow                  every agent(...) call in the script must pass
                              model: 'opus' | 'sonnet'
    mcp__*__create_session    tool_input.model must be an opus/sonnet model id

An omitted model counts as a violation: it inherits the parent session's
model, which is exactly how fable would leak into a sub-agent.

Protocol (https://code.claude.com/docs/en/hooks): the hook receives JSON on
stdin. To block a call it prints a hookSpecificOutput.permissionDecision="deny"
object on stdout, echoes the reason on stderr and exits 2. Exit 0 with no
output lets the call through. The guard fails closed: an internal error denies
the call and says why.
"""

import json
import re
import sys

FIX_HINT = 'Set model to "opus" or "sonnet": sub-agents never run on fable (project rule, see CLAUDE.md).'

_FORBIDDEN_TOKEN = re.compile(r"fable|mythos|haiku")
_ALLOWED_TOKEN = re.compile(r"(?<![a-z])(?:opus|sonnet)(?![a-z])")
_ALLOWED_ALIASES = {"opus", "sonnet", "opusplan"}


def model_allowed(model):
    """True for opus/sonnet aliases and any full model id built on opus or sonnet."""
    if not isinstance(model, str):
        return False
    m = model.strip().lower()
    if not m:
        return False
    if m in _ALLOWED_ALIASES:
        return True
    if _FORBIDDEN_TOKEN.search(m):
        return False
    return _ALLOWED_TOKEN.search(m) is not None


# --------------------------------------------------------------------------
# Agent / Task / create_session: a single `model` field
# --------------------------------------------------------------------------


def check_agent(inp):
    if inp.get("subagent_type") == "fork":
        return (
            'subagent_type "fork" always inherits the parent session\'s model. '
            "Use a regular subagent type. " + FIX_HINT
        )
    return _check_model_field(inp, "the sub-agent")


def check_create_session(inp):
    return _check_model_field(inp, "the child session")


def _check_model_field(inp, what):
    model = inp.get("model")
    if model is None or (isinstance(model, str) and not model.strip()):
        return f"model was omitted, so {what} would inherit the parent session's model. {FIX_HINT}"
    if not model_allowed(model):
        return f"model {model!r} is not allowed. {FIX_HINT}"
    return None


# --------------------------------------------------------------------------
# Workflow: inspect every agent(...) call in the script
# --------------------------------------------------------------------------


def check_workflow(inp):
    path = inp.get("scriptPath")
    script = inp.get("script")
    if path:  # scriptPath takes precedence over script/name in the Workflow tool
        try:
            with open(path, encoding="utf-8") as fh:
                script = fh.read()
        except OSError as exc:
            return (
                f"cannot read scriptPath {path!r} ({exc}), so sub-agent models cannot be "
                "verified. Pass the script inline."
            )
    if not script:
        if inp.get("name"):
            return (
                f"named workflow {inp['name']!r} cannot be inspected for sub-agent models. Pass "
                "the script inline with model: 'opus' or 'sonnet' on every agent() call."
            )
        return "no script to inspect, so sub-agent models cannot be verified."
    if not isinstance(script, str):
        return "script is not a string, so sub-agent models cannot be verified."
    return check_workflow_script(script)


_AGENT_CALL = re.compile(r"(?<![\w$.])agent\s*\(")
_WORKFLOW_CALL = re.compile(r"(?<![\w$.])workflow\s*\(")
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")


def check_workflow_script(src):
    """Return a denial reason, or None when every agent() call is on opus/sonnet."""
    masked = _mask(src)

    nested = _WORKFLOW_CALL.search(masked)
    if nested:
        return (
            f"nested workflow() call ({_snippet(src, nested.start())}) launches sub-agents this "
            "guard cannot inspect. Inline those agent() calls instead."
        )

    problems = []
    for call in _AGENT_CALL.finditer(masked):
        open_i = call.end() - 1
        close_i = _match_bracket(masked, open_i)
        where = _snippet(src, call.start())
        if close_i is None:
            return f"unbalanced parentheses in {where}; cannot verify sub-agent models."
        args = _split_top_level(masked, open_i + 1, close_i)
        if len(args) < 2:
            problems.append(f"{where} has no options object, so model is omitted")
            continue
        reason = _check_opts(src, masked, *args[1])
        if reason:
            problems.append(f"{where}: {reason}")

    if problems:
        return "; ".join(problems) + ". " + FIX_HINT
    return None


def _check_opts(src, masked, a, b):
    """Check the second argument of an agent() call (span [a, b) in masked)."""
    a, b = _strip_span(masked, a, b)
    if a >= b:
        return "options argument is empty, so model is omitted"
    if masked[a] == "{":
        return _check_object_literal(src, masked, a)
    ident = _IDENT.fullmatch(masked[a:b])
    if ident:
        init = _find_decl(masked, ident.group(0))
        if init is not None and masked[init] == "{":
            return _check_object_literal(src, masked, init)
        return (
            f"options identifier {ident.group(0)!r} is not bound to an inline object literal, so "
            "model cannot be verified"
        )
    return "options are a computed expression, so model cannot be verified"


_MODEL_KEY = re.compile(r"(?<![\w$.])model\s*(:|,|})")
_SPREAD = re.compile(r"\.\.\.")


def _check_object_literal(src, masked, open_i):
    close_i = _match_bracket(masked, open_i)
    if close_i is None:
        return "unbalanced braces in options object; cannot verify model"
    body_a, body_b = open_i + 1, close_i
    depth = _depth_map(masked, body_a, body_b)

    model_at = None
    value_at = None
    # Search one past the body so the shorthand `{model}` can see its closing brace.
    for m in _MODEL_KEY.finditer(masked, body_a, body_b + 1):
        if m.start() >= body_b or depth[m.start() - body_a] != 0:
            continue  # a `model` key nested inside schema/other object, not ours
        model_at = m.start()
        if m.group(1) == ":":
            value_at = m.end()
        else:  # ES2015 shorthand `{model}` → value is the identifier `model`
            value_at = None
        break

    spreads = [s.start() for s in _SPREAD.finditer(masked, body_a, body_b) if depth[s.start() - body_a] == 0]

    if model_at is None:
        if spreads:
            return "options spread another object without an explicit model, so model cannot be verified"
        return "model is omitted"
    if any(s > model_at for s in spreads):
        return "a spread after model: could override it, so model cannot be verified"

    if value_at is None:
        return _check_identifier_value(src, masked, "model")
    return _check_value(src, masked, value_at, close_i)


def _check_value(src, masked, pos, limit):
    pos, _ = _strip_span(masked, pos, limit)
    if pos >= limit:
        return "model has no value"
    c = masked[pos]
    if c in "'\"`":
        end = masked.find(c, pos + 1)
        if end == -1:
            return "unterminated string for model"
        value = src[pos + 1 : end]
        if c == "`" and "${" in value:
            return "model is a template literal with interpolation, so it cannot be verified"
        if not model_allowed(value):
            return f"model {value!r} is not allowed"
        return None
    ident = _IDENT.match(masked, pos)
    if ident and ident.end() <= limit:
        tail = masked[ident.end() : limit].strip()
        if tail == "" or tail[0] in ",}":
            return _check_identifier_value(src, masked, ident.group(0))
    return "model is a computed expression, so it cannot be verified"


def _check_identifier_value(src, masked, name):
    init = _find_decl(masked, name)
    if init is None:
        return (
            f"model identifier {name!r} has no const/let/var string binding in the script, "
            "so it cannot be verified"
        )
    c = masked[init]
    if c not in "'\"`":
        return f"model identifier {name!r} is not bound to a string literal, so it cannot be verified"
    end = masked.find(c, init + 1)
    if end == -1:
        return f"unterminated string bound to {name!r}"
    value = src[init + 1 : end]
    if c == "`" and "${" in value:
        return f"model identifier {name!r} is bound to an interpolated template, so it cannot be verified"
    if not model_allowed(value):
        return f"model {value!r} (via {name}) is not allowed"
    return None


def _find_decl(masked, name):
    """Index of the initializer of `const|let|var <name> =`, or None."""
    pat = re.compile(r"(?<![\w$.])(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*(?!=)")
    hits = list(pat.finditer(masked))
    if len(hits) != 1:
        return None  # zero or ambiguous (re-declared) bindings are unverifiable
    return hits[0].end()


# --------------------------------------------------------------------------
# Lexical helpers on the masked script
# --------------------------------------------------------------------------

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {")", "]", "}"}


def _match_bracket(masked, i):
    """Index of the bracket matching masked[i], or None."""
    want = _OPEN[masked[i]]
    stack = [want]
    for j in range(i + 1, len(masked)):
        c = masked[j]
        if c in _OPEN:
            stack.append(_OPEN[c])
        elif c in _CLOSE:
            if not stack or c != stack.pop():
                return None
            if not stack:
                return j
    return None


def _split_top_level(masked, a, b):
    """Split masked[a:b] on depth-0 commas; return list of (start, end) spans."""
    spans, depth, start = [], 0, a
    for j in range(a, b):
        c = masked[j]
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == "," and depth == 0:
            spans.append((start, j))
            start = j + 1
    if masked[start:b].strip() or spans:
        spans.append((start, b))
    return spans


def _depth_map(masked, a, b):
    out, depth = [], 0
    for j in range(a, b):
        c = masked[j]
        if c in _CLOSE:
            depth -= 1
        out.append(depth)
        if c in _OPEN:
            depth += 1
    return out


def _strip_span(masked, a, b):
    while a < b and masked[a].isspace():
        a += 1
    while b > a and masked[b - 1].isspace():
        b -= 1
    return a, b


def _snippet(src, i, width=60):
    line = src.count("\n", 0, i) + 1
    text = src[i : i + width].split("\n", 1)[0].strip()
    return f"line {line} `{text}`"


def _mask(src):
    """Copy of src with comment and string contents blanked (same length).

    Quotes and backticks stay in place so spans can be mapped back onto src;
    `${...}` code inside template literals is kept as code.
    """
    out = list(src)
    _scan_code(src, out, 0, False)
    return "".join(out)


def _blank(out, a, b):
    for k in range(a, min(b, len(out))):
        if out[k] != "\n":
            out[k] = " "


def _scan_code(src, out, i, in_template_expr):
    n, depth = len(src), 0
    while i < n:
        c = src[i]
        nxt = src[i + 1 : i + 2]
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            _blank(out, i, j)
            i = j
        elif c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            _blank(out, i, j)
            i = j
        elif c in "'\"":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            _blank(out, i + 1, j)
            i = j + 1
        elif c == "`":
            i = _scan_template(src, out, i + 1)
        elif in_template_expr and c == "{":
            depth += 1
            i += 1
        elif in_template_expr and c == "}":
            if depth == 0:
                return i
            depth -= 1
            i += 1
        else:
            i += 1
    return n


def _scan_template(src, out, i):
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            _blank(out, i, i + 2)
            i += 2
        elif c == "`":
            return i + 1
        elif c == "$" and src[i + 1 : i + 2] == "{":
            _blank(out, i, i + 2)
            j = _scan_code(src, out, i + 2, True)
            if j < n:
                out[j] = " "  # the closing } of ${...}
            i = j + 1
        else:
            _blank(out, i, i + 1)
            i += 1
    return n


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def evaluate(payload):
    """Return (tool_name, denial reason or None)."""
    if payload.get("hook_event_name", "PreToolUse") != "PreToolUse":
        return "", None
    tool = payload.get("tool_name") or ""
    inp = payload.get("tool_input")
    if inp is None:
        inp = {}
    if not isinstance(inp, dict):
        raise ValueError("tool_input is not an object")
    if tool in ("Agent", "Task"):
        return tool, check_agent(inp)
    if tool == "Workflow":
        return tool, check_workflow(inp)
    if tool.startswith("mcp__") and tool.endswith("__create_session"):
        return tool, check_create_session(inp)
    return tool, None


def main():
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("hook input is not a JSON object")
        tool, reason = evaluate(payload)
    except Exception as exc:  # fail closed
        tool = "?"
        reason = (
            f"guard could not evaluate this call ({type(exc).__name__}: {exc}); "
            f"blocking to be safe. {FIX_HINT}"
        )
    if reason is None:
        return 0
    message = f"[subagent-model-guard] {tool}: {reason}"
    sys.stdout.write(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": message,
                }
            }
        )
    )
    sys.stdout.flush()
    sys.stderr.write(message + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
