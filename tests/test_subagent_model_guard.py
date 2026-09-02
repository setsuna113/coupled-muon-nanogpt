"""Tests for the PreToolUse guard that keeps sub-agents on opus/sonnet (never fable).

The guard lives outside the package (.claude/hooks/) because it is Claude Code
configuration, but it is plain Python and cheap to test, so it is covered here.
No torch needed: `pytest --noconftest tests/test_subagent_model_guard.py` works
in a bare environment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HOOK_DIR = REPO / ".claude" / "hooks"
GUARD_PY = HOOK_DIR / "subagent-model-guard.py"
GUARD_SH = HOOK_DIR / "subagent-model-guard.sh"
SETTINGS = REPO / ".claude" / "settings.json"


def _load_guard():
    spec = importlib.util.spec_from_file_location("subagent_model_guard", GUARD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guard = _load_guard()


def run_hook(payload, *, via_wrapper=True, env=None):
    """Run the hook exactly as Claude Code does: JSON on stdin, decision by exit code."""
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    cmd = ["bash", str(GUARD_SH)] if via_wrapper else [sys.executable, str(GUARD_PY)]
    full_env = {**os.environ, "CLAUDE_PROJECT_DIR": str(REPO)}
    if env:
        full_env.update(env)
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=full_env, check=False)


def assert_allowed(payload):
    res = run_hook(payload)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "", res.stdout


def assert_denied(payload, *fragments):
    res = run_hook(payload)
    assert res.returncode == 2, (res.returncode, res.stdout, res.stderr)
    decision = json.loads(res.stdout)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    reason = decision["permissionDecisionReason"]
    assert reason.startswith("[subagent-model-guard]")
    assert res.stderr.strip() == reason
    for frag in fragments:
        assert frag in reason, reason
    return reason


def agent_call(**tool_input):
    return {"tool_name": "Agent", "tool_input": {"prompt": "x", "description": "d", **tool_input}}


def workflow_call(script):
    return {"tool_name": "Workflow", "tool_input": {"script": script}}


# ---------------------------------------------------------------------------
# model_allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "opus",
        "sonnet",
        "Opus",
        " sonnet ",
        "opusplan",
        "opus[1m]",
        "sonnet[1m]",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-sonnet-5[1m]",
        "claude-opus-4-1-20250805",
        "claude-3-5-sonnet-20241022",
        "us.anthropic.claude-sonnet-5-v1:0",
    ],
)
def test_model_allowed_accepts_opus_and_sonnet(model):
    assert guard.model_allowed(model)


@pytest.mark.parametrize(
    "model",
    [
        "fable",
        "Fable",
        "claude-fable-5-1",
        "mythos",
        "claude-mythos-5-1",
        "haiku",
        "claude-haiku-4-5-20251001",
        "",
        "   ",
        None,
        3,
        ["opus"],
        "opus-fable",
        "sonnetx",
        "default",
        "inherit",
    ],
)
def test_model_allowed_rejects_everything_else(model):
    assert not guard.model_allowed(model)


# ---------------------------------------------------------------------------
# Agent / Task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["opus", "sonnet", "claude-sonnet-5", "claude-opus-5"])
def test_agent_with_allowed_model_passes(model):
    assert_allowed(agent_call(model=model, subagent_type="general-purpose"))


def test_agent_without_model_is_denied():
    assert_denied(agent_call(), "model was omitted", "inherit", '"opus" or "sonnet"')


@pytest.mark.parametrize("model", ["fable", "claude-fable-5-1", "haiku", "", "   "])
def test_agent_with_forbidden_model_is_denied(model):
    assert_denied(agent_call(model=model))


def test_agent_fork_is_denied_even_with_allowed_model():
    assert_denied(agent_call(model="opus", subagent_type="fork"), "fork", "inherits")


def test_legacy_task_tool_name_is_covered():
    assert_denied({"tool_name": "Task", "tool_input": {"prompt": "x"}}, "Task:")
    assert_allowed({"tool_name": "Task", "tool_input": {"prompt": "x", "model": "sonnet"}})


def test_unrelated_tools_pass_through():
    assert_allowed({"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert_allowed(
        {"tool_name": "Edit", "tool_input": {"file_path": "a", "old_string": "b", "new_string": "c"}}
    )


def test_other_hook_events_are_ignored():
    assert_allowed({"hook_event_name": "PostToolUse", "tool_name": "Agent", "tool_input": {"prompt": "x"}})


# ---------------------------------------------------------------------------
# Remote child sessions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool", ["mcp__Claude_Code_Remote__create_session", "mcp__claude-code-remote__create_session"]
)
def test_create_session_requires_explicit_allowed_model(tool):
    assert_denied({"tool_name": tool, "tool_input": {"prompt": "x"}}, "child session", "inherit")
    assert_denied({"tool_name": tool, "tool_input": {"model": "claude-fable-5-1"}}, "not allowed")
    assert_allowed({"tool_name": tool, "tool_input": {"model": "claude-opus-5"}})


# ---------------------------------------------------------------------------
# Workflow scripts
# ---------------------------------------------------------------------------


def test_workflow_with_explicit_models_everywhere_passes():
    script = """
export const meta = {name: 'review', description: 'x', phases: [{title: 'Find', model: 'opus'}]}
const FINDINGS = {type: 'object', properties: {model: {type: 'string'}}}  // schema key named model
const found = await parallel(DIMS.map(d => () =>
  agent(`review (${d.key}) for ${d.desc}`,
        {label: `find:${d.key}`, phase: 'Find', schema: FINDINGS, model: 'opus'})))
const verified = await pipeline(found.flat(), f => agent("verify " + f.title, {schema: V, model: "sonnet"}))
return verified
"""
    assert_allowed(workflow_call(script))


def test_workflow_agent_missing_model_is_denied_with_location():
    script = 'const a = await agent("p", {model: "sonnet"})\nconst b = await agent("q", {label: "x"})\n'
    reason = assert_denied(workflow_call(script), "line 2", "model is omitted")
    assert "line 1" not in reason


def test_workflow_agent_without_options_is_denied():
    assert_denied(workflow_call('await agent("just a prompt")'), "no options object")


@pytest.mark.parametrize("model", ["fable", "claude-fable-5-1", "haiku"])
def test_workflow_forbidden_literal_is_denied(model):
    assert_denied(workflow_call(f"await agent('p', {{model: '{model}'}})"), "not allowed")


def test_workflow_model_via_const_binding():
    assert_allowed(workflow_call("const MODEL = 'sonnet'\nawait agent('p', {model: MODEL})"))
    assert_denied(workflow_call("const MODEL = 'fable'\nawait agent('p', {model: MODEL})"), "via MODEL")
    assert_denied(workflow_call("await agent('p', {model: MODEL})"), "no const/let/var")
    # Re-declared bindings are ambiguous → unverifiable
    assert_denied(
        workflow_call("let M = 'opus'\nlet M = 'fable'\nawait agent('p', {model: M})"), "cannot be verified"
    )


def test_workflow_options_via_const_object():
    assert_allowed(workflow_call("const OPTS = {schema: S, model: 'opus'}\nawait agent('p', OPTS)"))
    assert_denied(workflow_call("const OPTS = {schema: S}\nawait agent('p', OPTS)"), "model is omitted")
    assert_denied(workflow_call("await agent('p', mkOpts())"), "computed expression")


def test_workflow_shorthand_property():
    assert_allowed(workflow_call("const model = 'opus'\nawait agent('p', {model})"))
    assert_denied(workflow_call("await agent('p', {model})"), "cannot be verified")


def test_workflow_spread_semantics():
    assert_allowed(workflow_call("await agent('p', {...base, model: 'opus'})"))
    assert_denied(workflow_call("await agent('p', {model: 'opus', ...base})"), "spread after model")
    assert_denied(workflow_call("await agent('p', {...base})"), "spread")


def test_workflow_computed_model_is_denied():
    assert_denied(workflow_call("await agent('p', {model: pick ? 'opus' : 'sonnet'})"), "computed")
    assert_denied(workflow_call("await agent('p', {model: `${tier}`})"), "template literal")
    assert_denied(workflow_call("await agent('p', {model: args.model})"), "computed")


def test_workflow_ignores_agent_calls_in_comments_and_strings():
    script = """
// agent("old", {}) -- commented out
/* agent(`x`) */
const note = "please call agent(y) later"
const tpl = `agent(${'z'}) inside template`
await agent('p', {model: 'sonnet'})
"""
    assert_allowed(workflow_call(script))


def test_workflow_does_not_confuse_similar_identifiers():
    script = (
        "const subagent = x => x\n"
        "const r = obj.agent(1)\n"
        "await agent('p', {agentType: 'reviewer', model: 'opus'})"
    )
    assert_allowed(workflow_call(script))


def test_workflow_nested_model_key_inside_schema_is_not_ours():
    script = "await agent('p', {schema: {properties: {model: {enum: ['fable']}}}})"
    assert_denied(workflow_call(script), "model is omitted")


def test_workflow_nested_workflow_call_is_denied():
    assert_denied(workflow_call("const r = await workflow('review-changes', args)"), "nested workflow()")


def test_workflow_named_or_empty_is_denied():
    assert_denied({"tool_name": "Workflow", "tool_input": {"name": "review-changes"}}, "named workflow")
    assert_denied({"tool_name": "Workflow", "tool_input": {}}, "no script")


def test_workflow_without_agents_passes():
    assert_allowed(workflow_call("export const meta = {name: 'noop', description: 'x'}\nlog('hi'); return 1"))


def test_workflow_script_path_is_read_and_checked(tmp_path):
    good = tmp_path / "good.js"
    good.write_text("await agent('p', {model: 'opus'})")
    bad = tmp_path / "bad.js"
    bad.write_text("await agent('p', {})")
    assert_allowed({"tool_name": "Workflow", "tool_input": {"scriptPath": str(good)}})
    assert_denied({"tool_name": "Workflow", "tool_input": {"scriptPath": str(bad)}}, "model is omitted")
    # scriptPath wins over an inline script, as in the Workflow tool itself
    assert_denied(
        {
            "tool_name": "Workflow",
            "tool_input": {"scriptPath": str(bad), "script": "await agent('p', {model: 'opus'})"},
        }
    )
    assert_denied(
        {"tool_name": "Workflow", "tool_input": {"scriptPath": str(tmp_path / "missing.js")}}, "cannot read"
    )


def test_workflow_unbalanced_script_is_denied():
    assert_denied(workflow_call("await agent('p', {model: 'opus'"), "unbalanced")


# ---------------------------------------------------------------------------
# Fail-closed behaviour and wiring
# ---------------------------------------------------------------------------


def test_malformed_input_is_denied():
    assert_denied("not json", "could not evaluate")
    assert_denied("[1, 2]", "could not evaluate")
    assert_denied({"tool_name": "Agent", "tool_input": "oops"}, "could not evaluate")


def test_wrapper_denies_when_python_is_missing():
    res = subprocess.run(
        ["/bin/bash", str(GUARD_SH)],
        input=json.dumps(agent_call(model="opus")),
        capture_output=True,
        text=True,
        env={"PATH": "/nonexistent", "CLAUDE_PROJECT_DIR": str(REPO)},
        check=False,
    )
    assert res.returncode == 2
    assert "python3 not found" in res.stderr


def test_wrapper_denies_when_guard_crashes(tmp_path):
    fake = tmp_path / "subagent-model-guard.py"
    fake.write_text("import sys; sys.exit(1)\n")
    sh = tmp_path / "subagent-model-guard.sh"
    sh.write_text(GUARD_SH.read_text())
    res = subprocess.run(["bash", str(sh)], input="{}", capture_output=True, text=True, check=False)
    assert res.returncode == 2
    assert "crashed" in res.stderr


def test_settings_wire_the_guard_to_every_spawning_tool():
    settings = json.loads(SETTINGS.read_text())
    entries = settings["hooks"]["PreToolUse"]
    assert len(entries) == 1
    matcher = entries[0]["matcher"]
    import re

    for tool in ["Agent", "Task", "Workflow", "mcp__Claude_Code_Remote__create_session"]:
        assert re.search(matcher, tool), (matcher, tool)
    for tool in ["Bash", "AgentX", "mcp__github__create_branch"]:
        assert not re.search(matcher, tool), (matcher, tool)
    hook = entries[0]["hooks"][0]
    assert hook["type"] == "command"
    assert "${CLAUDE_PROJECT_DIR}/.claude/hooks/subagent-model-guard.sh" in hook["command"]
    assert GUARD_SH.exists() and os.access(GUARD_SH, os.X_OK)


def test_claude_md_states_the_rule():
    text = (REPO / "CLAUDE.md").read_text()
    assert "发起子代理时不能使用 fable，只能使用 opus 或 sonnet" in text
    assert "subagent-model-guard" in text
