"""Slash commands and command advertisement for the ACP front.

ACP carries commands as ordinary prompt text: the client sends what the user
typed and the agent recognizes the prefix.  The terminal front handles these
interactively; this module is the headless equivalent, so a client such as
agent-shell can run the same commands and see the same data.

Skills are advertised but not intercepted.  Their execution belongs to the
model's Skill tool in both fronts, so listing them makes ``/<skill>``
discoverable and lets the existing tool-calling path handle it.
"""

from __future__ import annotations

import json
import os

from . import attachments, loki, provider_controls


class Outcome:
    """What one command produced.

    ``text`` is streamed to the client.  ``model_text`` replaces the prompt
    text and runs a normal turn (used by ``!command``).  ``image`` is staged
    for the next prompt.
    """

    __slots__ = ("text", "model_text", "image")

    def __init__(self, text=None, model_text=None, image=None):
        self.text = text
        self.model_text = model_text
        self.image = image


# Advertised local commands.  /model and /effort are deliberately absent:
# ACP carries them as native session configuration options.
LOCAL_COMMANDS = (
    ("status", "Show provider response status (--json, all, save)"),
    ("account", "Live account data: usage and reset cards"),
    ("pwd", "Show the shell working directory"),
    ("cd", "Change the shell working directory"),
    ("ps", "List background jobs"),
    ("image", "Stage a local image for the next prompt"),
)

_SKILL_MARKER = "SKILL.md"
_SKILL_READ_BYTES = 32 * 1024


def _skill_description(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            head = stream.read(_SKILL_READ_BYTES)
    except OSError:
        return ""
    lines = head.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            key, separator, value = stripped.partition(":")
            if separator and key.strip() == "description":
                return value.strip().strip('"\'')
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped != "---":
            return stripped
    return ""


def _skill_names(root: str):
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    found = []
    for entry in entries:
        directory = os.path.join(root, entry)
        if not os.path.isdir(directory):
            continue
        skill_file = os.path.join(directory, _SKILL_MARKER)
        if os.path.isfile(skill_file):
            found.append((entry, skill_file))
            continue
        # A plugin namespaces its skills as "<plugin>:<skill>".
        try:
            nested = sorted(os.listdir(directory))
        except OSError:
            continue
        for child in nested:
            child_file = os.path.join(directory, child, _SKILL_MARKER)
            if os.path.isfile(child_file):
                found.append((f"{entry}:{child}", child_file))
    return found


def advertised_commands() -> list:
    """The availableCommands list for available_commands_update."""
    commands = [
        {"name": name, "description": description}
        for name, description in LOCAL_COMMANDS
    ]
    root = os.path.join(loki.LOKI_CONFIG_DIR, "skills")
    for name, path in _skill_names(root):
        description = _skill_description(path) or f"Skill: {name}"
        commands.append({"name": name, "description": description})
    return commands


_LOCAL_NAMES = frozenset(name for name, _description in LOCAL_COMMANDS)


def parse(text: str):
    """Return ``(name, argument)`` for a local command, else None."""
    stripped = text.strip()
    if stripped.startswith("!"):
        body = stripped[1:].strip()
        return ("!", body) if body else None
    if not stripped.startswith("/"):
        return None
    head, _separator, argument = stripped.partition(" ")
    name = head[1:]
    if name not in _LOCAL_NAMES:
        return None
    return name, argument.strip()


async def run(text: str, session):
    """Execute TEXT if it is a local command, else return None."""
    parsed = parse(text)
    if parsed is None:
        return None
    name, argument = parsed
    if name == "!":
        return await _bang(argument)
    if name == "status" and argument == "save":
        return await _status_save(argument, session)
    handler = _HANDLERS[name]
    return await handler(argument, session)


def _control_context(session):
    return provider_controls.ControlContext(
        config=loki.current_config(),
        credential_authority=session.credential_authority,
    )


async def _status(argument: str, session) -> Outcome:
    from . import response_headers
    tokens = argument.split()
    show_all = "all" in tokens
    as_json = "--json" in tokens
    config = loki.current_config()
    connected = (
        config is not None
        and config.chat_provider.kind != loki.protocols.DUMMY)
    store = session.response_headers
    if show_all:
        document = store.snapshot()
    elif connected:
        credential = (
            config.auth_spec.credential if config.auth_spec else None)
        document = store.snapshot(
            config.chat_provider.chat_url,
            credential=credential.encode() if credential else None)
    else:
        document = {"version": 1, "endpoints": []}
    if as_json:
        text = json.dumps(document, indent=2, ensure_ascii=True)
    else:
        scope = (
            "All known connections" if show_all
            else "Current endpoint and credential")
        text = (
            f"{scope} (last observed, not live balances).\n"
            "Saved observations plus this runtime's memory; other "
            "runtimes' unsaved observations are not visible.\n")
        if not show_all and not connected:
            text += "No active HTTP chat connection. Use /status all."
        elif not show_all and not document["endpoints"]:
            text += "No observations for the current connection."
        else:
            text += response_headers.render(document)
    hint = provider_controls.live_hint(
        provider_controls.ControlContext(config=config))
    if hint is not None:
        text += "\n" + hint
    return Outcome(text=text)


async def _status_save(argument: str, session) -> Outcome:
    await session.response_headers.save()
    return Outcome(text="Response status saved (this runtime only).")


async def _account(argument: str, session) -> Outcome:
    context = _control_context(session)
    tokens = argument.split()
    if not tokens:
        specs = provider_controls.available_controls(context)
        if not specs:
            return Outcome(
                text="No live account controls for the active connection.")
        lines = ["Account controls:"]
        lines.extend(
            f"  {spec.id} - {spec.title}: {spec.description}"
            for spec in specs)
        lines.append("Run /account CONTROL to read one.")
        return Outcome(text="\n".join(lines))
    spec = provider_controls.find_control(context, tokens[0])
    if spec is None:
        return Outcome(text=f"No such account control: {tokens[0]}")
    result = await spec.read(context)
    lines = list(result.lines)
    if len(tokens) == 1:
        if result.actions:
            lines.append(
                f"Actions (run /account {spec.id} ACTION to perform one; "
                "naming the action is the confirmation):")
            lines.extend(
                f"  {action.id} - {action.title}"
                for action in result.actions)
        return Outcome(text="\n".join(lines))
    action_id = tokens[1]
    action = next(
        (item for item in result.actions if item.id == action_id), None)
    if action is None:
        return Outcome(text=f"No such action: {action_id}")
    lines.append(f"{action.confirm}")
    outcome = await action.run()
    lines.extend(outcome.lines)
    return Outcome(text="\n".join(lines))


async def _pwd(argument: str, session) -> Outcome:
    return Outcome(text=f"cwd: {loki.current_cwd()}")


async def _cd(argument: str, session) -> Outcome:
    captured = []

    def writer(text, file=None):
        captured.append(str(text))

    if loki.change_shell_cwd_from_text(argument, text_writer=writer):
        return Outcome(text=f"cwd: {loki.current_cwd()}")
    return Outcome(text="cd: " + ("\n".join(captured) or "failed"))


async def _ps(argument: str, session) -> Outcome:
    return Outcome(text=loki.run_jobs())


async def _image(argument: str, session) -> Outcome:
    try:
        path = attachments.image_argument_path(argument)
        image = attachments.load_image_attachment(path)
    except attachments.ImageAttachmentError as error:
        return Outcome(text=f"image: {error}")
    return Outcome(
        text=(
            f"Attached image for next prompt: {loki.display_path(image.path)} "
            f"({image.media_type}, {image.byte_size} bytes)"),
        image=image,
    )


async def _bang(command: str) -> Outcome:
    output = await loki.run_bash_async(command)
    return Outcome(
        text=f"{loki.computer}: [Running local command: {command}]\n{output}",
        model_text=(
            f"I ran the local command `{command}`.\n"
            f"Output:\n```\n{output}\n```"),
    )


_HANDLERS = {
    "status": _status,
    "account": _account,
    "pwd": _pwd,
    "cd": _cd,
    "ps": _ps,
    "image": _image,
}
