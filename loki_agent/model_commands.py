"""Noninteractive commodity-model discovery and explicit provider selection."""

from dataclasses import dataclass
import shlex

from . import models


@dataclass
class ModelCommandResult:
    text: str
    selection: object = None


def is_model_command(text):
    first = text.split(maxsplit=1)[0] if text.split() else ""
    return first in ("/models", "/providers", "/model")


async def run(text, *, credentials, credential_authority=None,
              explicit_connection=None):
    args = shlex.split(text)
    command = args.pop(0)
    provider = None
    if command == "/model" and "--provider" in args:
        position = args.index("--provider")
        if position != len(args) - 2 or position == 0:
            raise ValueError("Usage: /model MODEL --provider PROVIDER")
        provider = args[-1]
        args = args[:position]
    if command == "/model" and provider is None:
        if args:
            raise ValueError("Usage: /model MODEL --provider PROVIDER")
        command = "/models"
    query = " ".join(args)
    if command == "/model" and provider == "explicit":
        if explicit_connection is None or query != explicit_connection.model:
            raise ValueError("No matching explicit LOKI_* connection")
        return ModelCommandResult("", explicit_connection)

    diagnostics = []
    try:
        _catalog, groups = await models.ensure_index(
            credential_authority=credential_authority,
            diagnostic_writer=diagnostics.append)
    except (OSError, ValueError) as error:
        if explicit_connection is None:
            raise
        groups = {}
        diagnostics.append(f"Catalog unavailable: {error}")
    groups = models.filter_supported_groups(groups, credentials)
    if command == "/models":
        names = list(groups)
        if explicit_connection and explicit_connection.model not in names:
            names.append(explicit_connection.model)
        words = query.casefold().split()
        names = [name for name in sorted(names)
                 if all(word in name.casefold() for word in words)]
        return ModelCommandResult("\n".join(
            diagnostics + names + [
                "Use /providers MODEL, then /model MODEL --provider PROVIDER."]))

    # Prefer the commodity's exact display name. Otherwise accept an exact
    # provider model ID; never guess from a substring or choose a provider.
    members = groups.get(query)
    if members is None:
        members = [member for values in groups.values() for member in values
                   if member[2].get("id") == query]
    explicit_match = (explicit_connection is not None
                      and explicit_connection.model == query)
    if not members and not explicit_match:
        raise ValueError("Unknown model; use /models [FILTER]")
    if command == "/providers":
        rows = [f"{pid}: {entry.get('id') or entry.get('name')}"
                for pid, _provider, entry in members]
        if explicit_match:
            rows.append(f"explicit: {explicit_connection.model} (LOKI_* connection)")
        return ModelCommandResult("\n".join(diagnostics + rows))
    choices = [member for member in members if member[0] == provider]
    if len(choices) != 1:
        raise ValueError(
            "Provider/model selection is unavailable or ambiguous; "
            "use /providers MODEL and select an exact model ID")
    return ModelCommandResult("", choices[0])
