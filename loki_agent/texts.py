"""Frontend-neutral handling of logical text."""

from pprint import pformat


_SINGLE_LINE_CONTROL_TRANSLATIONS = {
    code: "^" + chr(code + 0x40)
    for code in range(0x20)
}
_SINGLE_LINE_CONTROL_TRANSLATIONS[0x7f] = "^?"
_SINGLE_LINE_CONTROL_TRANSLATIONS.update({
    code: f"\\x{code:02x}"
    for code in range(0x80, 0xa0)
})
_MULTILINE_CONTROL_TRANSLATIONS = dict(
    _SINGLE_LINE_CONTROL_TRANSLATIONS)
del _MULTILINE_CONTROL_TRANSLATIONS[ord("\n")]


def escape_terminal_text(text: str, *, multiline: bool) -> str:
    """Neutralize terminal controls without classifying other Unicode."""
    if not isinstance(text, str):
        raise TypeError("terminal text must be a string")
    translations = (
        _MULTILINE_CONTROL_TRANSLATIONS
        if multiline else _SINGLE_LINE_CONTROL_TRANSLATIONS)
    return text.translate(translations)


TOOL_ARG_INDENT = "    "


def format_multiline_string(value, indent=""):
    """Render each physical line as a normal single-line string literal.

    ``splitlines(keepends=True)`` keeps every line's trailing newline, so
    ``repr`` writes it as ``\\n`` inside the literal while keeping the
    literal a one-line, valid Python string; the literals are joined with
    real newlines so the argument reads line by line.  Over-long lines
    are left for the terminal to soft-wrap.
    """
    return ("\n" + indent).join(
        repr(line) for line in value.splitlines(keepends=True))


def format_tool_arg(name, value):
    """Render one tool argument, ``name: value``, for display."""
    head = f"{name}: "
    if isinstance(value, str) and "\n" in value:
        # Indent continuation lines under the opening value.
        rendered = format_multiline_string(value, " " * len(head))
    elif isinstance(value, list):
        # Keep the brackets and commas, but break after each top-level
        # comma so items wrap one per line instead of one long repr.
        rendered = "[" + (",\n" + " " * (len(head) + 1)).join(
            pformat(item, width=10000) for item in value) + "]"
    else:
        # A huge width stops pformat hard-wrapping; the terminal
        # soft-wraps instead.
        rendered = pformat(value, width=10000)
    return head + rendered


def indent_tool_arg(text):
    """Indent every physical line of one rendered tool argument.

    Continuation lines produced by ``format_tool_arg`` already carry their
    own relative alignment, so prefixing the same base indent to the first
    and every wrapped line keeps them aligned under the value.
    """
    return "\n".join(
        TOOL_ARG_INDENT + line if line else line
        for line in text.split("\n"))


def format_tool_args(args):
    """Render a tool call's arguments the way the terminal displays them."""
    if not isinstance(args, dict):
        return indent_tool_arg(pformat(args, width=10000))
    return "\n".join(
        indent_tool_arg(format_tool_arg(name, value))
        for name, value in args.items())
