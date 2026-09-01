"""Frontend-neutral handling of logical text."""


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
