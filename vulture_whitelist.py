"""Names vulture must treat as used, grouped by why they are not dead.

Run the dead-code gate with:

    python3 -m vulture

``[tool.vulture]`` in pyproject.toml feeds this file to vulture together
with the package, so anything vulture reports beyond this list is dead.
"""

# Kept APIs: batch and plain-text renderers used as independent oracles.
render_markdown
status_text

# Test seams: production never reads these; the tests exercise them.
_.credential_broker
_.from_fd
_.lock_path
_.mode_cycle_requested
_.message_count
_.retained_characters
_.has_custom_hooks
_._mode

# Intentional protocol vocabulary and terminal interface surface.
event_id
REPEAT
ESCAPE
_.clear_screen
_.__stdout__

# False positives: base-class callbacks and destructuring targets.
_.handle_starttag
_.handle_endtag
_.handle_data
x
iflag
oflag
cflag
lflag
ispeed
ospeed
