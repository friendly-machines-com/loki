"""Names vulture must treat as used, grouped by why they are not dead.

Run the dead-code gate with:

    python3 -m vulture

``[tool.vulture]`` in pyproject.toml feeds this file to vulture together
with the package, so anything vulture reports beyond this list is dead.
"""

# Kept APIs: batch and plain-text renderers used as independent oracles.
render_markdown
status_text

# Entry points for the chat's start-up container verification.  The Windows gate
# that calls them is not wired yet; they live in the read-only module so the
# runtime has no built-in grant path.
verify_workspace
probe_containment

# Container identity and launch declarations.  The launcher and the runtime gate
# that will use these are not written yet, so nothing calls them.
PROCESS_QUERY_LIMITED_INFORMATION
close_handle
open_process_token
token_is_app_container
token_app_container_sid
resume_thread
terminate_process
create_process_in_app_container
_.cb
_.lpAttributeList

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
