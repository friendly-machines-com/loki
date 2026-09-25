"""Names vulture must treat as used, grouped by why they are not dead.

Run the dead-code gate with:

    python3 -m vulture

``[tool.vulture]`` in pyproject.toml feeds this file to vulture together
with the package, so anything vulture reports beyond this list is dead.
"""

# Kept APIs: batch and plain-text renderers used as independent oracles.
render_markdown
status_text

# Windows constants tested against the native API; ctypes reads these fields.
PROCESS_QUERY_LIMITED_INFORMATION
# TokenAccess and ProcessAccess are complete sets; production reads only some
# members, the escape probes request the rest on tokens and processes they open
# themselves.
_.TOKEN_ASSIGN_PRIMARY
_.TOKEN_DUPLICATE
_.TOKEN_IMPERSONATE
_.TOKEN_ADJUST_PRIVILEGES
_.TOKEN_ADJUST_DEFAULT
ProcessAccess
_.PROCESS_TERMINATE
_.PROCESS_VM_OPERATION
_.PROCESS_VM_READ
_.PROCESS_VM_WRITE
_.PROCESS_DUP_HANDLE
_.PROCESS_CREATE_PROCESS
_.PROCESS_QUERY_LIMITED_INFORMATION
# FileCreateDisposition is the complete dwCreationDisposition set; Loki opens
# with OPEN_EXISTING only.
_.CREATE_NEW
_.CREATE_ALWAYS
_.OPEN_ALWAYS
_.TRUNCATE_EXISTING
_.cb
_.lpAttributeList
_.dwFlags
_.hStdInput
_.hStdOutput
_.hStdError
# FILE_RENAME_INFORMATION is filled in for NtSetInformationFile; nothing reads
# the fields back on this side, the kernel does.
_.ReplaceIfExists
_.RootDirectory
_.FileNameLength

# Test seams: production never reads these; the tests exercise them.
_.credential_broker
_.from_fd
_.lock_path
_.mode_cycle_requested
_.message_count
_.retained_characters
_.has_custom_hooks
_._mode
# The Ctrl+C tests assert through this seam that TerminalMode really cleared
# interrupt processing; production has no caller, only test_pty_ui does.
interrupt_processing_enabled
# The field partition is asserted against the dataclass by
# tests/test_connections.py rather than read by production code.
DISPLAYED_CONNECTION_FIELDS
UNDISPLAYED_CONNECTION_FIELDS

# Stage 1 of the Windows credential transport: the anonymous-pipe pair has no
# production caller until the Stage 2 endpoint swap consumes it.  The portable
# tests exercise it now, so it is not dead -- removing the wire-up later would
# not be silent, because this entry and the tests name it.
_private_pipe_pair

# Intentional protocol vocabulary and terminal interface surface.
event_id
REPEAT
ESCAPE
_.clear_screen
_.__stdout__

# Vendored asyncio protocol/transport surface in windows_subprocesses: these
# are called by the event loop, by StreamWriter and by the copied base class,
# never by a Loki call site.  `_log_traceback` is the CPython attribute set on
# the stdin close waiter so an unawaited wait_closed() logs no traceback.
_.connection_made
_.connection_lost
_.data_received
_.eof_received
_.pause_writing
_.resume_writing
_.set_protocol
_.get_protocol
_.is_closing
_.pipe_data_received
_.pipe_connection_lost
_.process_exited
_._drain_helper
_._get_close_waiter
_._log_traceback

# Read by the workspace-access diagnostic in tests/loki_entrypoints.py, which
# is outside the vulture paths: the mandatory label is what a low-integrity
# AppContainer meets before the DACL, so the diagnostic has to print it.
label_sddl

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
