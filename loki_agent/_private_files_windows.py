"""Windows credential-file primitives: handle-relative, reparse-refusing.

The POSIX module's two guarantees are reproduced here with the native calls
the storage investigation measured:

* **No reparse point is followed.**  Every child is opened with
  ``FILE_OPEN_REPARSE_POINT``, so the object acted on is the link itself; the
  storage then refuses it because :class:`FileFacts` reports ``reparse_point``.
  ``open_directory`` additionally refuses a reparse-point directory outright,
  because a handle to one would still traverse an intermediate link.
* **Every operation is bound to the retained directory handle.**  Children are
  opened with ``NtCreateFile(RootDirectory=...)`` and renamed with
  ``NtSetInformationFile``, not by re-resolving the directory's pathname.  This
  is the Windows equivalent of ``dir_fd``; ``SetFileInformationByHandle`` with a
  relative name is the form that returned ``ERROR_INVALID_PARAMETER`` in the
  investigation, so it is not used.

The token is a raw ``HANDLE`` (a POSIX descriptor is an ``int``), which is why
``private_files`` takes read/write/fsync/close from this module too.

What the privacy check does and does not cover is stated at
``_PRIVATE_TRUSTEES``.
"""

from __future__ import annotations

from . import windows_acl, windows_api
from .credential_errors import CredentialStorageError
from .file_facts import FileFacts


# Trustees that may hold access to a private credential object besides its
# owner: the local SYSTEM (S-1-5-18) and the built-in Administrators group
# (S-1-5-32-544).  Both can take ownership of any object on the volume, and
# Windows bakes them into the DACL a private directory is created with, so
# refusing on their presence would refuse every real machine while buying
# nothing -- the same reasoning that calls read-versus-execute "theater" in
# ``windows_state``.
#
# This is deliberately the coarse analogue of the POSIX check
# (``st_mode & 0o077``), not an access-control proof.  It reads the DACL only:
# it does NOT examine the SACL or the mandatory integrity label (where the
# AppContainer lowbox label lives), privileges (SeBackupPrivilege,
# SeTakeOwnershipPrivilege), or a same-user process.  Group membership and
# inherited-from-above ACEs are seen as the DACL states them, not resolved.
#
# The two owner spellings are here because that is how the kernel describes a
# private directory: ``os.mkdir(mode=0o700)`` produces
# ``D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)``, measured on the
# Windows runners.  ``OW`` (OWNER RIGHTS, S-1-3-4) grants the object's owner,
# and ``CO`` (CREATOR OWNER, S-1-3-0) the creator on inheritance, so both are
# the user, not a third party.
_PRIVATE_TRUSTEES = frozenset({
    "S-1-5-18",      # LOCAL SYSTEM
    "S-1-5-32-544",  # BUILTIN\Administrators
    "S-1-3-0",       # CREATOR OWNER: the object's creator, i.e. the user
    "S-1-3-4",       # OWNER RIGHTS: the object's current owner, i.e. the user
})

# NTSTATUS/Win32 error values that map onto the OSError subclasses the storage
# protocol branches on; everything else becomes a plain OSError.
_MISSING = frozenset({
    windows_api.STATUS_OBJECT_NAME_NOT_FOUND,
    windows_api.STATUS_OBJECT_PATH_NOT_FOUND,
    windows_api.ERROR_FILE_NOT_FOUND,
    windows_api.ERROR_PATH_NOT_FOUND,
})
_DENIED = frozenset({
    windows_api.STATUS_ACCESS_DENIED,
    windows_api.ERROR_ACCESS_DENIED,
})


def _check_name(name: str) -> None:
    """Refuse anything that is not a single path component.

    ``RootDirectory`` binds the open to the retained directory, but a name
    containing a separator would still walk past it; the names the storage uses
    are constants, and this keeps it that way.
    """
    if not name or name in (".", "..") or any(
            part in name for part in ("/", "\\", ":")):
        raise CredentialStorageError(
            f"credential file name is not a single path component: {name!r}")


def _raise_oserror(error: windows_api.WindowsApiError) -> None:
    """Translate a failed call into the OSError the storage protocol expects."""
    status = error.status
    message = str(error)
    if status is None:
        raise OSError(message) from error
    unsigned = status & 0xffffffff
    if unsigned in _MISSING:
        raise FileNotFoundError(message) from error
    if unsigned == windows_api.STATUS_OBJECT_NAME_COLLISION:
        raise FileExistsError(message) from error
    if unsigned in _DENIED:
        raise PermissionError(message) from error
    if unsigned == windows_api.STATUS_NOT_A_DIRECTORY:
        raise NotADirectoryError(message) from error
    raise OSError(message) from error


def _is_shared(sddl, owner: str, inheritable: bool = False) -> bool:
    """Whether the DACL grants access to anyone beyond the allowed trustees.

    Fails closed in every direction that matters: a missing DACL (which grants
    everyone), a trustee ``canonical_sid`` cannot classify (it refuses rather
    than guesses), and any allow ACE naming an unlisted trustee all answer
    "shared", so the storage refuses the object instead of trusting it.

    ``inheritable`` is set when describing a directory: its contents inherit
    its ACEs, so a grant that applies only to children still reaches the
    credential JSON that will be created there.  For a file it is the object's
    own access that counts.
    """
    if sddl is None:
        # No DACL at all grants everyone full access; never "private".
        return True
    allowed = {windows_api.canonical_sid(sid)
               for sid in (_PRIVATE_TRUSTEES | {owner})}
    trustees = {windows_api.canonical_sid(sid) for sid in
                windows_acl.allow_trustees(
                    sddl, include_inherit_only=inheritable)}
    return not trustees <= allowed


def _facts_from_handle(handle) -> FileFacts:
    """Describe an open object the way the storage needs it.

    The object's facts come from the handle, so a rename between the open and
    the query cannot change which object is described.
    """
    try:
        information = windows_api.by_handle_file_information(handle)
        owner = windows_api.handle_owner_sid(handle)
        dacl = windows_api.handle_dacl_sddl(handle)
        current_user = windows_api.current_user_sid()
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    attributes = information.dwFileAttributes
    directory = bool(
        attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY)
    return FileFacts(
        # Windows has no device/pipe distinction here; a non-directory is the
        # regular-file equivalent, and a reparse point is refused separately.
        regular=not directory,
        directory=directory,
        reparse_point=bool(
            attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_REPARSE_POINT),
        size=(information.nFileSizeHigh << 32) | information.nFileSizeLow,
        owned_by_current_user=(owner == current_user),
        group_or_other_access=_is_shared(dacl, owner, inheritable=directory),
    )


def _open_relative(directory, name, desired_access, disposition, options,
                   attributes=0):
    try:
        return windows_api.nt_create_file(
            directory, name, desired_access, disposition, options, attributes)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)


# FILE_SYNCHRONOUS_IO_NONALERT makes every read/write return synchronously, so
# the storage never has to wait on an IO_STATUS_BLOCK it did not provide.
_SYNCHRONOUS = windows_api.FILE_SYNCHRONOUS_IO_NONALERT
_NO_FOLLOW = windows_api.FILE_NON_DIRECTORY_FILE | windows_api.FILE_OPEN_REPARSE_POINT
_NON_DIRECTORY = _NO_FOLLOW | _SYNCHRONOUS


# -- descriptor operations ------------------------------------------------

def read(handle, size):
    try:
        return windows_api.read_file(handle, size)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)


def write(handle, data):
    try:
        return windows_api.write_file(handle, data)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)


def fsync(handle) -> None:
    # Windows has no directory fsync: FlushFileBuffers needs write access and a
    # directory cannot be opened for writing, so a directory handle is a
    # deliberate no-op here.  The rename that publishes the JSON is a metadata
    # operation the filesystem journals; the kill-at-each-checkpoint experiment
    # is the evidence that the published file is whole old or whole new.
    try:
        information = windows_api.by_handle_file_information(handle)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    if information.dwFileAttributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY:
        return
    try:
        windows_api.flush_file(handle)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)


def close(handle) -> None:
    windows_api.close_handle(handle)


# -- opens ----------------------------------------------------------------

def open_directory(path: str):
    try:
        handle = windows_api.open_with_access(
            path, windows_api.AccessMask.GENERIC_READ,
            flags=(windows_api.FileFlags.FILE_FLAG_BACKUP_SEMANTICS
                   | windows_api.FILE_OPEN_REPARSE_POINT))
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    try:
        information = windows_api.by_handle_file_information(handle)
    except windows_api.WindowsApiError as error:
        windows_api.close_handle(handle)
        _raise_oserror(error)
    attributes = information.dwFileAttributes
    # Only the mechanism is checked here: a root has to be a directory at
    # all.  Whether a reparse-point directory is acceptable is policy, and
    # the caller decides it from describe() on the returned handle.
    if not attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY:
        windows_api.close_handle(handle)
        raise NotADirectoryError(path)
    return handle


def open_read_at(directory, name: str):
    _check_name(name)
    return _open_relative(
        directory, name,
        windows_api.AccessMask.GENERIC_READ | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.FILE_OPEN, _NON_DIRECTORY)


def create_exclusive_at(directory, name: str, mode: int):
    _check_name(name)
    # ``mode`` is a POSIX concept.  On POSIX the mode makes the new file
    # private regardless of the directory; here the file's access comes from the
    # directory's DACL by inheritance, and the directory was checked private.
    # The file is what holds the secret, though, and its DACL is fixed at
    # creation, so describe *this handle* before any byte is written: a
    # directory widened between the check and the create would otherwise hand
    # the file out.  Fails closed -- anything but a private, owned, regular file
    # refuses.
    # The new file's privacy is the caller's decision: it describes the
    # handle it was given before writing anything.
    return _open_relative(
        directory, name,
        windows_api.AccessMask.GENERIC_WRITE
        | windows_api.AccessMask.FILE_READ_ATTRIBUTES
        | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.FILE_CREATE, _NON_DIRECTORY,
        windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL)


def open_lock_file_at(directory, name: str, mode: int):
    _check_name(name)
    return _open_relative(
        directory, name,
        windows_api.AccessMask.GENERIC_READ
        | windows_api.AccessMask.GENERIC_WRITE
        | windows_api.AccessMask.FILE_READ_ATTRIBUTES
        | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.FILE_OPEN_IF, _NON_DIRECTORY)


# -- publish and remove ---------------------------------------------------

def replace_at(directory, temporary: str, name: str) -> None:
    _check_name(temporary)
    _check_name(name)
    # Delete access on the source is what a rename needs; the source handle
    # names the object even if its entry has been swapped meanwhile.
    source = _open_relative(
        directory, temporary,
        windows_api.AccessMask.DELETE | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.FILE_OPEN, _NON_DIRECTORY)
    try:
        try:
            windows_api.nt_rename(source, directory, name)
        except windows_api.WindowsApiError as error:
            _raise_oserror(error)
    finally:
        windows_api.close_handle(source)


def unlink_at(directory, name: str) -> None:
    _check_name(name)
    # Opened reparse-point-refusing, so a link is removed rather than its
    # target; delete is disposition-on-close, performed when the handle closes.
    target = _open_relative(
        directory, name,
        windows_api.AccessMask.DELETE | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.FILE_OPEN, _NON_DIRECTORY)
    try:
        try:
            windows_api.set_delete_disposition(target)
        except windows_api.WindowsApiError as error:
            _raise_oserror(error)
    finally:
        windows_api.close_handle(target)


# -- description ----------------------------------------------------------

def describe(handle) -> FileFacts:
    return _facts_from_handle(handle)


def describe_path(path: str) -> FileFacts:
    try:
        handle = windows_api.open_with_access(
            path, windows_api.AccessMask.GENERIC_READ,
            flags=(windows_api.FileFlags.FILE_FLAG_BACKUP_SEMANTICS
                   | windows_api.FILE_OPEN_REPARSE_POINT))
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    try:
        return _facts_from_handle(handle)
    finally:
        windows_api.close_handle(handle)
