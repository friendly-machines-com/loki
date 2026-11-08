# pytype: skip-file
"""Windows credential-file primitives: handle-relative, reparse-refusing.

The POSIX module's two guarantees are reproduced here with the native calls
the storage investigation measured:

* **Protected child opens refuse reparse points.** Every child is opened with
  ``FILE_OPEN_REPARSE_POINT``, so the object acted on is the link itself; the
  storage then refuses it because :class:`FileFacts` reports ``reparse_point``.
  State owners validate the opened root; ``open_directory_at`` refuses a
  reparse-point directory before it can become an intermediate lookup root.
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

import ntpath
import os

from . import windows_acl, windows_api
from .credential_errors import CredentialStorageError
from .file_facts import FileFacts, FileInfo


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
    if not name or name in [".", ".."] or any(
            part in name for part in ["/", "\\", ":"]):
        raise CredentialStorageError(
            f"credential file name is not a single path component: {name!r}")


def _raise_oserror(error: windows_api.WindowsApiError) -> None:
    """Translate a failed call into the OSError the storage protocol expects."""
    status = error.status
    message = str(error)
    if isinstance(error, windows_api.WindowsResourceError):
        # A cleanup code is not evidence that acquisition was refused. In
        # particular it must never trigger a missing-parent creation retry.
        raise OSError(message) from error
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


def information(handle) -> FileInfo:
    """Read type and size without applying credential-specific ACL policy."""
    try:
        result = windows_api.by_handle_file_information(handle)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    attributes = result.dwFileAttributes
    directory = bool(attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY)
    return FileInfo(
        regular=not directory,
        directory=directory,
        reparse_point=bool(attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_REPARSE_POINT),
        size=(result.nFileSizeHigh << 32) | result.nFileSizeLow,
    )


def _facts_from_handle(handle) -> FileFacts:
    """The credential store additionally requires owner and DACL validation."""
    info = information(handle)
    try:
        owner = windows_api.handle_owner_sid(handle)
        dacl = windows_api.handle_dacl_sddl(handle)
        current_user = windows_api.current_user_sid()
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    return FileFacts(
        **vars(info),
        owned_by_current_user=(owner == current_user),
        group_or_other_access=_is_shared(dacl, owner, inheritable=info.directory),
    )


def _open_relative(directory, name, desired_access, disposition, options,
                   attributes=0, *, user_owned=False, private=False):
    creation = {}
    try:
        if user_owned or private:
            user = windows_api.current_user_sid()
            creation['security_descriptor'] = 'O:' + user
            if private:
                # Setup authority files must not inherit runtime grants from
                # their shared state parent. Setting just ownership retains its
                # DACL; FILE_OPEN_IF never repairs existing permissions.
                creation['security_descriptor'] += windows_acl.private_dacl_sddl(user)
        return windows_api.nt_create_file(
            directory, name, desired_access, disposition, options, attributes, **creation)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)


# FILE_SYNCHRONOUS_IO_NONALERT makes every read/write return synchronously, so
# the storage never has to wait on an IO_STATUS_BLOCK it did not provide.
_SYNCHRONOUS = windows_api.NtCreateOptions.FILE_SYNCHRONOUS_IO_NONALERT
_NO_FOLLOW = (windows_api.NtCreateOptions.FILE_NON_DIRECTORY_FILE
              | windows_api.NtCreateOptions.FILE_OPEN_REPARSE_POINT)
_NON_DIRECTORY = _NO_FOLLOW | _SYNCHRONOUS


# -- descriptor operations ------------------------------------------------

def read(handle, size):
    try:
        return windows_api.read_file(handle, size)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)


def binary_file(handle, mode):
    """Consume a native HANDLE as a standard Python binary stream.

    Relative opens return HANDLEs, but fdopen and subprocess fileno consumers
    require CRT descriptors. Convert without reopening the selected pathname:
    each stage closes only the representation it owns, including on failure.
    A successful Python file is the sole close owner afterward.
    """
    import msvcrt

    try:
        fd = msvcrt.open_osfhandle(
            handle, os.O_BINARY | os.O_NOINHERIT | (os.O_RDONLY if mode == 'rb' else os.O_WRONLY))
    except BaseException:
        close(handle)
        raise
    try:
        return os.fdopen(fd, mode)
    except BaseException:
        os.close(fd)
        raise


def seek(handle, offset, whence):
    try:
        return windows_api.set_file_pointer(handle, offset, whence)
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
    # Storage owns one release attempt, just like os.close; never retry a
    # native numeric handle whose validity is unknown after failure.
    try:
        windows_api.close_handle(handle)
    except windows_api.WindowsApiError as error:
        # Closing is not an acquisition result, regardless of its error code.
        raise OSError(str(error)) from error


# -- opens ----------------------------------------------------------------

def make_private_directory(path: str) -> None:
    """Create missing private user-owned objects; leave existing ones unchanged.

    Try the exact native operand first. Only a missing parent requires splitting
    it for recursive creation; the final CreateDirectoryW operand is unchanged.
    """
    try:
        user = windows_api.current_user_sid()
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    descriptor = 'O:' + user + windows_acl.private_dacl_sddl(user)

    def create(name):
        try:
            windows_api.create_directory(name, security_descriptor=descriptor)
        except windows_api.WindowsApiError as error:
            if (isinstance(error, windows_api.WindowsResourceError)
                    or error.status != windows_api.ERROR_PATH_NOT_FOUND):
                raise
            parent, _leaf = ntpath.split(name)
            if not parent or parent == name:
                raise
            create(parent)
            windows_api.create_directory(name, security_descriptor=descriptor)

    try:
        create(path)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)


def open_directory(path: str, *, follow_symlinks=False):
    # Preserve protected-root selection by default. Ordinary file publishers
    # request native parent lookup explicitly, without rewriting the pathname.
    flags = windows_api.FileFlags.FILE_FLAG_BACKUP_SEMANTICS
    if not follow_symlinks:
        flags |= windows_api.NtCreateOptions.FILE_OPEN_REPARSE_POINT
    try:
        handle = windows_api.open_with_access(
            path, windows_api.AccessMask.GENERIC_READ, flags=flags)
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    try:
        information = windows_api.by_handle_file_information(handle)
    except windows_api.WindowsApiError as error:
        close(handle)
        _raise_oserror(error)
    attributes = information.dwFileAttributes
    # Only the mechanism is checked here: a root has to be a directory at
    # all.  Whether a reparse-point directory is acceptable is policy, and
    # the caller decides it from describe() on the returned handle.
    if not attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY:
        close(handle)
        raise NotADirectoryError(path)
    return handle


def open_read_at(directory, name: str):
    _check_name(name)
    return _open_relative(
        directory, name,
        windows_api.AccessMask.GENERIC_READ | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.NtCreateDisposition.FILE_OPEN, _NON_DIRECTORY)


def create_exclusive_at(directory, name: str, mode: int, *, permission_control=False, user_owned=False):
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
    access = (windows_api.AccessMask.GENERIC_WRITE | windows_api.AccessMask.FILE_READ_ATTRIBUTES
              | windows_api.AccessMask.SYNCHRONIZE)
    if permission_control:
        # Editor publication copies the destination's DACL onto this object.
        # Ordinary state/credential creation and inherited job captures must
        # not acquire that additional authority merely because they can write.
        access |= windows_api.AccessMask.WRITE_DAC
    return _open_relative(
        directory, name, access,
        windows_api.NtCreateDisposition.FILE_CREATE, _NON_DIRECTORY,
        windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL, user_owned=user_owned)


def create_private_at(directory, name: str):
    """Exclusively create a private user-owned file, even in a shared parent."""
    _check_name(name)
    return _open_relative(
        directory, name, windows_api.AccessMask.GENERIC_WRITE | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.NtCreateDisposition.FILE_CREATE, _NON_DIRECTORY,
        windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL, private=True)


def open_lock_file_at(directory, name: str, mode: int, *, user_owned=False):
    _check_name(name)
    return _open_relative(
        directory, name,
        windows_api.AccessMask.GENERIC_READ
        | windows_api.AccessMask.GENERIC_WRITE
        | windows_api.AccessMask.FILE_READ_ATTRIBUTES
        | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.NtCreateDisposition.FILE_OPEN_IF, _NON_DIRECTORY, user_owned=user_owned)


def _directory_result(handle):
    try:
        try:
            attributes = windows_api.by_handle_file_information(handle).dwFileAttributes
        except windows_api.WindowsApiError as error:
            _raise_oserror(error)
        if attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError("state directory is a reparse point")
        if not attributes & windows_api.FileAttribute.FILE_ATTRIBUTE_DIRECTORY:
            raise NotADirectoryError("state path is not a directory")
    except BaseException:
        close(handle)
        raise
    return handle


def open_directory_at(directory, name: str):
    _check_name(name)
    # FILE_OPEN_REPARSE_POINT opens the link itself; reject that handle before
    # it can become the RootDirectory of another relative operation.
    return _directory_result(_open_relative(
        directory, name,
        windows_api.AccessMask.GENERIC_READ
        | windows_api.AccessMask.FILE_READ_ATTRIBUTES
        | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.NtCreateDisposition.FILE_OPEN,
        windows_api.NtCreateOptions.FILE_DIRECTORY_FILE
        | windows_api.NtCreateOptions.FILE_OPEN_REPARSE_POINT
        | _SYNCHRONOUS))


def create_directory_at(directory, name: str, mode: int = 0o700):
    _check_name(name)
    # ``mode`` is a POSIX concept; on this platform the new directory's access
    # comes from the parent's DACL by inheritance, exactly as create_exclusive_at
    # documents for files.  OPEN_IF makes an existing directory a success, so
    # every failure reaching the caller is a real one.
    handle = _open_relative(
        directory, name,
        windows_api.AccessMask.FILE_READ_ATTRIBUTES
        | windows_api.AccessMask.SYNCHRONIZE
        | windows_api.AccessMask.READ_CONTROL,
        windows_api.NtCreateDisposition.FILE_OPEN_IF,
        windows_api.NtCreateOptions.FILE_DIRECTORY_FILE
        | windows_api.NtCreateOptions.FILE_OPEN_REPARSE_POINT
        | _SYNCHRONOUS,
        windows_api.FileAttribute.FILE_ATTRIBUTE_NORMAL)
    close(_directory_result(handle))


def open_append_at(directory, name: str):
    _check_name(name)
    # Owner diagnostics append at EOF without changing the child's capture
    # position. This does not restrict the child's independently held handle.
    return _open_relative(
        directory, name,
        windows_api.AccessMask.FILE_APPEND_DATA
        | windows_api.AccessMask.FILE_READ_ATTRIBUTES
        | windows_api.AccessMask.READ_CONTROL
        | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.NtCreateDisposition.FILE_OPEN_IF, _NON_DIRECTORY)


# -- publish and remove ---------------------------------------------------

def replace_at(directory, temporary: str, name: str) -> None:
    _check_name(temporary)
    _check_name(name)
    # Delete access on the source is what a rename needs; the source handle
    # names the object even if its entry has been swapped meanwhile.
    source = _open_relative(
        directory, temporary,
        windows_api.AccessMask.DELETE | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.NtCreateDisposition.FILE_OPEN, _NON_DIRECTORY)
    try:
        try:
            windows_api.nt_rename(source, directory, name)
        except windows_api.WindowsApiError as error:
            _raise_oserror(error)
    finally:
        close(source)


def unlink_at(directory, name: str) -> None:
    _check_name(name)
    # Opened reparse-point-refusing, so a link is removed rather than its
    # target; delete is disposition-on-close, performed when the handle closes.
    target = _open_relative(
        directory, name,
        windows_api.AccessMask.DELETE | windows_api.AccessMask.SYNCHRONIZE,
        windows_api.NtCreateDisposition.FILE_OPEN, _NON_DIRECTORY)
    try:
        try:
            windows_api.set_delete_disposition(target)
        except windows_api.WindowsApiError as error:
            _raise_oserror(error)
    finally:
        close(target)


# -- description ----------------------------------------------------------

def describe(handle) -> FileFacts:
    return _facts_from_handle(handle)


def describe_path(path: str) -> FileFacts:
    try:
        handle = windows_api.open_with_access(
            path, windows_api.AccessMask.GENERIC_READ,
            flags=(windows_api.FileFlags.FILE_FLAG_BACKUP_SEMANTICS
                   | windows_api.NtCreateOptions.FILE_OPEN_REPARSE_POINT))
    except windows_api.WindowsApiError as error:
        _raise_oserror(error)
    try:
        return _facts_from_handle(handle)
    finally:
        close(handle)
