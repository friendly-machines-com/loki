"""Native Landlock binding for Linux ordinary tool launches.

The tool bootstrap needs this scope because the mount and IPC namespaces do
not separate abstract Unix addresses in a shared network namespace. Keep the
native ABI and syscall choices here: distributors can replace this binding
without changing the tool launch protocol or its authorization decision.
"""

import ctypes
import errno
import os
import sys


# Linux v6.12 UAPI: the common syscall table and the architecture syscall
# tables share this pair. MIPS uses the same rows plus its ABI's __NR_Linux
# base. This is a process ABI decision, not a kernel-machine (uname) test.
_CREATE_RULESET = 444
_RESTRICT_SELF = 446
_CREATE_RULESET_VERSION = 1
_SCOPE_ABSTRACT_UNIX_SOCKET = 1

_SHARED = {
    4: {'i386', 'i486', 'i586', 'i686', 'riscv32', 'powerpc', 'powerpcle', 's390', 'hppa', 'sparc',
        'arc', 'arceb', 'csky', 'hexagon', 'm68k', 'microblaze', 'microblazeel', 'nios2', 'nios2eb',
        'or1k', 'openrisc', 'sh', 'sh3', 'sh3eb', 'sh4', 'sh4eb', 'xtensa', 'xtensaeb'},
    8: {'x86_64', 'aarch64', 'aarch64_be', 'riscv64', 'loongarch64', 'powerpc64', 'powerpc64le',
        's390x', 'hppa', 'hppa64', 'sparc64'},
}
_ARM_EABI = {'gnueabi', 'gnueabihf', 'musleabi', 'musleabihf'}
_MIPS_O32 = {'mips', 'mipsel', 'mipsisa32r6', 'mipsisa32r6el'}
_MIPS_64 = {'mips64', 'mips64el', 'mipsisa64r6', 'mipsisa64r6el'}


class _Ruleset(ctypes.Structure):
    # The ABI 6 payload is three native-endian u64 fields, even in a 32-bit
    # process. Pointer and size_t widths matter only for the syscall arguments.
    _fields_ = [('handled_access_fs', ctypes.c_uint64),
                ('handled_access_net', ctypes.c_uint64),
                ('scoped', ctypes.c_uint64)]


def _syscall_numbers(tag: str, width: int) -> tuple[int, int]:
    arch, linux, abi = tag.partition('-linux-')
    if linux and abi:
        # In particular, neither an x32 interpreter nor ambiguous MIPS ILP32
        # can use the plain numbers. OABI is not an ARM EABI tag.
        if arch in _SHARED.get(width, set()) and abi != 'gnux32':
            return _CREATE_RULESET, _RESTRICT_SELF
        if width == 4 and arch in {'arm', 'armeb'} and abi in _ARM_EABI:
            return _CREATE_RULESET, _RESTRICT_SELF
        if width == 4 and arch in _MIPS_O32 and abi in {'gnu', 'musl'}:
            base = 4000
        elif width == 8 and arch in _MIPS_64 and abi in {'gnuabi64', 'muslabi64', 'gnu', 'musl'}:
            base = 5000
        elif width == 4 and arch in _MIPS_64 and abi in {'gnuabin32', 'muslabin32'}:
            base = 6000
        else:
            base = None
        if base is not None:
            return base + _CREATE_RULESET, base + _RESTRICT_SELF
    raise OSError(errno.ENOTSUP, 'blocking host abstract Unix sockets is unavailable for this interpreter ABI')


def _checked_call(syscall, number: int, operation: str, *arguments) -> int:
    result = syscall(number, *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, f'cannot block host abstract Unix sockets: {operation}: {os.strerror(error)}')
    return result


def restrict_abstract_unix_sockets() -> None:
    """Restrict the single-threaded tool bootstrap and its descendants.

    State staging already sets no_new_privs and drops capabilities. The
    ruleset fd belongs only to this call; once applied, it can be closed
    without undoing the scope. Never return success without enforcement.
    """
    width = ctypes.sizeof(ctypes.c_void_p)
    if not sys.platform.startswith('linux') or width != ctypes.sizeof(ctypes.c_long):
        raise OSError(errno.ENOTSUP, 'blocking host abstract Unix sockets is unavailable for this interpreter ABI')
    # CPython generates this tag at configure time; PyPy builds it into the
    # translation. We do not guess a raw syscall number if it is absent.
    create, restrict = _syscall_numbers(getattr(sys.implementation, '_multiarch', ''), width)
    if ctypes.sizeof(_Ruleset) != 24:
        raise OSError(errno.ENOTSUP, 'blocking host abstract Unix sockets is unavailable for this interpreter ABI')
    try:
        syscall = ctypes.CDLL(None, use_errno=True).syscall
    except AttributeError as error:
        raise OSError(errno.ENOSYS, 'blocking host abstract Unix sockets requires the Linux Landlock interface') from error
    syscall.argtypes = [ctypes.c_long]  # The remaining arguments are variadic.
    syscall.restype = ctypes.c_long

    abi = _checked_call(syscall, create, 'Landlock availability check failed',
                        ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint32(_CREATE_RULESET_VERSION))
    if abi < 6:
        raise OSError(errno.ENOTSUP, f'blocking host abstract Unix sockets requires Landlock ABI 6 or later (found {abi})')
    attr = _Ruleset(0, 0, _SCOPE_ABSTRACT_UNIX_SOCKET)
    fd = _checked_call(syscall, create, 'Landlock ruleset creation failed',
                       ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
    try:
        _checked_call(syscall, restrict, 'Landlock enforcement failed', ctypes.c_int(fd), ctypes.c_uint32(0))
    finally:
        os.close(fd)
