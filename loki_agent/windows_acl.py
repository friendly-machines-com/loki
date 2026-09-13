"""Pure reading of a Windows DACL in SDDL form.

A DACL reaches this module as SDDL text: ``windows_api.dacl_sddl`` converts a
path's descriptor, and a handle's descriptor is converted the same way.  This
module answers what that string says -- which trustees an ACE names, what access
mask it confers -- and makes no OS call, so it runs and is tested on every
platform.

It is deliberately separate from ``windows_state``: that module owns the grant
*policy* (the read/read-write levels, appending and removing Loki's own ACE, the
ledger), while this one only parses.  The credential layer reads a DACL to
decide whether a directory or file is private, and must not pull in the
workspace/AppContainer policy to do it; the workspace side must not grow a
second parser that could disagree about what a string means.

Refusal is the rule for anything unsupported: these results decide whether a
grant is present, so an ACE or rights string this module does not understand
raises instead of being silently under-counted into a false "no access".
"""

from __future__ import annotations

import re

from . import windows_api


# SDDL file-object rights strings (Ace Strings reference) and their masks.
# The numeric rights come from ``windows_api`` so the probe and this table
# cannot disagree about what a right means; the generic codes are expanded
# here to the file rights the kernel maps them to, so an ACE that literally
# stores ``GR`` is judged by the access it actually confers.
_RIGHTS = {
    "FA": windows_api.FILE_ALL_ACCESS,
    "FR": windows_api.FILE_GENERIC_READ,
    "FW": windows_api.FILE_GENERIC_WRITE,
    "FX": windows_api.FILE_GENERIC_EXECUTE,
    "GA": windows_api.FILE_ALL_ACCESS,
    "GR": windows_api.FILE_GENERIC_READ,
    "GW": windows_api.FILE_GENERIC_WRITE,
    "GX": windows_api.FILE_GENERIC_EXECUTE,
    "RC": windows_api.READ_CONTROL,
    "SD": windows_api.DELETE,
    "WD": windows_api.WRITE_DAC,
    "WO": windows_api.WRITE_OWNER,
    "SY": windows_api.SYNCHRONIZE,
    "AS": windows_api.ACCESS_SYSTEM_SECURITY,
}


def rights_mask(rights: str) -> int:
    """Parse an SDDL rights field to a numeric access mask.

    Refuse an unknown string instead of under-counting it: this result decides
    whether a grant is present, so guessing low would turn a real grant into a
    false "no access".
    """
    if rights[:2].lower() == "0x":
        try:
            return int(rights, 16)
        except ValueError as error:
            raise windows_api.WindowsApiError(
                f"malformed SDDL rights {rights!r}") from error
    if not rights or len(rights) % 2:
        raise windows_api.WindowsApiError(
            f"unsupported SDDL rights {rights!r}")
    mask = 0
    for index in range(0, len(rights), 2):
        code = rights[index:index + 2]
        if code not in _RIGHTS:
            raise windows_api.WindowsApiError(
                f"unsupported SDDL rights {rights!r}")
        mask |= _RIGHTS[code]
    return mask


def dacl_parts(sddl: str):
    """Parse basic DACL SDDL; refuse unsupported forms rather than corrupt them."""
    header, separator, rest = sddl.partition("(")
    if not re.fullmatch(r"D:(?:P|AI|AR)*", header):
        raise windows_api.WindowsApiError("unsupported DACL header")
    body = separator + rest
    aces = re.findall(r"\([^()]*\)", body)
    if ''.join(aces) != body or any(len(ace[1:-1].split(';')) != 6 for ace in aces):
        raise windows_api.WindowsApiError("unsupported or malformed DACL ACE")
    return header, aces


def ace_fields(sddl: str):
    """Yield ``(kind, flags, rights, object, inherited_object, sid)`` per ACE.

    Naming an ACE's fields lets a caller tell an allow from a deny and a SID
    from a rights string without assuming basic (non-object) ACEs: object and
    inherited-object GUIDs occupy the same columns, and the SID is always last.
    """
    _header, aces = dacl_parts(sddl)
    for ace in aces:
        yield tuple(ace[1:-1].split(';'))


def flag_codes(flags: str) -> set[str]:
    """Split an ACE-flags field into its two-character codes."""
    if len(flags) % 2:
        raise windows_api.WindowsApiError(f"unsupported ACE flags {flags!r}")
    return {flags[index:index + 2] for index in range(0, len(flags), 2)}


def ace_mask(kind, rights, obj, inherited):
    """The access mask an ACE grants or denies, or ``None`` if it does neither.

    Object ACEs with a GUID restrict the entry to a property or object type,
    which this check cannot evaluate; refuse rather than assume the mask
    applies unqualified.
    """
    if kind in ("A", "D"):
        return rights_mask(rights)
    if kind in ("OA", "OD"):
        if obj or inherited:
            raise windows_api.WindowsApiError(
                "cannot evaluate an object ACE for the package SID")
        return rights_mask(rights)
    return None


def package_access(sddl: str, package: str) -> int:
    """The rights ``sddl`` actually confers on ``package``.

    This is the ACL's own answer for one trustee: the union of the allow masks
    that apply to the object, minus the union of the deny masks that apply to
    it.  An inherit-only ACE (``IO``) applies only to children and is not
    counted; an inherited ACE (``ID``) does apply and is.  Audit, alarm and
    mandatory-label entries neither grant nor deny and are skipped.  Group
    membership and implicit owner rights are outside this computation, which
    answers only what the DACL says about this SID.
    """
    allowed = denied = 0
    for kind, flags, rights, obj, inherited, sid in ace_fields(sddl):
        if sid != package or "IO" in flag_codes(flags):
            continue
        mask = ace_mask(kind, rights, obj, inherited)
        if mask is None:
            continue
        if kind in ("D", "OD"):
            denied |= mask
        else:
            allowed |= mask
    return allowed & ~denied


def package_allow(sddl: str, package: str) -> int:
    """The allow rights any ACE naming ``package`` could confer.

    Unlike :func:`package_access`, inherit-only allows are counted: an entry
    that grants only children still hands the package the contents of a
    protected tree, so the private-tree check must not read the tree's own
    DACL as clearing an inheritable grant.  Deny entries alone contribute
    nothing, which is what keeps a deny-only DACL from being reported as a
    grant.
    """
    allowed = 0
    for kind, _flags, rights, obj, inherited, sid in ace_fields(sddl):
        if sid != package:
            continue
        mask = ace_mask(kind, rights, obj, inherited)
        if mask is not None and kind in ("A", "OA"):
            allowed |= mask
    return allowed


def allow_trustees(sddl: str, include_inherit_only: bool = False) -> set[str]:
    """The SIDs an allow ACE confers access on for the object itself.

    Inherit-only entries (``IO``) are excluded by default: they apply to
    children, not to this object, and each child restates them as its own
    inherited ACE, which a check on the child's handle then reads.  A directory
    that grants its *children* to someone is still not private, so a caller
    asking about a directory passes ``include_inherit_only=True`` and sees the
    trustees its contents would inherit.  Deny and audit entries confer
    nothing.  Group membership and implicit owner rights are not resolved --
    this is the DACL's own list of trustees, which is what a "is this shared"
    check needs.
    """
    trustees = set()
    for kind, flags, rights, obj, inherited, sid in ace_fields(sddl):
        if not include_inherit_only and "IO" in flag_codes(flags):
            continue
        if kind in ("A", "OA") and ace_mask(kind, rights, obj, inherited) is not None:
            trustees.add(sid)
    return trustees


def names_package(sddl: str, package: str) -> bool:
    """Whether any ACE in ``sddl`` names ``package``, allow or deny.

    Presence only, for deciding whether an edit has anything to touch; it says
    nothing about access, which is :func:`package_access`'s question.
    """
    return any(sid == package
               for _kind, _flags, _rights, _object, _inherited, sid
               in ace_fields(sddl))
