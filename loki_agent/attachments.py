"""Local image attachments staged for a later prompt.

Both fronts stage images: the terminal via ``/image`` and the ACP worker via
the same command arriving as prompt text.  Reading, sniffing and bounding the
bytes is front-independent, so it lives here once; only the presentation of
the resulting ``StagedImage`` differs.
"""

from __future__ import annotations

import base64
import os
import shlex
import stat
from dataclasses import dataclass

from .loki import current_cwd, display_path


IMAGE_ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024


class ImageAttachmentError(ValueError):
    pass


@dataclass(frozen=True)
class StagedImage:
    path: str
    media_type: str
    encoded_data: str
    byte_size: int

    def content_block(self) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.encoded_data,
            },
        }


def image_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (len(data) >= 12
            and data.startswith(b"RIFF")
            and data[8:12] == b"WEBP"):
        return "image/webp"
    return None


def image_argument_path(argument: str) -> str:
    """Parse the single path argument of an image command."""
    try:
        parts = shlex.split(argument.strip())
    except ValueError as error:
        raise ImageAttachmentError(f"invalid path quoting: {error}") from error
    if len(parts) != 1:
        raise ImageAttachmentError(
            "usage: /image PATH (quote a path containing spaces)")
    return parts[0]


def load_image_attachment(path_text: str, *,
                          base_dir: str | None = None,
                          max_bytes: int | None = None) -> StagedImage:
    """Read one local image snapshot for a later prompt."""
    limit = IMAGE_ATTACHMENT_MAX_BYTES if max_bytes is None else max_bytes
    if limit < 1:
        raise ValueError("image attachment limit must be positive")

    path = path_text
    if not os.path.isabs(path):
        path = os.path.join(base_dir or current_cwd(), path)

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    fd = None
    try:
        fd = os.open(path, flags)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ImageAttachmentError(
                f"not a regular file: {display_path(path)}")
        if file_stat.st_size > limit:
            raise ImageAttachmentError(
                f"image is {file_stat.st_size} bytes; maximum is "
                f"{limit} bytes")
        with os.fdopen(fd, "rb") as image_file:
            fd = None
            data = image_file.read(limit + 1)
    except ImageAttachmentError:
        raise
    except (OSError, ValueError) as error:
        detail = getattr(error, "strerror", None) or str(error)
        raise ImageAttachmentError(
            f"cannot read {display_path(path)}: {detail}") from error
    finally:
        if fd is not None:
            os.close(fd)

    if len(data) > limit:
        raise ImageAttachmentError(
            f"image exceeds the {limit}-byte maximum")
    media_type = image_media_type(data)
    if media_type is None:
        raise ImageAttachmentError(
            "unsupported image data; expected PNG, JPEG, GIF, or WebP")
    return StagedImage(
        path=path,
        media_type=media_type,
        encoded_data=base64.b64encode(data).decode("ascii"),
        byte_size=len(data),
    )
