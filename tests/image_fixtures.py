"""A real one-pixel PNG for Read snapshot, presentation and request tests.

Signature-only bytes suffice for MIME detection tests, but do not demonstrate
that ordinary image-file data survives a complete tool continuation.
"""

import base64


# 1x1, 8-bit RGB red pixel: filter byte 0 followed by ff 00 00. Both PNG
# chunk CRCs and the zlib stream checksum are valid, unlike signature fixtures.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
