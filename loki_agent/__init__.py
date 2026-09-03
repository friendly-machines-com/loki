"""loki-agent package."""


# Flit reads this value for distribution metadata, and the HTTP transport uses
# the same value for Loki's application identity. Keep one version authority so
# installed packages and their outbound User-Agent cannot drift apart.
__version__ = "0.1.0"
