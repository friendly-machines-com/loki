"""Test support: treat catalog endpoints as already approved.

Endpoint approval has its own tests (tests/test_endpoint_pins.py).  Tests that
exercise unrelated behavior -- picker flow, provider overrides, config building
-- assume every endpoint was approved, so they neither prompt for confirmation
nor refuse a selection.
"""

from unittest import mock

from loki_agent import endpoint_pins


def assume_endpoints_approved(testcase):
    """Make endpoint_pins.status report PINNED for the duration of TESTCASE."""
    patch = mock.patch.object(
        endpoint_pins, "status",
        lambda provider_id, api_url, credential: (
            endpoint_pins.PINNED, None))
    patch.start()
    testcase.addCleanup(patch.stop)
