"""Every core test runs under the core's own test host (`core_policy`); a
test about the empty policy installs `EMPTY` itself."""

import pytest

from core_policy import CORE_POLICY
from llm_agents import host_policy


@pytest.fixture(autouse=True)
def _core_host():
    with host_policy.installed(CORE_POLICY):
        yield
