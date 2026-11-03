"""Keep tiny classifier fixtures from oversubscribing native math libraries."""

import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="session", autouse=True)
def bounded_native_threads():
    with threadpool_limits(limits=1):
        yield
