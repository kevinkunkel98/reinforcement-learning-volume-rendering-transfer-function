import pytest

import transfer


@pytest.fixture(autouse=True)
def _reset_unlabelled_opacity():
    """`goals.use_task_version` sets a process-wide shell opacity (task v3);
    no test may leak it into the next."""
    yield
    transfer.set_unlabelled_opacity(1.0)
    transfer.set_solo_probe_heights((1.0,))
