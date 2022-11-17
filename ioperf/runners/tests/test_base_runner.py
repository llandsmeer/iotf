import pytest
from .. import BaseRunner

def test_base_runner_not_impl():
    x = BaseRunner()
    with pytest.raises(NotImplementedError):
        x.run_unconnected(1, 2)
