from unittest.mock import patch

import pytest

from gpuopt_bench.cuopt_adapter import CuOptUnavailable, cuopt_available, solve_cuopt
from gpuopt_bench.generators import transportation_lp


def test_cuopt_available_returns_boolean():
    assert isinstance(cuopt_available(), bool)


def test_missing_cuopt_has_explicit_skip_path():
    model = transportation_lp(n_sources=2, n_sinks=2, seed=2)
    with patch("gpuopt_bench.cuopt_adapter.cuopt_available", return_value=False):
        with pytest.raises(CuOptUnavailable, match="CUOPT_REMOTE_HOST"):
            solve_cuopt(model)
