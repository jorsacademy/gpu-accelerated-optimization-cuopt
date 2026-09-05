import itertools

import numpy as np
import pytest

from gpuopt_bench.cpu import solve_scipy
from gpuopt_bench.generators import multidimensional_knapsack_mip, transportation_lp


def test_transportation_lp_solves_and_is_feasible():
    model = transportation_lp(n_sources=4, n_sinks=5, seed=3)
    result = solve_scipy(model)
    assert result.status == "optimal"
    assert result.x is not None
    assert model.max_violation(result.x) <= 1e-7


def test_mip_matches_bruteforce_on_tiny_instance():
    model = multidimensional_knapsack_mip(n_items=10, n_resources=3, seed=5)
    result = solve_scipy(model)
    assert result.status == "optimal"
    best = float("inf")
    for bits in itertools.product((0.0, 1.0), repeat=model.n_variables):
        x = np.asarray(bits)
        if model.max_violation(x) <= 1e-9:
            best = min(best, model.objective(x))
    assert result.objective == pytest.approx(best, abs=1e-7)
