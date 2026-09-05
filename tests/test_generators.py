import numpy as np

from gpuopt_bench.generators import multidimensional_knapsack_mip, transportation_lp


def test_generators_are_deterministic():
    a = transportation_lp(seed=9, n_sources=3, n_sinks=4)
    b = transportation_lp(seed=9, n_sources=3, n_sinks=4)
    assert np.array_equal(a.c, b.c)
    assert np.array_equal(a.A_ub, b.A_ub)

    m = multidimensional_knapsack_mip(seed=11, n_items=12, n_resources=2)
    assert m.is_mip
    assert np.all(m.integrality == 1)
