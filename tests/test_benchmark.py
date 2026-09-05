from unittest.mock import patch

from gpuopt_bench.benchmark import compare_with_cuopt
from gpuopt_bench.cuopt_adapter import CuOptUnavailable
from gpuopt_bench.generators import transportation_lp


def test_cpu_only_comparison_never_fabricates_speedup():
    model = transportation_lp(n_sources=3, n_sinks=4, seed=4)
    with patch(
        "gpuopt_bench.benchmark.solve_cuopt",
        side_effect=CuOptUnavailable("no GPU runtime"),
    ):
        result = compare_with_cuopt(model)
    assert result.reference.status == "optimal"
    assert result.candidate is None
    assert result.wall_speedup is None
    assert "no GPU" in result.note
