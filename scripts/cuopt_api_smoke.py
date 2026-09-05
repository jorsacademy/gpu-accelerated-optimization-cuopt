from __future__ import annotations

import json

from gpuopt_bench.cuopt_adapter import CuOptUnavailable, solve_cuopt
from gpuopt_bench.generators import transportation_lp
from gpuopt_bench.hardware import environment_report


def main() -> None:
    model = transportation_lp(n_sources=3, n_sinks=4, seed=7)
    try:
        result = solve_cuopt(model, time_limit=20)
    except CuOptUnavailable as exc:
        payload = {
            "environment": environment_report(),
            "status": "skipped",
            "reason": str(exc),
        }
        print(json.dumps(payload, indent=2))
        return
    violation = model.max_violation(result.x) if result.x is not None else None
    payload = {
        "environment": environment_report(),
        "status": result.status,
        "objective": result.objective,
        "max_violation": violation,
        "wall_seconds": result.wall_seconds,
        "solver_seconds": result.solver_seconds,
    }
    print(json.dumps(payload, indent=2))
    if not result.has_solution or violation is None or violation > 1e-6:
        raise SystemExit("cuOpt smoke solve did not return a feasible LP solution")


if __name__ == "__main__":
    main()
