from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .model import LinearModel, SolveResult


def solve_scipy(model: LinearModel, *, time_limit: float | None = None) -> SolveResult:
    start = perf_counter()
    if model.is_mip:
        constraints = LinearConstraint(model.A_ub, -np.inf, model.b_ub)
        options = {} if time_limit is None else {"time_limit": time_limit}
        result = milp(
            c=model.c,
            integrality=model.integrality,
            bounds=Bounds(model.lb, model.ub),
            constraints=constraints,
            options=options,
        )
        status_map = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "error"}
        gap = getattr(result, "mip_gap", None)
    else:
        bounds = list(zip(model.lb, model.ub, strict=True))
        options = {} if time_limit is None else {"time_limit": time_limit}
        result = linprog(
            model.c,
            A_ub=model.A_ub,
            b_ub=model.b_ub,
            bounds=bounds,
            method="highs",
            options=options,
        )
        status_map = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "error"}
        gap = None
    wall = perf_counter() - start
    x = None if result.x is None else np.asarray(result.x, dtype=float)
    objective = None if result.fun is None else float(result.fun)
    return SolveResult(
        solver="scipy-highs",
        status=status_map.get(int(result.status), f"status-{result.status}"),
        objective=objective,
        x=x,
        wall_seconds=wall,
        mip_gap=None if gap is None else float(gap),
        message=str(result.message),
    )
