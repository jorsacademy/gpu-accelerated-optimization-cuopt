from __future__ import annotations

from importlib.util import find_spec
from time import perf_counter

import numpy as np

from .model import LinearModel, SolveResult


class CuOptUnavailable(RuntimeError):
    """Raised when cuOpt is not installed in the current Python environment."""


def cuopt_available() -> bool:
    return find_spec("cuopt") is not None


def solve_cuopt(model: LinearModel, *, time_limit: float = 60.0) -> SolveResult:
    """Solve a canonical linear model using NVIDIA cuOpt's Python API.

    Imports are intentionally local so CPU-only CI can import and test this package.
    API shape follows the cuOpt 26.08 Problem/SolverSettings interface.
    """
    if not cuopt_available():
        raise CuOptUnavailable(
            "NVIDIA cuOpt is not installed. Run inside a supported cuOpt GPU environment "
            "or configure CUOPT_REMOTE_HOST/CUOPT_REMOTE_PORT for LP/MIP remote execution."
        )

    from cuopt.linear_programming.problem import CONTINUOUS, INTEGER, MINIMIZE, Problem
    from cuopt.linear_programming.solver_settings import SolverSettings

    problem = Problem(model.name)
    variables = []
    for i in range(model.n_variables):
        vtype = INTEGER if model.integrality[i] else CONTINUOUS
        ub = float(model.ub[i]) if np.isfinite(model.ub[i]) else float("inf")
        variables.append(
            problem.addVariable(lb=float(model.lb[i]), ub=ub, vtype=vtype, name=f"x_{i}")
        )

    for r in range(model.n_constraints):
        nz = np.flatnonzero(model.A_ub[r])
        expr = 0
        for j in nz:
            expr = expr + float(model.A_ub[r, j]) * variables[int(j)]
        problem.addConstraint(expr <= float(model.b_ub[r]), name=f"c_{r}")

    objective = 0
    for j in np.flatnonzero(model.c):
        objective = objective + float(model.c[j]) * variables[int(j)]
    problem.setObjective(objective, sense=MINIMIZE)

    settings = SolverSettings()
    settings.set_parameter("time_limit", float(time_limit))
    start = perf_counter()
    problem.solve(settings)
    wall = perf_counter() - start

    status_name = getattr(problem.Status, "name", str(problem.Status))
    status = status_name.lower().replace(" ", "_")
    try:
        x = np.array([float(v.getValue()) for v in variables], dtype=float)
    except Exception:
        x = None
    objective_value = None
    if x is not None:
        objective_value = float(problem.ObjValue)
    solver_seconds = getattr(problem, "SolveTime", None)
    gap = getattr(problem, "MIPGap", None)
    return SolveResult(
        solver="nvidia-cuopt",
        status=status,
        objective=objective_value,
        x=x,
        wall_seconds=wall,
        solver_seconds=None if solver_seconds is None else float(solver_seconds),
        mip_gap=None if gap is None else float(gap),
        message="",
    )
