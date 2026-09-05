from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class LinearModel:
    """Canonical minimization model.

    Minimize c @ x subject to A_ub @ x <= b_ub and lb <= x <= ub.
    integrality[i] is 0 for continuous and 1 for integer variables.
    """

    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray
    name: str = "model"

    def __post_init__(self) -> None:
        c = np.asarray(self.c, dtype=float)
        a = np.asarray(self.A_ub, dtype=float)
        b = np.asarray(self.b_ub, dtype=float)
        lb = np.asarray(self.lb, dtype=float)
        ub = np.asarray(self.ub, dtype=float)
        integ = np.asarray(self.integrality, dtype=int)
        n = c.size
        if c.ndim != 1 or n == 0:
            raise ValueError("c must be a non-empty vector")
        if a.ndim != 2 or a.shape[1] != n:
            raise ValueError("A_ub must have shape (m, n)")
        if b.shape != (a.shape[0],):
            raise ValueError("b_ub must match A_ub rows")
        if lb.shape != (n,) or ub.shape != (n,) or integ.shape != (n,):
            raise ValueError("bounds and integrality must have shape (n,)")
        if np.any(lb > ub):
            raise ValueError("lower bounds cannot exceed upper bounds")
        if not np.all(np.isin(integ, [0, 1])):
            raise ValueError("integrality must contain only 0/1")
        for field, value in (
            ("c", c),
            ("A_ub", a),
            ("b_ub", b),
            ("lb", lb),
            ("ub", ub),
            ("integrality", integ),
        ):
            object.__setattr__(self, field, value)

    @property
    def n_variables(self) -> int:
        return self.c.size

    @property
    def n_constraints(self) -> int:
        return self.A_ub.shape[0]

    @property
    def is_mip(self) -> bool:
        return bool(np.any(self.integrality))

    def objective(self, x: np.ndarray) -> float:
        return float(self.c @ np.asarray(x, dtype=float))

    def max_violation(self, x: np.ndarray, *, tol_integrality: bool = True) -> float:
        x = np.asarray(x, dtype=float)
        violations = [0.0]
        if self.n_constraints:
            violations.append(float(np.max(self.A_ub @ x - self.b_ub)))
        violations.append(float(np.max(self.lb - x)))
        violations.append(float(np.max(x - self.ub)))
        if tol_integrality and self.is_mip:
            idx = self.integrality == 1
            violations.append(float(np.max(np.abs(x[idx] - np.rint(x[idx])))))
        return max(violations)


@dataclass(frozen=True)
class SolveResult:
    solver: str
    status: str
    objective: float | None
    x: np.ndarray | None
    wall_seconds: float
    solver_seconds: float | None = None
    mip_gap: float | None = None
    message: str = ""

    @property
    def has_solution(self) -> bool:
        return self.x is not None and self.objective is not None


def timed_call(func):
    start = perf_counter()
    result = func()
    return result, perf_counter() - start
