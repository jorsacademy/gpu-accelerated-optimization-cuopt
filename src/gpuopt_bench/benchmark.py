from __future__ import annotations

from dataclasses import asdict, dataclass

from .cpu import solve_scipy
from .cuopt_adapter import CuOptUnavailable, solve_cuopt
from .model import LinearModel, SolveResult


@dataclass(frozen=True)
class Comparison:
    model: str
    model_type: str
    n_variables: int
    n_constraints: int
    reference: SolveResult
    candidate: SolveResult | None
    objective_relative_error: float | None
    candidate_max_violation: float | None
    wall_speedup: float | None
    note: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("reference", "candidate"):
            obj = payload[key]
            if obj is not None and obj["x"] is not None:
                obj["x"] = obj["x"].tolist()
        return payload


def compare_with_cuopt(model: LinearModel, *, time_limit: float = 60.0) -> Comparison:
    reference = solve_scipy(model, time_limit=time_limit)
    try:
        candidate = solve_cuopt(model, time_limit=time_limit)
    except CuOptUnavailable as exc:
        return Comparison(
            model=model.name,
            model_type="mip" if model.is_mip else "lp",
            n_variables=model.n_variables,
            n_constraints=model.n_constraints,
            reference=reference,
            candidate=None,
            objective_relative_error=None,
            candidate_max_violation=None,
            wall_speedup=None,
            note=str(exc),
        )

    error = None
    violation = None
    speedup = None
    if reference.objective is not None and candidate.objective is not None:
        denom = max(1.0, abs(reference.objective))
        error = abs(candidate.objective - reference.objective) / denom
    if candidate.x is not None:
        violation = model.max_violation(candidate.x)
    if candidate.wall_seconds > 0:
        speedup = reference.wall_seconds / candidate.wall_seconds
    note = (
        "LP objective error is a correctness diagnostic, not a performance claim."
        if not model.is_mip
        else "cuOpt MIP is beta; compare incumbent feasibility/objective and reported gap/status."
    )
    return Comparison(
        model=model.name,
        model_type="mip" if model.is_mip else "lp",
        n_variables=model.n_variables,
        n_constraints=model.n_constraints,
        reference=reference,
        candidate=candidate,
        objective_relative_error=error,
        candidate_max_violation=violation,
        wall_speedup=speedup,
        note=note,
    )
