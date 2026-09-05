from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpuopt_bench.benchmark import compare_with_cuopt
from gpuopt_bench.generators import multidimensional_knapsack_mip, transportation_lp
from gpuopt_bench.hardware import environment_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()

    models = [
        transportation_lp(n_sources=18, n_sinks=24, seed=args.seed),
        multidimensional_knapsack_mip(n_items=70, n_resources=5, seed=args.seed + 1),
    ]
    comparisons = [compare_with_cuopt(model, time_limit=args.time_limit) for model in models]
    payload = {
        "environment": environment_report(),
        "results": [item.to_dict() for item in comparisons],
        "policy": {
            "speedup_claim_requires_cuopt_execution": True,
            "lp_objective_relative_error_tolerance": 1e-6,
            "mip_is_beta": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    for result in comparisons:
        if result.reference.status != "optimal":
            raise SystemExit(f"CPU reference failed for {result.model}: {result.reference.status}")
        if result.candidate is not None and result.model_type != "mip":
            if result.objective_relative_error is None or result.objective_relative_error > 1e-6:
                raise SystemExit(f"cuOpt LP objective mismatch for {result.model}")
            if result.candidate_max_violation is None or result.candidate_max_violation > 1e-6:
                raise SystemExit(f"cuOpt LP feasibility failure for {result.model}")


if __name__ == "__main__":
    main()
