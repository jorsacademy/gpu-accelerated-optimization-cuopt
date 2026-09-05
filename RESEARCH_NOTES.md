# Research notes

## What is being tested

The repository separates three questions that are often conflated in GPU-optimization demos:

1. **Model correctness:** does the same canonical LP/MIP represent the same mathematics for each backend?
2. **Solution correctness:** is a returned solution feasible and, for LP, does its objective agree with an independent HiGHS reference?
3. **Performance:** only when cuOpt actually executes on supported GPU hardware do we report wall-clock ratios.

A missing GPU or missing `cuopt` package is therefore a first-class `skipped/unavailable` state, never a fabricated benchmark value.

## cuOpt scope

The implementation targets the NVIDIA cuOpt 26.08 Python `Problem` API for LP/MIP. The adapter mirrors the documented pattern: construct a `Problem`, add continuous/integer variables and linear constraints, set a minimization objective, configure a `SolverSettings` time limit, then call `problem.solve(settings)`.

cuOpt MIP is beta in the 26.08 documentation. For this reason MIP results are treated as primal/incumbent experiments: feasibility, objective, status and a solver-reported gap (when available) are recorded. The project does not assume that every feasible MIP incumbent has been proven optimal.

## Benchmark design

The CPU reference uses SciPy's HiGHS-backed `linprog` for LP and `milp` for MIP. Two generated problem families are included:

- balanced transportation LPs, which produce sparse network-flow-like linear programs;
- multidimensional 0-1 knapsack MIPs, which stress integer primal search.

Generation is seeded and deterministic. All timing is wall clock around the solve call. Solver-reported time is retained separately when a backend exposes it.

## Remote execution

NVIDIA documents transparent remote LP/MIP solving through `CUOPT_REMOTE_HOST` and `CUOPT_REMOTE_PORT`. This means the same adapter can remain on a CPU client while the actual cuOpt engine runs on a GPU server. Routing has different remote-server semantics and is intentionally outside this first benchmark harness.

## Interpretation

Do not infer GPU acceleration from CPU-only CI. CI proves package integrity, model generation, HiGHS reference correctness, benchmark schema, and the explicit no-cuOpt path. A GPU result is publishable only when the generated JSON records `candidate.solver == "nvidia-cuopt"` and the hardware report shows the actual environment.
