from __future__ import annotations

import numpy as np

from .model import LinearModel


def transportation_lp(
    *,
    n_sources: int = 24,
    n_sinks: int = 32,
    seed: int = 0,
) -> LinearModel:
    """Generate a balanced transportation LP in <= canonical form."""
    if min(n_sources, n_sinks) < 1:
        raise ValueError("source and sink counts must be positive")
    rng = np.random.default_rng(seed)
    supply = rng.integers(20, 80, size=n_sources).astype(float)
    total = int(supply.sum())
    raw = rng.random(n_sinks)
    demand = np.floor(raw / raw.sum() * total).astype(int)
    demand[0] += total - int(demand.sum())
    cost = rng.uniform(1.0, 30.0, size=(n_sources, n_sinks))
    n = n_sources * n_sinks
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for i in range(n_sources):
        row = np.zeros(n)
        row[i * n_sinks : (i + 1) * n_sinks] = 1.0
        rows.extend([row, -row])
        rhs.extend([supply[i], -supply[i]])
    for j in range(n_sinks):
        row = np.zeros(n)
        row[j::n_sinks] = 1.0
        rows.extend([row, -row])
        rhs.extend([demand[j], -demand[j]])

    return LinearModel(
        c=cost.ravel(),
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs),
        lb=np.zeros(n),
        ub=np.full(n, np.inf),
        integrality=np.zeros(n, dtype=int),
        name=f"transport_{n_sources}x{n_sinks}",
    )


def multidimensional_knapsack_mip(
    *,
    n_items: int = 80,
    n_resources: int = 6,
    seed: int = 0,
) -> LinearModel:
    """Generate a 0-1 multidimensional knapsack as minimization of negative profit."""
    if min(n_items, n_resources) < 1:
        raise ValueError("dimensions must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 40, size=(n_resources, n_items)).astype(float)
    profit = rng.integers(10, 100, size=n_items).astype(float)
    capacity = np.floor(weights.sum(axis=1) * rng.uniform(0.28, 0.42, n_resources))
    return LinearModel(
        c=-profit,
        A_ub=weights,
        b_ub=capacity,
        lb=np.zeros(n_items),
        ub=np.ones(n_items),
        integrality=np.ones(n_items, dtype=int),
        name=f"mdkp_{n_items}x{n_resources}",
    )
