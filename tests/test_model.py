import numpy as np
import pytest

from gpuopt_bench.model import LinearModel


def test_model_validation_and_violation():
    model = LinearModel(
        c=np.array([1.0, 2.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([3.0]),
        lb=np.zeros(2),
        ub=np.ones(2) * 3,
        integrality=np.zeros(2, dtype=int),
    )
    assert model.max_violation(np.array([1.0, 2.0])) == pytest.approx(0.0)
    assert model.max_violation(np.array([2.0, 2.0])) == pytest.approx(1.0)


def test_model_rejects_bad_shapes():
    with pytest.raises(ValueError):
        LinearModel(
            c=np.array([1.0, 2.0]),
            A_ub=np.ones((2, 3)),
            b_ub=np.ones(2),
            lb=np.zeros(2),
            ub=np.ones(2),
            integrality=np.zeros(2),
        )
