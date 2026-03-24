"""
Юнит-тесты для variance_components.py.
Не требуют запуска реальных моделей — используют mock EvalLog.
"""
import numpy as np
import pytest

from homework_week_2.mmlu.task7_estimate_variance_components.variance_components import (
    estimate_variance_components,
    minimum_detectable_effect,
    required_sample_size,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal mock EvalLog objects
# ---------------------------------------------------------------------------

class MockScore:
    def __init__(self, value: str):
        self.value = value


class MockSample:
    def __init__(self, id, epoch, correct: bool, subject="test"):
        self.id = id
        self.epoch = epoch
        self.scores = {"choice": MockScore("C" if correct else "I")}
        self.metadata = {"subject": subject}


class MockLog:
    def __init__(self, samples):
        self.samples = samples


def make_log(scores_by_id: dict) -> MockLog:
    """
    scores_by_id: {question_id: [score_epoch0, score_epoch1, ...]}
    score values: 1 = correct, 0 = incorrect
    """
    samples = []
    for qid, epochs in scores_by_id.items():
        for ep, val in enumerate(epochs):
            samples.append(MockSample(id=qid, epoch=ep, correct=bool(val)))
    return MockLog(samples)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sigma2_zero_when_model_always_agrees():
    """Если модель всегда даёт одинаковый ответ — sigma2 = 0."""
    # Model A всегда правильно, Model B всегда неправильно
    log_a = make_log({i: [1, 1] for i in range(10)})
    log_b = make_log({i: [0, 0] for i in range(10)})
    params = estimate_variance_components([log_a], [log_b])
    assert params["sigma2_a"] == pytest.approx(0.0)
    assert params["sigma2_b"] == pytest.approx(0.0)


def test_omega2_nonnegative():
    """omega2 должна быть >= 0."""
    np.random.seed(42)
    log_a = make_log({i: list(np.random.randint(0, 2, 3)) for i in range(20)})
    log_b = make_log({i: list(np.random.randint(0, 2, 3)) for i in range(20)})
    params = estimate_variance_components([log_a], [log_b])
    assert params["omega2"] >= 0.0


def test_required_sample_size_roundtrip():
    """required_sample_size и minimum_detectable_effect — обратные функции."""
    params = {"omega2": 0.15, "sigma2_a": 0.05, "sigma2_b": 0.05}
    for delta in (0.05, 0.10, 0.20):
        n = required_sample_size(delta=delta, **params)
        mde = minimum_detectable_effect(n=n, **params)
        assert abs(mde - delta) < 0.005, f"Round-trip failed for delta={delta}: MDE={mde:.4f}"


def test_required_sample_size_larger_delta_needs_fewer_questions():
    """Чем больше delta, тем меньше вопросов нужно."""
    params = {"omega2": 0.15, "sigma2_a": 0.05, "sigma2_b": 0.05}
    n5  = required_sample_size(delta=0.05, **params)
    n10 = required_sample_size(delta=0.10, **params)
    assert n5 > n10


def test_mde_decreases_with_more_questions():
    """MDE уменьшается при увеличении числа вопросов."""
    params = {"omega2": 0.15, "sigma2_a": 0.05, "sigma2_b": 0.05}
    mde_100 = minimum_detectable_effect(n=100, **params)
    mde_400 = minimum_detectable_effect(n=400, **params)
    assert mde_400 < mde_100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
