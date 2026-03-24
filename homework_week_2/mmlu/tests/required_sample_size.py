"""
Юнит-тесты для required_sample_size.py.
"""
import math

import numpy as np
import pytest

from homework_week_2.mmlu.task8_inplement_required_sample_size.required_sample_size import (
    minimum_detectable_effect,
    required_sample_size,
)


# ---------------------------------------------------------------------------
# Round-trip: required_sample_size и minimum_detectable_effect — обратные функции
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [0.05, 0.10, 0.20])
def test_roundtrip(delta):
    """minimum_detectable_effect(required_sample_size(delta)) ≈ delta."""
    params = {"omega2": 0.15, "sigma2_a": 0.05, "sigma2_b": 0.05}
    n = required_sample_size(delta=delta, **params)
    mde = minimum_detectable_effect(n=n, **params)
    assert abs(mde - delta) < 0.005, f"Round-trip failed for delta={delta}: MDE={mde:.4f}"


def test_roundtrip_no_within_model_variance():
    """Round-trip работает при sigma2_a = sigma2_b = 0."""
    params = {"omega2": 0.20, "sigma2_a": 0.0, "sigma2_b": 0.0}
    n = required_sample_size(delta=0.05, **params)
    mde = minimum_detectable_effect(n=n, **params)
    assert abs(mde - 0.05) < 0.005


# ---------------------------------------------------------------------------
# Монотонность: больший delta -> меньше вопросов
# ---------------------------------------------------------------------------

def test_larger_delta_needs_fewer_questions():
    params = {"omega2": 0.15, "sigma2_a": 0.05, "sigma2_b": 0.05}
    n5  = required_sample_size(delta=0.05, **params)
    n10 = required_sample_size(delta=0.10, **params)
    n20 = required_sample_size(delta=0.20, **params)
    assert n5 > n10 > n20


# ---------------------------------------------------------------------------
# Возвращает целое число и округляет вверх
# ---------------------------------------------------------------------------

def test_returns_int():
    params = {"omega2": 0.10, "sigma2_a": 0.0, "sigma2_b": 0.0}
    n = required_sample_size(delta=0.07, **params)
    assert isinstance(n, int)


def test_ceiling_rounding():
    """n >= точного вещественного значения."""
    params = {"omega2": 0.10, "sigma2_a": 0.0, "sigma2_b": 0.0}
    z_alpha_beta = 1.959964 + 0.841621  # z_{0.025} + z_{0.20}
    exact = z_alpha_beta ** 2 * params["omega2"] / 0.07 ** 2
    n = required_sample_size(delta=0.07, **params)
    assert n == math.ceil(exact)


# ---------------------------------------------------------------------------
# Влияние alpha и power
# ---------------------------------------------------------------------------

def test_higher_power_needs_more_questions():
    """Больше мощности -> больше вопросов."""
    params = {"omega2": 0.15, "sigma2_a": 0.05, "sigma2_b": 0.05}
    n80 = required_sample_size(delta=0.10, power=0.80, **params)
    n90 = required_sample_size(delta=0.10, power=0.90, **params)
    assert n90 > n80


def test_stricter_alpha_needs_more_questions():
    """Меньший alpha -> больше вопросов."""
    params = {"omega2": 0.15, "sigma2_a": 0.05, "sigma2_b": 0.05}
    n05 = required_sample_size(delta=0.10, alpha=0.05, **params)
    n01 = required_sample_size(delta=0.10, alpha=0.01, **params)
    assert n01 > n05


# ---------------------------------------------------------------------------
# Влияние ka/kb: больше эпох -> меньше вопросов
# ---------------------------------------------------------------------------

def test_more_epochs_reduces_required_n():
    """Больше прогонов (ka, kb) снижает требуемое n."""
    params = {"omega2": 0.15, "sigma2_a": 0.10, "sigma2_b": 0.10}
    n_k1 = required_sample_size(delta=0.10, ka=1, kb=1, **params)
    n_k3 = required_sample_size(delta=0.10, ka=3, kb=3, **params)
    assert n_k3 < n_k1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
