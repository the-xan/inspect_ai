"""
Assignment 8: Implement required_sample_size.

Обратная функция к minimum_detectable_effect:
по заданному delta возвращает минимальное количество вопросов.

Формула (Eq. 9):
    n = ceil((z_alpha + z_beta)^2 * (omega2 + sigma2_a/ka + sigma2_b/kb) / delta^2)
"""
import math

import numpy as np
from scipy import stats


def minimum_detectable_effect(
    n: int,
    omega2: float,
    sigma2_a: float = 0.0,
    sigma2_b: float = 0.0,
    ka: int = 1,
    kb: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """MDE для парного сравнения моделей (Eq. 10 в статье)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * np.sqrt(
        (omega2 + sigma2_a / ka + sigma2_b / kb) / n
    ))


def required_sample_size(
    delta: float,
    omega2: float,
    sigma2_a: float = 0.0,
    sigma2_b: float = 0.0,
    ka: int = 1,
    kb: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Минимальное количество вопросов для обнаружения разницы delta с заданной мощностью.

    Обратная функция к minimum_detectable_effect (Eq. 9 в статье):

        n = ceil((z_alpha + z_beta)^2 * (omega2 + sigma2_a/ka + sigma2_b/kb) / delta^2)

    Parameters
    ----------
    delta    : целевой эффект (разница в accuracy), который нужно обнаружить
    omega2   : дисперсия истинных per-question разностей (из estimate_variance_components)
    sigma2_a : внутримодельная дисперсия модели A
    sigma2_b : внутримодельная дисперсия модели B
    ka, kb   : число прогонов (epochs) для каждой модели
    alpha    : уровень значимости (default 0.05)
    power    : желаемая мощность теста (default 0.80)

    Returns
    -------
    Минимальное целое n (округлено вверх).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    n = (z_alpha + z_beta) ** 2 * (omega2 + sigma2_a / ka + sigma2_b / kb) / delta ** 2
    return math.ceil(n)
