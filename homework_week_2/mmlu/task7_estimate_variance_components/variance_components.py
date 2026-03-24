import math
from typing import List

import numpy as np
from scipy import stats

from inspect_ai.log import EvalLog

from homework_week_2.mmlu.analysis import log_to_df


def estimate_variance_components(
    logs_a: List[EvalLog],
    logs_b: List[EvalLog],
) -> dict:
    """
    Оценивает omega2, sigma2_a, sigma2_b из двух EvalLog объектов (см. §5 статьи).

    Оба лога должны охватывать одни и те же вопросы. Используйте epochs >= 2,
    чтобы можно было оценить дисперсию внутри вопроса.

    Возвращает dict с ключами 'omega2', 'sigma2_a', 'sigma2_b'.

    Компоненты дисперсии:
    - omega2   — дисперсия истинных per-question разностей (сложность вопроса)
    - sigma2_a — средняя дисперсия ответов модели A на один вопрос (шум модели)
    - sigma2_b — то же для модели B

    Формула: Var(d_bar_i) = omega2 + sigma2_a/Ka + sigma2_b/Kb
    Откуда:  omega2 = Var(d_bar_i) - sigma2_a/Ka - sigma2_b/Kb
    """
    df_a = log_to_df(logs_a[0])
    df_b = log_to_df(logs_b[0])

    # Per-question mean scores (усреднение по эпохам)
    mean_a = df_a.groupby("id")["score"].mean()
    mean_b = df_b.groupby("id")["score"].mean()

    # Работаем только с общими вопросами
    common_ids = mean_a.index.intersection(mean_b.index)
    mean_a = mean_a.loc[common_ids]
    mean_b = mean_b.loc[common_ids]

    # Число эпох (прогонов) для каждой модели
    ka = int(df_a.groupby("id")["epoch"].count().loc[common_ids].iloc[0])
    kb = int(df_b.groupby("id")["epoch"].count().loc[common_ids].iloc[0])

    # sigma2_m = средняя дисперсия ответов внутри одного вопроса для модели m
    # Оценивается как среднее per-question var (с ddof=1)
    sigma2_a = float(
        df_a[df_a["id"].isin(common_ids)]
        .groupby("id")["score"]
        .var(ddof=1)
        .mean()
    )
    sigma2_b = float(
        df_b[df_b["id"].isin(common_ids)]
        .groupby("id")["score"]
        .var(ddof=1)
        .mean()
    )

    # Дисперсия per-question разностей (выборочная дисперсия d_bar_i)
    d = (mean_a - mean_b).values
    var_d = float(np.var(d, ddof=1))

    # omega2 = var(d_bar_i) - sigma2_a/Ka - sigma2_b/Kb  (clip to 0)
    omega2 = max(0.0, var_d - sigma2_a / ka - sigma2_b / kb)

    return {
        "omega2":   omega2,
        "sigma2_a": sigma2_a,
        "sigma2_b": sigma2_b,
    }


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
    Минимальное количество вопросов для обнаружения разницы delta
    с заданной мощностью (Eq. 9 в статье).

    n = ceil((z_alpha + z_beta)^2 * (omega2 + sigma2_a/ka + sigma2_b/kb) / delta^2)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    n = (z_alpha + z_beta) ** 2 * (omega2 + sigma2_a / ka + sigma2_b / kb) / delta ** 2
    return math.ceil(n)
