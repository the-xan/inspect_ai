from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from inspect_ai.log import EvalLog


def log_to_df(log: EvalLog) -> pd.DataFrame:
    """
    Конвертирует EvalLog в DataFrame с одной строкой на (вопрос, epoch).

    Колонки:
        id      – идентификатор вопроса
        epoch   – индекс эпохи (0 если epochs=1) (это повторение запроса)
        score   – 1 если правильно, 0 если нет
        subject – тема MMLU из metadata

    Scorer choice() хранит результат как "C" (correct) или "I" (incorrect).
    """
    rows = []
    for sample in log.samples:
        value = sample.scores["choice"].value
        rows.append({
            "id": sample.id,
            "epoch": sample.epoch,
            "score": 1 if value == "C" else 0,
            "subject": sample.metadata.get("subject"),
        })
    return pd.DataFrame(rows)

def ci_accuracy_basic(scores: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    CLT-based confidence interval for accuracy -- single run per question (K = 1).

    Parameters
    ----------
    scores : 1-D array of per-question binary scores (0 or 1)
    ci     : confidence level (default 0.95)

    Returns
    -------
    (lower_bound, mean_accuracy, upper_bound)
    """
    n = len(scores)
    mean = float(np.mean(scores))
    se = np.sqrt(mean * (1 - mean) / n)
    z = stats.norm.ppf((1 + ci) / 2)
    return mean - z * se, mean, mean + z * se


def ci_accuracy(df: pd.DataFrame, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    CLT-based confidence interval for accuracy, supporting multiple epochs (K >= 1).

    Parameters
    ----------
    df : DataFrame returned by log_to_df, with columns 'id', 'score', 'epoch'
    ci : confidence level (default 0.95)

    Returns
    -------
    (lower_bound, mean_accuracy, upper_bound)
    """
    per_q = df.groupby("id")["score"].mean()
    n = len(per_q)
    grand_mean = float(per_q.mean())
    var = float(per_q.var(ddof=1))
    se = np.sqrt(var / n)
    z = stats.norm.ppf((1 + ci) / 2)
    return grand_mean - z * se, grand_mean, grand_mean + z * se