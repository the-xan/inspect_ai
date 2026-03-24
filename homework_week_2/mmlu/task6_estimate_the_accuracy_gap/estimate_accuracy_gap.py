from typing import Tuple

import numpy as np
from scipy import stats


def ci_accuracy_for_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    CLT-based confidence interval for the accuracy difference (A - B).

    Вычисляет per-question разности d[i] = scores_a[i] - scores_b[i],
    затем строит доверительный интервал для среднего d.

    Parameters
    ----------
    scores_a : 1-D array of per-question scores for model A (mean across epochs)
    scores_b : 1-D array of per-question scores for model B (mean across epochs)
    ci       : confidence level (default 0.95)

    Returns
    -------
    (lower_bound, mean_difference, upper_bound)
    """
    assert len(scores_a) == len(scores_b), "arrays must cover the same questions"

    d = scores_a - scores_b
    n = len(d)
    mean_d = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf((1 + ci) / 2)

    return mean_d - z * se, mean_d, mean_d + z * se
