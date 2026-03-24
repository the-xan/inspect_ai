"""
Assignment 9: Compare a model with itself — baseline vs chain-of-thought.

Evaluates the same model twice on the same subset:
  - once with the default multiple_choice() solver
  - once with multiple_choice(cot=True)

Uses significance_by_paired_ttest from Assignment 5 to test whether
the accuracy difference is statistically significant.
"""
from typing import Tuple

import numpy as np
from inspect_ai import eval
from inspect_ai.log import EvalLog

from homework_week_2.mmlu.analysis import log_to_df
from homework_week_2.mmlu.tasks import mmlu_subset, mmlu_subset_cot
from homework_week_2.mmlu.task5_compare_two_models.compare_two_models import (
    significance_by_paired_ttest,
)


def run_and_get_scores_with_task(task_fn, model_name: str, dataset, epochs: int = 1) -> np.ndarray:
    """Run eval with a given task and return mean-per-question scores, sorted by question id."""
    print(f"  Running {model_name} ...")
    run_logs = eval(task_fn(dataset), model=model_name, epochs=epochs)
    df = log_to_df(run_logs[0])
    return df.groupby("id")["score"].mean().sort_index().values


def compare_baseline_vs_cot(
    model: str,
    dataset,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> Tuple[float, float, bool]:
    """
    Evaluate one model with baseline solver and with chain-of-thought on the same dataset.

    Returns (p_value, mean_difference baseline - cot, is_significant).
    """
    print("Running baseline (direct answer)...")
    scores_baseline = run_and_get_scores_with_task(mmlu_subset, model, dataset)

    print("Running chain-of-thought...")
    scores_cot = run_and_get_scores_with_task(mmlu_subset_cot, model, dataset)

    return significance_by_paired_ttest(scores_baseline, scores_cot, alpha, two_tailed)
