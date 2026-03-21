from typing import List

import numpy as np
import pandas as pd
from inspect_ai import eval
from inspect_ai.log import EvalLog

from homework_week_2.mmlu.visualize import CIPlotter
from homework_week_2.mmlu.analysis import log_to_df, ci_accuracy_basic, ci_accuracy
from homework_week_2.mmlu.config import MODEL_A, MY_SUBSET, MODEL_B
from homework_week_2.mmlu.dataset import load_dataset, filter_subset
from homework_week_2.mmlu.tasks import mmlu_subset

if __name__ == "__main__":

    dataset = load_dataset()
    subset = filter_subset(dataset, MY_SUBSET)

    plotter = CIPlotter(model_name=MODEL_B, subset=subset, task_fn=mmlu_subset)

    # Задание 4.1: CI vs количество эпох
    plotter.plot_ci_vs_epochs(limit=50)


    # # Задание 4.2: CI vs количество вопросов
    # # (нужен df из любого предыдущего eval)
    # logs: List[EvalLog] = eval(
    #     mmlu_subset(subset),
    #     model=MODEL_A,
    #     limit=10  # evaluate only the first 10 questions
    # )
    # df = log_to_df(logs[0])
    # plotter.plot_ci_vs_n(df)
