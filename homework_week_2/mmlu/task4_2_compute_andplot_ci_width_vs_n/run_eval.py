from typing import List

from inspect_ai import eval
from inspect_ai.log import EvalLog

from homework_week_2.mmlu.analysis import log_to_df
from homework_week_2.mmlu.config import MODEL_A, MY_SUBSET
from homework_week_2.mmlu.dataset import filter_subset, load_dataset
from homework_week_2.mmlu.visualize import CIPlotter
from homework_week_2.mmlu.tasks import mmlu_subset

dataset = load_dataset()
subset = filter_subset(dataset, MY_SUBSET)

plotter = CIPlotter(model_name=MODEL_A, subset=subset, task_fn=mmlu_subset)

logs: List[EvalLog] = eval(
    mmlu_subset(subset),
    model=MODEL_A,
   # limit=50
)

# Задание 4.2: CI vs количество вопросов
df = log_to_df(logs[0])
plotter.plot_ci_vs_n(df)
