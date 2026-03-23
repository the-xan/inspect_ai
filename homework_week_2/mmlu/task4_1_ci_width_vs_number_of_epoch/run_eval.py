from homework_week_2.mmlu.config import MY_SUBSET, MODEL_B
from homework_week_2.mmlu.dataset import load_dataset, filter_subset
from homework_week_2.mmlu.tasks import mmlu_subset
from homework_week_2.mmlu.visualize import CIPlotter

dataset = load_dataset()
subset = filter_subset(dataset, MY_SUBSET)

plotter = CIPlotter(model_name=MODEL_B, subset=subset, task_fn=mmlu_subset)

# Задание 4.1: CI vs количество эпох
plotter.plot_ci_vs_epochs(limit=50)

