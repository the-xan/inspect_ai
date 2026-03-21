from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from inspect_ai import eval

from homework_week_2.mmlu.analysis import log_to_df, ci_accuracy


class CIPlotter:
    """Визуализация доверительных интервалов (Задание 4)."""

    def __init__(self, model_name: str, subset, task_fn):
        self.model_name = model_name
        self.subset = subset
        self.task_fn = task_fn

    def plot_ci_vs_epochs(self, k_values: List[int] = None, limit: int = 30):
        """
        Задание 4.1: график ширины CI в зависимости от числа эпох (K).

        Запускает eval один раз с max(k_values) эпохами,
        затем для каждого K берёт первые K эпох и считает CI.

        limit: сколько вопросов брать (по умолчанию 30, чтобы не ждать вечно)
        """
        if k_values is None:
            k_values = [5, 6, 7, 8, 9, 10]

        logs = eval(self.task_fn(self.subset), model=self.model_name,
                    epochs=max(k_values),
                    limit=limit)
        df_full = log_to_df(logs[0])

        accuracies, ci_lowers, ci_uppers = [], [], []
        for k in k_values:
            df_k = df_full[df_full["epoch"] < k]
            lower, mean, upper = ci_accuracy(df_k)
            accuracies.append(mean)
            ci_lowers.append(lower)
            ci_uppers.append(upper)

        plt.figure(figsize=(8, 4))
        plt.fill_between(k_values, ci_lowers, ci_uppers, alpha=0.25, label="95% CI")
        plt.plot(k_values, accuracies, "o-", lw=2, label="Accuracy")
        plt.xlabel("Number of runs per question (K)")
        plt.ylabel("Accuracy")
        plt.title(f"{self.model_name} on MMLU-subset — accuracy and CI vs k")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    def plot_ci_vs_n(self, df: pd.DataFrame):
        """
        Задание 4.2: график ширины CI в зависимости от числа вопросов (n).

        Принимает готовый DataFrame из log_to_df.
        """
        question_ids = df["id"].unique()
        dataset_sizes = range(10, len(question_ids) + 1, 10)

        accuracies, ci_lowers, ci_uppers = [], [], []
        for n in dataset_sizes:
            ids_n = question_ids[:n]
            df_n = df[df["id"].isin(ids_n)]
            lower, mean, upper = ci_accuracy(df_n)
            accuracies.append(mean)
            ci_lowers.append(lower)
            ci_uppers.append(upper)

        plt.figure(figsize=(8, 4))
        plt.fill_between(list(dataset_sizes), ci_lowers, ci_uppers, alpha=0.25, label="95% CI")
        plt.plot(list(dataset_sizes), accuracies, "o-", lw=2, label="Accuracy")
        plt.xlabel("Number of questions (n)")
        plt.ylabel("Accuracy")
        plt.title(f"{self.model_name} on MMLU-subset — accuracy and CI vs n")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
