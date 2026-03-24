"""
Assignment 9: Compare a model with itself — baseline vs chain-of-thought.
"""
from pathlib import Path

from inspect_ai import eval
from inspect_ai.log import read_eval_log

from homework_week_2.mmlu.analysis import log_to_df
from homework_week_2.mmlu.config import MODEL_A, MY_SUBSET
from homework_week_2.mmlu.dataset import load_dataset, filter_subset
from homework_week_2.mmlu.tasks import mmlu_subset_cot
from homework_week_2.mmlu.task9_compare_cot.compare_cot import compare_baseline_vs_cot

subset = filter_subset(load_dataset(), MY_SUBSET)

print(f"Model: {MODEL_A}")
print(f"Subset: {MY_SUBSET} ({len(subset)} questions)\n")

# Baseline: загружаем ранее выполненный eval
# logs_baseline = eval(mmlu_subset(subset), model=MODEL_A)
BASELINE_LOG = Path(__file__).parent / "logs" / "2026-03-24T11-26-32+00-00_mmlu-subset_ddGb4PXh8xU26BF5EsPtRM.eval"
log_baseline = read_eval_log(str(BASELINE_LOG))

logs_cot = eval(mmlu_subset_cot(subset), model=MODEL_A)

df_baseline = log_to_df(log_baseline)
df_cot      = log_to_df(logs_cot[0])

acc_baseline = df_baseline["score"].mean()
acc_cot      = df_cot["score"].mean()

scores_baseline = df_baseline.groupby("id")["score"].mean().sort_index().values
scores_cot      = df_cot.groupby("id")["score"].mean().sort_index().values

from homework_week_2.mmlu.task5_compare_two_models.compare_two_models import significance_by_paired_ttest

p_value, mean_diff, is_significant = significance_by_paired_ttest(scores_baseline, scores_cot)

print(f"\nBaseline accuracy:        {acc_baseline:.1%}")
print(f"Chain-of-thought accuracy: {acc_cot:.1%}")
print(f"Mean difference (baseline - CoT): {mean_diff:+.3f}")
print(f"p-value:                  {p_value:.4f}")
print(f"Significant at α=0.05:    {is_significant}")

# Result
# choice
# accuracy 0.260
# stderr 0.038
#
#
# Baseline accuracy: 31.3 %
# Chain - of - thought accuracy: 26.0 %
# Mean difference(baseline - CoT): +0.053
# p - value: 0.2099
# Significant at α = 0.05:    False