from pathlib import Path

from inspect_ai.log import read_eval_log

from homework_week_2.mmlu.analysis import log_to_df
from homework_week_2.mmlu.task5_compare_two_models.compare_two_models import significance_by_paired_ttest
from homework_week_2.mmlu.task6_estimate_the_accuracy_gap.estimate_accuracy_gap import ci_accuracy_for_difference

LOGS_DIR = Path(__file__).parent.parent / "task5_compare_two_models" / "logs"

log_a = read_eval_log(str(LOGS_DIR / "2026-03-23T13-28-28+00-00_mmlu-subset_4K4PnFEhHmxVWXdrYZTHpU.eval"))
log_b = read_eval_log(str(LOGS_DIR / "2026-03-23T13-44-32+00-00_mmlu-subset_UiPNzyJnXhhELTtoHo8sbg.eval"))

print(f"Model A: {log_a.eval.model} ({len(log_a.samples)} samples)")
print(f"Model B: {log_b.eval.model} ({len(log_b.samples)} samples)\n")

scores_a = log_to_df(log_a).groupby("id")["score"].mean().sort_index().values
scores_b = log_to_df(log_b).groupby("id")["score"].mean().sort_index().values

lower, mean_diff, upper = ci_accuracy_for_difference(scores_a, scores_b)

print(f"95% CI for accuracy difference (A - B):")
print(f"  Mean difference: {mean_diff:+.4f}")
print(f"  95% CI:          [{lower:+.4f}, {upper:+.4f}]")
print(f"  Contains zero:   {lower < 0 < upper}")

p_value, _, is_significant = significance_by_paired_ttest(scores_a, scores_b)
print(f"\nДля сравнения (задание 5):")
print(f"  p-value:         {p_value:.4f}")
print(f"  Significant:     {is_significant}")
print()
if lower < 0 < upper:
    print("Интервал содержит 0 -> нет статистически значимой разницы между моделями.")
else:
    direction = "A лучше B" if mean_diff > 0 else "B лучше A"
    print(f"Интервал не содержит 0 -> {direction} (значимо на уровне 95%).")

# Result:
# Model A: ollama/llama2 (50 samples)
# Model B: ollama/qwen2:latest (50 samples)
#
# 95% CI for accuracy difference (A - B):
#   Mean difference: -0.3200
#   95% CI:          [-0.4827, -0.1573]
#   Contains zero:   False
#
# Для сравнения (задание 5):
#   p-value:         0.0003
#   Significant:     True
#
# Интервал не содержит 0 -> B лучше A (значимо на уровне 95%).
