from homework_week_2.mmlu.config import MODEL_A, MODEL_B, MY_SUBSET
from homework_week_2.mmlu.dataset import filter_subset, load_dataset
from homework_week_2.mmlu.task5_compare_two_models.compare_two_models import compare_models_paired

dataset = load_dataset()
subset = filter_subset(dataset, MY_SUBSET)

print(f"Comparing {MODEL_A} vs {MODEL_B} on '{MY_SUBSET}' ({len(subset)} questions)...\n")

p_value, mean_diff, is_significant = compare_models_paired(
    MODEL_A, MODEL_B, subset[:50]  # первые 50 вопросов
)

print(f"\nResults:")
print(f"  Mean difference (A - B): {mean_diff:+.4f}")
print(f"  p-value:                 {p_value:.4f}")
print(f"  Significant (α=0.05):   {is_significant}")

# Results:
#   Mean difference (A - B): -0.3200
#   p-value:                 0.0003
#   Significant (α=0.05):   True
