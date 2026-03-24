"""
Assignment 7: Estimate variance components & power analysis.

Запускает пилотные прогоны (epochs=2) для двух моделей, оценивает компоненты
дисперсии и вычисляет MDE и требуемый размер выборки.
"""
from inspect_ai import eval

from homework_week_2.mmlu.config import MODEL_A, MODEL_B, MY_SUBSET
from homework_week_2.mmlu.dataset import load_dataset, filter_subset
from homework_week_2.mmlu.tasks import mmlu_subset
from homework_week_2.mmlu.task7_estimate_variance_components.variance_components import (
    estimate_variance_components,
    minimum_detectable_effect,
    required_sample_size,
)

PILOT_LIMIT = 15  # число вопросов для пилота (как в ноутбуке)

subset = filter_subset(load_dataset(), MY_SUBSET)

print(f"Subset: '{MY_SUBSET}' — {len(subset)} questions total")
print(f"Running pilot evals (limit={PILOT_LIMIT}, epochs=2) ...")

logs_a = eval(mmlu_subset(subset), model=MODEL_A, epochs=2, limit=PILOT_LIMIT)
logs_b = eval(mmlu_subset(subset), model=MODEL_B, epochs=2, limit=PILOT_LIMIT)

params = estimate_variance_components(logs_a, logs_b)
print(f"\nomega2   = {params['omega2']:.4f}")
print(f"sigma2_A = {params['sigma2_a']:.4f}")
print(f"sigma2_B = {params['sigma2_b']:.4f}")

mde = minimum_detectable_effect(n=len(subset), **params)
print(f"\nWith n={len(subset)} questions -> MDE = {mde:.1%}")
print("(наименьшая разница, обнаруживаемая с мощностью 80%, alpha=0.05)")

print("\n--- Required sample sizes ---")
for delta in (0.05, 0.10):
    n_needed = required_sample_size(delta=delta, **params)
    mde_check = minimum_detectable_effect(n=n_needed, **params)
    print(f"  delta={delta:.0%}: нужно {n_needed} вопросов  (MDE check: {mde_check:.3f})")

# Round-trip self-check
n5 = required_sample_size(delta=0.05, **params)
mde5 = minimum_detectable_effect(n=n5, **params)
assert abs(mde5 - 0.05) < 0.005, f"Round-trip failed: MDE={mde5:.4f}"
print("\nRound-trip check passed!")

# Results:
# epochs: 2, subset: cais / mmlu, dataset: cais / mmlu
#
# total time: 0:05: 33
# ollama / qwen2: latest - 3, 582 tokens[I: 3, 434, O: 148]
#
# choice
# accuracy 0.767
# stderr 0.108
#
#
# omega2 = 0.2333
# sigma2_A = 0.2000
# sigma2_B = 0.0333
#
# With n = 131 questions -> MDE = 16.7 %
# (наименьшая разница, обнаруживаемая с мощностью 80 %, alpha=0.05)
#
# --- Required sample sizes - --
# delta = 5 %: нужно 1466 вопросов(MDE check: 0.050)
# delta = 10 %: нужно 367 вопросов(MDE check: 0.100)