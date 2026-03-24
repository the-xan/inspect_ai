"""
Assignment 8: required_sample_size.

Использует компоненты дисперсии из task7 (логи уже сохранены),
вычисляет required_sample_size для delta = 5% и 10%,
проверяет round-trip с minimum_detectable_effect.
"""
from pathlib import Path

from inspect_ai.log import read_eval_log

from homework_week_2.mmlu.config import MY_SUBSET
from homework_week_2.mmlu.dataset import load_dataset, filter_subset
from homework_week_2.mmlu.task7_estimate_variance_components.variance_components import (
    estimate_variance_components,
)
from homework_week_2.mmlu.task8_inplement_required_sample_size.required_sample_size import (
    minimum_detectable_effect,
    required_sample_size,
)

LOGS_DIR = Path(__file__).parent.parent / "task7_estimate_variance_components" / "logs"

log_files = sorted(LOGS_DIR.glob("*.eval"))
if len(log_files) < 2:
    raise FileNotFoundError(
        f"Нужно 2 лога в {LOGS_DIR}. Запусти сначала task7/run_eval.py"
    )

log_a = read_eval_log(str(log_files[0]))
log_b = read_eval_log(str(log_files[1]))

print(f"Model A: {log_a.eval.model} ({len(log_a.samples)} samples)")
print(f"Model B: {log_b.eval.model} ({len(log_b.samples)} samples)\n")

params = estimate_variance_components([log_a], [log_b])
print(f"omega2   = {params['omega2']:.4f}")
print(f"sigma2_A = {params['sigma2_a']:.4f}")
print(f"sigma2_B = {params['sigma2_b']:.4f}")

subset = filter_subset(load_dataset(), MY_SUBSET)
n_subset = len(subset)
mde = minimum_detectable_effect(n=n_subset, **params)
print(f"\nС n={n_subset} вопросов -> MDE = {mde:.1%}")
print("(наименьшая разница, обнаруживаемая с мощностью 80%, alpha=0.05)\n")

print("--- Требуемый размер выборки ---")
for delta in (0.05, 0.10):
    n_needed = required_sample_size(delta=delta, **params)
    mde_check = minimum_detectable_effect(n=n_needed, **params)
    print(f"  delta={delta:.0%}: нужно {n_needed} вопросов  (MDE check: {mde_check:.4f})")

# Round-trip проверка
n5 = required_sample_size(delta=0.05, **params)
mde5 = minimum_detectable_effect(n=n5, **params)
assert abs(mde5 - 0.05) < 0.005, f"Round-trip failed: MDE={mde5:.4f}"
print("\nRound-trip check passed!")

print(f"\nВыводы:")
print(f"  MY_SUBSET содержит {n_subset} вопросов.")
n5  = required_sample_size(delta=0.05, **params)
n10 = required_sample_size(delta=0.10, **params)
if n_subset >= n5:
    print(f"  Достаточно для обнаружения разницы 5% (нужно {n5}).")
elif n_subset >= n10:
    print(f"  Достаточно для 10%, но не для 5% (нужно {n5} vs {n_subset}).")
else:
    print(f"  Недостаточно даже для 10% (нужно {n10} vs {n_subset}).")

# Result
# Model A: ollama/llama2 (30 samples)
# Model B: ollama/qwen2:latest (30 samples)
#
# omega2   = 0.2333
# sigma2_A = 0.2000
# sigma2_B = 0.0333
#
# С n=131 вопросов -> MDE = 16.7%
# (наименьшая разница, обнаруживаемая с мощностью 80%, alpha=0.05)
#
# --- Требуемый размер выборки ---
#   delta=5%: нужно 1466 вопросов  (MDE check: 0.0500)
#   delta=10%: нужно 367 вопросов  (MDE check: 0.0999)
#
# Round-trip check passed!
#
# Выводы:
#   MY_SUBSET содержит 131 вопросов.
#   Недостаточно даже для 10% (нужно 367 vs 131).