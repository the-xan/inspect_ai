from inspect_ai import Task, task
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice


@task
def mmlu_subset(subset):
    """Минимальная MMLU-задача для любого subset датасета."""
    return Task(
        dataset=subset,
        solver=[multiple_choice()],
        scorer=choice()
    )


@task
def mmlu_subset_cot(subset):
    """MMLU-задача с chain-of-thought рассуждением перед ответом."""
    return Task(
        dataset=subset,
        solver=[multiple_choice(cot=True)],
        scorer=choice()
    )
