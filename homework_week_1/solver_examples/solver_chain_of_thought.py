from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, system_message, chain_of_thought
)


@task
def example_chain_of_thought():
    """
    Демонстрирует солвер chain_of_thought().
    Сравните точность с CoT и без него в inspect view.

    Назначение: просить модель «думать шаг за шагом» перед
      ответом.

    Когда использовать:
      - математические и логические задачи
      - многошаговые задачи на рассуждение
      - когда нужно видеть процесс мышления модели
    """
    return Task(
        dataset=[
            Sample(
                input="If Alice has 3 apples and Bob gives her 2 more, how many does she have?",
                target="5"
            ),
            Sample(
                input="A train travels 100 km in 2 hours. At this rate, how far in 5 hours?",
                target="250"
            ),
        ],
        solver=[
            system_message("Solve the problem. End with: ANSWER: < number > "),
            chain_of_thought(),
            generate()
        ],
        scorer=match(numeric=True),
    )


eval(example_chain_of_thought, model="ollama/llama2")
