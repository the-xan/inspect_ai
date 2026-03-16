from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import (
    multiple_choice
)


@task
def example_multiple_choice_with_cot():
    """
    Демонстрирует multiple_choice(cot=True).
    Модель рассуждает перед выбором ответа.

    Специальный солвер для вопросов A/B/C/D. Автоматически
      форматирует вопрос и извлекает ответ.

    Когда использовать:
      - вопросы с множественным выбором (вместо generate)
      - цель должна быть буквой: "A", "B", "C" и т.д.

    Важно: при использовании multiple_choice() в качестве скорера
      используйте choice().
    """
    return Task(
        dataset=[
            Sample(
                input="Light travels faster than sound. If you see lightning and hear thunder 3 seconds later, approximately how far away was the strike?",
                choices=["100 meters", "1 kilometer", "3 kilometers", "10 kilometers"],
                target="B"  # ~1 км (звук ~340 м/с)
            ),
        ],
        solver=multiple_choice(cot=True),
        scorer=choice(),
    )


eval(example_multiple_choice_with_cot, model="ollama/llama2")
