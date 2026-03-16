from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, system_message
)


@task
def example_system_message():
    """
    Демонстрирует солвер system_message().
    Системный промпт требует краткости.
    Назначение: добавить системное сообщение для управления
    поведением модели.

    Когда использовать:
      - задать роль или персону модели
      - установить глобальные правила или ограничения
      - определить контекст оценки
    """
    return Task(
        dataset=[
            Sample(input="What is 15 * 8?", target="120"),
            Sample(input="What is 99 + 1?", target="100"),
        ],
        solver=[
            system_message("You are a calculator. Reply with only the number, nothing else."),
            generate()
        ],
        scorer=match(numeric=True),
    )


# Запустите и проверьте вкладку Messages в inspect view
eval(example_system_message, model="ollama/llama2")
