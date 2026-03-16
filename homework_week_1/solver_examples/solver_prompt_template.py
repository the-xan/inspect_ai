from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, system_message, prompt_template
)

STEP_BY_STEP_TEMPLATE = '''
  Solve this problem step by step:

  Problem: {prompt}

  Structure:
  1. Understand the problem
  2. Plan your approach
  3. Solve it
  4. Final answer format: ANSWER: <value>
  '''.strip()


@task
def example_prompt_template():
    """
    Демонстрирует солвер prompt_template().
    Шаблон добавляет структуру к промпту.

    Назначение: подставлять переменные в шаблон для
    переформатирования промптов.

      Когда использовать:
      - добавить требования к формату вывода
      - включить примеры или демонстрации
      - структурировать промпты единообразно
      - добавить шаги рассуждения
    """
    return Task(
        dataset=[
            Sample(input="What is 25 * 4?", target="100"),
            Sample(input="What is 144 / 12?", target="12"),
            Sample(input="What is 15 * 8?", target="120")
        ],
        solver=[
            system_message("You are a math tutor."),
            prompt_template(STEP_BY_STEP_TEMPLATE),
            generate()
        ],
        scorer=match(numeric=True),
    )


# Запустите и посмотрите, как шаблон структурирует промпт
eval(example_prompt_template, model="ollama/llama2")
