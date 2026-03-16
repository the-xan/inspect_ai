from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import (
    multiple_choice
)

"""
Ключевые правила:
  - choices: список вариантов ответа (без букв — они добавляются
   автоматически)
  - target: буква правильного ответа ("A", "B", "C" или "D")
  - Используйте солвер multiple_choice() + скорер choice()
  
  
"""
@task
def mc_with_metadata():
    return Task(
        dataset=[
            Sample(
                input="Capital of Japan?",
                choices=["Seoul", "Tokyo", "Bangkok", "Beijing"],
                target="B",
                metadata={
                    "difficulty": "easy",
                    "category": "geography"
                }
            ),
            Sample(
                input="What is the Heisenberg Uncertainty Principle?",
                choices=[
                    "Cannot know both position and momentum precisely",
                    "Energy cannot be created or destroyed",
                    "All matter has wave-particle duality",
                    "Time always moves forward"
                ],
                target="A",
                metadata={
                    "difficulty": "hard",
                    "category": "physics"
                }
            ),
        ],
        solver=multiple_choice(),
        scorer=choice(),
    )


# Запустите и проверьте результаты в inspect view — фильтруйте по метаданным!
eval(mc_with_metadata, model="ollama/llama2")
