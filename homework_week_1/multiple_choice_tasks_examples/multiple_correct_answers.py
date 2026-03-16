from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import (
    multiple_choice, system_message
)

@task
def mc_multiple_correct():
    return Task(
        dataset=[
            Sample(
                input="Which are programming languages?",
                choices=["Python", "HTML", "JavaScript", "CSS"],
                target=["A", "C"]  # Python, JavaScript
            ),
            Sample(
                input="Which continents border the Atlantic Ocean?",
                choices=["Africa", "Asia", "Europe", "South America"],
                target=["A", "C", "D"]  # Africa, Europe, South America
            ),
        ],
        solver=[
            system_message("Select ALL correct answers. You may choose multiple options."),
            multiple_choice(multiple_correct=True)
        ],
        scorer=choice(),
    )

eval(mc_multiple_correct, model="ollama/llama2")