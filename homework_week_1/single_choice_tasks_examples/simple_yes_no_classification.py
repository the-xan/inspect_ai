from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import (
    generate, system_message
)


@task
def yes_no_classification():
    return Task(
        dataset=[
            Sample(
                input="Is Python a programming language?",
                target="Yes"
            ),
            Sample(
                input="Is water dry?",
                target="No"
            ),
            Sample(
                input="Is the Earth round?",
                target="Yes"
            ),
        ],
        solver=[
            system_message("Answer 'Yes' or 'No'. Be concise."),
            generate()
        ],
        scorer=exact(),
    )


eval(yes_no_classification, model="ollama/llama2")
