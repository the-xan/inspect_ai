from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate
)


@task
def hello_model():
    """Test your model setup with simple questions."""
    return Task(
        dataset=[
            Sample(
                input="Say 'Hello world!' and nothing else.",
                target="Hello world!"
            ),
            Sample(
                input="2+2=",
                target="4"
            ),
            Sample(
                input="What is the surname of Sheldon from The Big Bang Theory?",
                target="Cooper"
            ),

            Sample(
                input="What is the capital of the Kazakhstan?",
                target="Astana"
            )
        ],
        solver=[generate()],
        scorer=match(
            location="end",  # where to look for the answer: "begin", "end", "any", "exact"
            ignore_case=True,  # ignore case when comparing
            numeric=False  # treat as numeric comparison (normalizes numbers, different punctuation rules)
        )
    )

eval(
    hello_model,
    model="ollama/llama2",
    # limit=1  # Uncomment to test with just 1 sample
)

# inspect view --log-dir homework_week_1/hello_world/logs - логи