from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact, match, pattern
from inspect_ai.solver import (
    generate, system_message, chain_of_thought, prompt_template
)


@task
def choice_with_reasoning():
    PROMPT = '''
Classify as True or False:

Statement: {prompt}

Provide:
1. REASONING: [Your explanation]
2. ANSWER: [True or False]
    '''.strip()

    return Task(
        dataset=[
            Sample(
                input="The Earth is flat.",
                target="False"
            ),
            Sample(
                input="Water boils at 100°C at sea level.",
                target="True"
            ),
        ],
        solver=[
            chain_of_thought(),
            prompt_template(PROMPT),
            generate()
        ],
        scorer=pattern(r'ANSWER:\s*(True|False)'),
    )


eval(choice_with_reasoning, model='ollama/llama2')
