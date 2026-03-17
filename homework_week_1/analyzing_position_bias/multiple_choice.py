import random

from inspect_ai import Task, task, eval
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message

from homework_week_1.analyzing_position_bias.questions_helper import create_samples, generate_questions


@task
def position_bias_task(
        questions: list[tuple[str, str]],
        correct_position: int | None = None
):
    """
    Multiple choice evaluation task.

    Args:
        questions: list of (question_text, correct_answer) tuples
        correct_position: None for random, 0-3 for fixed position
    """
    samples = create_samples(questions, correct_position)

    return Task(
        dataset=samples,
        solver=[
            system_message("Select correct answer"),
            multiple_choice()
        ],
        scorer=choice()
    )


MODEL = "ollama/llama2"
N_QUESTIONS = 10

random.seed(42)
questions = generate_questions(N_QUESTIONS)

eval(
    position_bias_task,
    model=MODEL,
    task_args={"questions": questions, "correct_position": 0}
)

eval(
    position_bias_task,
    model=MODEL,
    task_args={"questions": questions, "correct_position":None}
)
