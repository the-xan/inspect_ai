from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, system_message, chain_of_thought
)


@task
def sentiment_classification():
    return Task(
        dataset=[
            Sample(
                input="I love this course",
                target="Positive",
                metadata={
                    "category": "home-work"
                }
            ),
            Sample(
                input="I hate this course - the assignments were too complicated",
                target="Negative",
                metadata={
                    "category": "home-work"
                }
            ),
            Sample(
                input="Well, this course is not bad",
                target="Neutral",
                metadata={
                    "category": "home-work"
                }
            ),
            Sample(
                input="Amazing! This course is absolutely insane!",
                target="Positive",
                metadata={
                    "category": "home-work"
                }
            )
        ],
        solver=[
            system_message(
                "Classify the sentiment. Reply with exactly one word: ANSWER: < Positive, Negative, or Neutral > "),
            generate()
        ],
        scorer=match(ignore_case=True)
    )


@task
def sport_classification():
    return Task(
        dataset=[
            Sample(
                input="Two teams of eleven players play on a large grass field with a round ball. They mostly kick the ball and try to score in a goal while a goalkeeper protects the net.",
                target="Football",
                metadata={
                    "category": "home-work"
                }
            ),
            Sample(
                input="Players use a long wooden stick to hit a small white ball. One player throws the ball while another tries to send it far across the field.",
                target="Baseball",
                metadata={
                    "category": "home-work"
                }
            ),
            Sample(
                input="A person stands on a smooth ice surface wearing special boots with wheels. They move quickly and try to send a small black disk into the opponent’s net.",
                target="Ice Hockey",
                metadata={
                    "category": "home-work"
                }
            ),
            Sample(
                input="Two athletes stand on opposite sides of a high net and hit a light ball with their hands. They try to make the ball touch the ground on the other side",
                target="Volleyball",
                metadata={
                    "category": "home-work"
                }
            )
        ],
        solver=[
            system_message(
                "Tell what sport the text talks about. Reply with exactly one word: ANSWER: < Volleyball, Ice Hockey, Baseball or Football > "),
            chain_of_thought(),
            generate()
        ],
        scorer=match(ignore_case=True)
    )


# eval(sentiment_classification, model="ollama/llama2")
eval(sport_classification, model="ollama/llama2")
