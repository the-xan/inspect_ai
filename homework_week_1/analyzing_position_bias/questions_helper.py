import random

from inspect_ai.dataset import Sample

# For reproducibility
random.seed(42)


def generate_questions(n: int) -> list[tuple[str, int]]:
    """
    Generate n simple math problems.

    Args:
        n: number of problems to generate

    Returns:
        List of (question_text, correct_answer) tuples
    """
    problems = []

    for _ in range(n):
        # YOUR CODE HERE
        # Generate a simple addition or subtraction problem
        # Hint: use random.randint() for numbers, random.choice() for operation
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        operation = random.choice(["+", "-"])
        answer = a + b if operation == "+" else a - b
        problems.append((f"What is {a} {operation} {b}?", str(answer)))

    return problems


def generate_distractors(correct: str, n: int = 3) -> list[str]:
    """
    Generate n plausible wrong answers.

    For numeric answers: generates nearby numbers.
    For other types: you'll need to customize this.

    Args:
        correct: the correct answer (string)
        n: number of distractors to generate

    Returns:
        List of n distinct wrong answers (strings)
    """
    distractors = set()

    offsets = [-3, -2, -1, 1, 2, 3]

    while len(distractors) < n:
        offset = random.choice(offsets)
        candidate = str(int(correct) + offset)
        if candidate != correct:
            distractors.add(candidate)

    return list(distractors)


def create_samples(
        questions: list[tuple[str, str]],
        correct_position: int | None = None
) -> list[Sample]:
    """
    Convert questions to multiple-choice Samples.

    Args:
        questions: list of (question_text, correct_answer) tuples
        correct_position:
            None → randomize position (A/B/C/D) for each question
            0 → correct answer always at position A
            1 → correct answer always at position B
            2 → correct answer always at position C
            3 → correct answer always at position D

    Returns:
        List of Sample objects ready for Inspect AI.
        Each Sample has:
            - input: str (the question)
            - choices: list[str] (4 options, no letters)
            - target: str (correct letter: "A", "B", "C", or "D")
    """
    samples = []

    for question, correct in questions:

        # 1. Generate 3 distractors (use generate_distractors() function)
        # 2. Build list of 4 options
        # 3. Place correct answer at the right position:
        #    - If correct_position is None → put all options together and shuffle
        #    - Otherwise → put correct at that index
        # 4. Determine target letter based on where correct ended up
        # 5. Create Sample(input=..., choices=..., target=...)
        distractors = generate_distractors(correct)
        choices = distractors + [correct]

        if correct_position is None:
            random.shuffle(choices)
        else:
            choices.remove(correct)
            choices.insert(correct_position, correct)

        target = "ABCD"[choices.index(correct)]
        samples.append(Sample(input=question, choices=choices, target=target))

    return samples









# ===== TESTS =====
test_q = [("What is 2 + 2?", "4"), ("What is 10 - 3?", "7"), ("What is 5 + 5?", "10")]
samples_fixed = create_samples(test_q, correct_position=0)
samples_random = create_samples(test_q, correct_position=None)

assert len(samples_fixed) == len(test_q), f"Expected {len(test_q)} samples, got {len(samples_fixed)}"
assert all(hasattr(s, 'input') and hasattr(s, 'choices') and hasattr(s, 'target') for s in
           samples_fixed), "Each sample must have 'input', 'choices', and 'target' attributes"
assert all(len(s.choices) == 4 for s in samples_fixed), "Each sample must have exactly 4 choices"
assert all(s.target == "A" for s in samples_fixed), "With correct_position=0, all targets should be 'A'"
assert all(s.choices[0] == correct for s, (_, correct) in
           zip(samples_fixed, test_q)), "With correct_position=0, correct answer should be first in choices"
assert all(s.target in "ABCD" for s in samples_random), "Target must be one of A, B, C, D"

# Check that correct answer is actually at the target position
for s, (_, correct) in zip(samples_random, test_q):
    target_index = "ABCD".index(s.target)
    assert s.choices[
               target_index] == correct, f"Correct answer '{correct}' should be at position {s.target}, but found '{s.choices[target_index]}'"

# ===== TESTS =====
test_distractors = generate_distractors("10", n=3)

assert len(test_distractors) == 3, f"Expected 3 distractors, got {len(test_distractors)}"
assert all(isinstance(d, str) for d in test_distractors), "All distractors must be strings"
assert "10" not in test_distractors, "Distractors must not include the correct answer"
assert len(set(test_distractors)) == 3, "All distractors must be unique"

print(f"   Distractors for '10': {test_distractors}")

# ===== TESTS =====
test_questions = generate_questions(5)

assert len(test_questions) == 5, f"Expected 5 questions, got {len(test_questions)}"
assert all(isinstance(q, tuple) and len(q) == 2 for q in
           test_questions), "Each question must be a tuple of (question_text, answer)"
assert all(
    isinstance(q[0], str) and isinstance(q[1], str) for q in test_questions), "Both question and answer must be strings"
assert all(len(q[0]) > 0 and len(q[1]) > 0 for q in test_questions), "Question and answer cannot be empty"

print("\nSample output:")
for q, a in test_questions:
    print(f"  {q} → {a}")
