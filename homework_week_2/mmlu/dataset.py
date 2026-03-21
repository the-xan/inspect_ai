from string import ascii_uppercase

from inspect_ai.dataset import Sample, hf_dataset


def record_to_sample(record: dict) -> Sample:
    """
    Конвертирует сырую запись MMLU в inspect_ai Sample.

    MMLU хранит правильный ответ как целое число (0=A, 1=B, 2=C, 3=D).
    Конвертируем в букву, чтобы совпадало с форматом scorer choice().
    """
    answer_idx = int(record["answer"])
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=ascii_uppercase[answer_idx],  # 0->'A', 1->'B', ...
        metadata=dict(subject=record.get("subject"))
    )


def load_dataset():
    """Загружает полный датасет MMLU (test split) с кешированием."""
    return hf_dataset(
        path="cais/mmlu",
        name="all",
        split="test",
        sample_fields=record_to_sample,
        cached=True
    )


def filter_subset(dataset, subject: str):
    """Фильтрует датасет по теме (subject)."""
    return dataset.filter(
        lambda s: s.metadata.get("subject") == subject
    )
