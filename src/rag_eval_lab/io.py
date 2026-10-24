"""Small, strict JSONL helpers for reproducible artifacts."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def read_jsonl[ModelT: BaseModel](path: str | Path, model: type[ModelT]) -> list[ModelT]:
    source = Path(path)
    records: list[ModelT] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid {source} line {line_number}: {error}") from error
    return records
