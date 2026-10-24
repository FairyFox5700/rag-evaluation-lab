import json

import pytest

from rag_eval_lab.io import read_jsonl, write_jsonl
from rag_eval_lab.models import EvalCase


def test_jsonl_round_trip_preserves_validated_records(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    records = [
        EvalCase(id="q1", question="Question 1?", expected_answer="Answer 1"),
        EvalCase(id="q2", question="Question 2?", expected_answer="Answer 2"),
    ]

    write_jsonl(path, records)
    loaded = read_jsonl(path, EvalCase)

    assert loaded == records
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_jsonl_reader_reports_the_bad_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "ok", "question": "Q", "expected_answer": "A"}) + "\n{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path, EvalCase)
