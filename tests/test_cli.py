import json

import pymupdf

from rag_eval_lab.cli import main
from rag_eval_lab.io import write_jsonl
from rag_eval_lab.models import EvalCase


def test_demo_command_runs_end_to_end_without_api_calls(tmp_path) -> None:
    pdf_path = tmp_path / "knowledge.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Penguins swim using strong flippers.")
    document.save(pdf_path)
    document.close()

    dataset = tmp_path / "dataset.jsonl"
    write_jsonl(
        dataset,
        [
            EvalCase(
                id="swim",
                question="How do penguins swim?",
                expected_answer="Penguins swim using strong flippers.",
            )
        ],
    )
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "demo",
            "--pdf",
            str(pdf_path),
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--experiment",
            "candidate",
            "--k",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "scored.jsonl").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "candidate"
    assert summary["test_count"] == 1
