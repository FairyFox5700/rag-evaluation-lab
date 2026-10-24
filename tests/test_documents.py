import pymupdf

from rag_eval_lab.documents import chunk_paragraphs, extract_paragraphs


def test_extract_paragraphs_reads_nonempty_pdf_blocks(tmp_path) -> None:
    pdf_path = tmp_path / "knowledge.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Penguins are flightless seabirds.\nThey swim with flippers.")
    document.save(pdf_path)
    document.close()

    paragraphs = extract_paragraphs(pdf_path)

    assert paragraphs == ["Penguins are flightless seabirds. They swim with flippers."]


def test_chunk_paragraphs_respects_size_and_overlap() -> None:
    paragraphs = ["one two three four five six seven eight nine ten"]

    chunks = chunk_paragraphs(paragraphs, chunk_size=24, chunk_overlap=8)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 24 for chunk in chunks)
    assert "one two" in chunks[0]
    assert "ten" in chunks[-1]
