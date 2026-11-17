"""Read the source resume (DOCX) as plain text.

The source file is only ever read here. Tailored resumes are written elsewhere,
as copies.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document


def resume_text(path: str | Path) -> str:
    """Return the resume's text, paragraphs and table cells in document order."""
    doc = Document(str(path))
    blocks: list[str] = []

    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            for paragraph in doc.paragraphs:
                if paragraph._p is child:
                    text = paragraph.text.strip()
                    if text:
                        blocks.append(text)
                    break
        elif tag == "tbl":
            for table in doc.tables:
                if table._tbl is child:
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        if any(cells):
                            blocks.append(" | ".join(cells))
                    break

    if not blocks:  # fallback: paragraphs only
        blocks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(blocks)
