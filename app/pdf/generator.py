"""Render the interview Q&A PDF from stored records."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.models import PdfDocumentData, StoredQA

TEMPLATE_DIR = Path(__file__).resolve().parent
LATIN_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def render_html(data: PdfDocumentData) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("template.html")
    return template.render(
        creator=data.creator,
        generated=data.generated_at.strftime("%Y-%m-%d"),
        categories=data.categories,
    )


def _ascii_safe(value: str) -> str:
    return value.translate(LATIN_REPLACEMENTS).encode("latin-1", "replace").decode("latin-1")


def generate_pdf(data: PdfDocumentData, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(data)
    try:
        from weasyprint import HTML

        HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(str(output_path))
        return output_path
    except Exception:
        return _generate_with_fpdf(data, output_path)


def _mc(pdf, text: str, height: float = 6) -> None:
    """Write a wrapping line and return the cursor to the left margin.

    fpdf2 defaults multi_cell to new_x=RIGHT, which leaves no room for the next line.
    """
    pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


def _generate_with_fpdf(data: PdfDocumentData, output_path: Path) -> Path:
    from fpdf import FPDF

    class NumberedPDF(FPDF):
        def footer(self) -> None:
            self.set_y(-15)
            self.set_x(self.l_margin)
            self.set_font("Helvetica", size=9)
            self.set_text_color(102, 112, 133)
            self.cell(0, 10, str(self.page_no()), align="C")

    pdf = NumberedPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_title("LinkedIn Interview Questions & Answers")

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(10, 102, 194)
    _mc(pdf, "LinkedIn Interview Questions & Answers", 10)
    pdf.ln(2)
    pdf.set_draw_color(10, 102, 194)
    pdf.set_line_width(0.8)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(6)

    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(71, 84, 103)
    _mc(pdf, _ascii_safe(f"Source: {data.creator}"))
    _mc(pdf, f"Generated: {data.generated_at.strftime('%Y-%m-%d')}")
    pdf.ln(6)

    if not data.categories:
        pdf.set_text_color(102, 112, 133)
        pdf.set_font("Helvetica", "I", 11)
        _mc(pdf, "No answered interview Q&A pairs were found.")

    for category, items in data.categories.items():
        pdf.set_text_color(16, 24, 40)
        pdf.set_font("Helvetica", "B", 16)
        pdf.ln(4)
        _mc(pdf, _ascii_safe(category), 8)
        pdf.set_draw_color(10, 102, 194)
        pdf.set_line_width(0.4)
        y = pdf.get_y()
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(6)
        for index, item in enumerate(items, start=1):
            _write_qa(pdf, index, item)

    pdf.output(str(output_path))
    return output_path


def _write_qa(pdf, index: int, item: StoredQA) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(16, 24, 40)
    _mc(pdf, _ascii_safe(f"Q{index}. {item.question}"))
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    _mc(pdf, "Answer:")
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(29, 41, 57)
    _mc(pdf, _ascii_safe(item.answer))
    if item.source_url:
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(102, 112, 133)
        _mc(pdf, _ascii_safe(f"Source: {item.source_url}"), 5)
    pdf.ln(4)
