"""Export helpers for dashboard data."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd  # type: ignore[import-untyped]
from fpdf import FPDF  # type: ignore[import-untyped]


def _safe(text: Any) -> str:
    return str(text).encode("latin-1", "ignore").decode("latin-1")


def experiments_to_csv(experiments: List[Dict]) -> bytes:
    df = pd.json_normalize(experiments)
    return df.to_csv(index=False).encode("utf-8")


def _pdf_bytes(pdf: FPDF) -> bytes:
    """Return the PDF contents as bytes regardless of FPDF backend behaviour."""
    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    if isinstance(output, str):
        return output.encode("latin-1")
    raise TypeError(f"Unexpected type from FPDF.output: {type(output)!r}")


def build_experiments_pdf(experiments: List[Dict], title: str = "Experiment Summary") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _safe(title), ln=True)

    pdf.set_font("Arial", size=10)
    for exp in experiments[:50]:
        line = f"{exp.get('run_id', 'N/A')} | {exp.get('model_id', 'model')} | Acc: {exp.get('validation_metrics', {}).get('accuracy', 'N/A')}"
        pdf.multi_cell(0, 7, txt=_safe(line))
        pdf.ln(1)

    return _pdf_bytes(pdf)


def analytics_to_csv(data: Dict[str, List[Dict]]) -> bytes:
    csv_parts = []
    for name, rows in data.items():
        df = pd.DataFrame(rows)
        csv_parts.append(f"## {name}\n" + df.to_csv(index=False))
    return "\n".join(csv_parts).encode("utf-8")


def analytics_to_pdf(title: str, metrics: Dict[str, str], tables: Dict[str, List[Dict]]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _safe(title), ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Metrics", ln=True)
    pdf.set_font("Arial", size=10)
    for key, value in metrics.items():
        pdf.cell(0, 6, _safe(f"- {key}: {value}"), ln=True)

    for table_name, rows in tables.items():
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, table_name, ln=True)
        pdf.set_font("Arial", size=10)
        for row in rows[:25]:
            line = " | ".join(f"{k}: {v}" for k, v in row.items())
            pdf.multi_cell(0, 6, _safe(line))
            pdf.ln(1)

    return _pdf_bytes(pdf)
