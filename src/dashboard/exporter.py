"""Export helpers for dashboard data."""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List

import pandas as pd
from fpdf import FPDF


def experiments_to_csv(experiments: List[Dict]) -> bytes:
    df = pd.json_normalize(experiments)
    return df.to_csv(index=False).encode("utf-8")


def build_experiments_pdf(experiments: List[Dict], title: str = "Experiment Summary") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True)

    pdf.set_font("Arial", size=10)
    for exp in experiments[:50]:
        line = f"{exp.get('run_id', 'N/A')} | {exp.get('model_id', 'model')} | Acc: {exp.get('validation_metrics', {}).get('accuracy', 'N/A')}"
        pdf.multi_cell(0, 7, txt=line)
        pdf.ln(1)

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


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
    pdf.cell(0, 10, title, ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Metrics", ln=True)
    pdf.set_font("Arial", size=10)
    for key, value in metrics.items():
        pdf.cell(0, 6, f"- {key}: {value}", ln=True)

    for table_name, rows in tables.items():
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, table_name, ln=True)
        pdf.set_font("Arial", size=10)
        for row in rows[:25]:
            line = " | ".join(f"{k}: {v}" for k, v in row.items())
            pdf.multi_cell(0, 6, line)
            pdf.ln(1)

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
