"""Render the Markdown reports to PDF (python-markdown -> HTML + CSS -> headless Chrome). Run: python -m sunfix.md2pdf
Output: results/reports/*.pdf. Key figures are appended to the matching report."""
import subprocess
import tempfile
from pathlib import Path

import markdown

from .load import ROOT

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "results" / "reports"
FIG = ROOT / "results" / "figures"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.2pt; line-height: 1.42; color: #111; }
h1 { font-size: 19pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 16pt 0 4pt; color: #1b5fb4; border-bottom: 1px solid #d8e2f0; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 3pt; } h1, h2, h3 { page-break-after: avoid; } p { margin: 4pt 0; } li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccd; padding: 3pt 5pt; text-align: left; vertical-align: top; } th { background: #eef4fc; }
code { font-family: Menlo, monospace; font-size: 8.6pt; background: #f3f3f3; padding: 0 2pt; }
pre { background: #f3f3f3; padding: 6pt; font-size: 8.4pt; white-space: pre-wrap; }
img { max-width: 100%; margin: 6pt 0; page-break-inside: avoid; } .fig { page-break-inside: avoid; } .cap { color: #555; font-size: 8.5pt; }
"""
REPORTS = {   # markdown file -> (pdf name, figures)
    "REPORT.md": ("SunFix_final_report", ["phase3_summary.png", "phase4_minimum_data_sub40.png"]),
    "docs/phase3_report.md": ("Phase3_EOL_prediction", ["phase3_summary.png"]),
    "docs/phase4_report.md": ("Phase4_minimum_data", ["phase4_minimum_data_sub40.png"]),
    "docs/phase6a_report.md": ("Phase6a_cost_aware_evaluation", ["cost_vs_threshold.png", "leadtime_hist.png", "calibration_bboxx.png", "breakeven_bboxx.png"]),
    "docs/phase6b_report.md": ("Phase6b_benchmark_models", ["shap_lightgbm_bboxx.png"]),
    "docs/phase6c_report.md": ("Phase6c_conformal_persistence", ["conformal_far_by_month.png", "persistence_effect.png"]),
    "docs/phase6d_report.md": ("Phase6d_LFP_transfer", []),
    "docs/phase2_report.md": ("Phase2_resistance_model", ["phase2_trajectories_ocv.png", "phase2_operating_point_slices_ocv.png"]),
    "docs/eda_report.md": ("Phase1_EDA", ["eda_distributions.png", "eda_examples.png"]),
}


def render(md_path: str, name: str, figs: list[str]):
    text = (ROOT / md_path).read_text()
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    figs_html = "".join(f'<div class="fig"><img src="{(FIG / f).as_uri()}"><div class="cap">{f}</div></div>' for f in figs if (FIG / f).exists())
    page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}{('<h2>Figures</h2>' + figs_html) if figs_html else ''}</body></html>"
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / f"{name}.html"; h.write_text(page)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={td}/prof", "--no-pdf-header-footer", f"--print-to-pdf={OUT / (name + '.pdf')}", h.as_uri()],
                       check=True, capture_output=True, timeout=120)
    print("wrote", OUT / f"{name}.pdf")


def main():
    for md, (name, figs) in REPORTS.items():
        if (ROOT / md).exists():
            render(md, name, figs)


if __name__ == "__main__":
    main()
