"""Two-page findings summary (results/SunFix_findings_summary.pdf)."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .load import ROOT

INK, MUTED, BLUE, LIGHT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#2a78d6"), colors.HexColor("#eef4fc")
base = ParagraphStyle("b", fontName="Helvetica", fontSize=8.2, leading=10.4, textColor=INK)
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=15, leading=18)
sub = ParagraphStyle("sub", parent=base, textColor=MUTED, fontSize=8)
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=5, spaceAfter=2, textColor=BLUE)
bul = ParagraphStyle("bul", parent=base, leftIndent=9, bulletIndent=0, spaceAfter=1.4)
small = ParagraphStyle("sm", parent=base, fontSize=7, leading=8.6, textColor=MUTED)
cell = ParagraphStyle("cell", parent=base, fontSize=7.6, leading=9)


def B(t):
    return Paragraph(t, bul, bulletText="•")


def build(path=ROOT / "results" / "SunFix_findings_summary.pdf"):
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title="SunFix: predicting solar battery failure from telemetry", author="SunFix research code")
    S = [Paragraph("Predicting solar battery failure from telemetry: findings so far", h1),
         Paragraph("SunFix research summary, 2 November 2026. Data: 1,027 Bboxx lead-acid batteries (Aitio &amp; Howey 2021, CC BY-NC 4.0, research use only). "
                   "Method: BattGP Gaussian-process resistance model (Schaeffer et al. 2024).", sub), Spacer(1, 5)]

    tldr = [Paragraph("<b>TLDR</b>", ParagraphStyle("t", parent=base, textColor=BLUE, fontSize=10, leading=12)),
            B("<b>It works.</b> From voltage, current and temperature alone, we flag failing batteries with balanced accuracy of <b>81% at failure, 79% 4 weeks ahead and 74% 8 weeks ahead</b> "
              "(published: 82 / 76 / 73%). 50% is a coin flip. Intervals are about ±2.5 points."),
            B("<b>Most of the value comes from combining two things.</b> Battery resistance alone gives 70% at 4 weeks; usage history alone (age, charge throughput, average temperature and voltage) "
              "gives 71%; together 79%. Resistance adds about 8 points at 4 weeks and 4 points at 8 weeks."),
            B("<b>The false-alarm rate is the open problem.</b> At the default threshold about 19% of healthy batteries are flagged. The test set is 48% failed batteries; in a fleet with 5% failures "
              "only about 1 in 6 alerts would be a real failure."),
            B("<b>Two data traps found and removed.</b> Failed batteries keep sending data for a median of 89 days after their repair date (would leak the answer), and a data-availability feature "
              "inflated the at-failure result from 81% to 87%."),
            B("<b>Not done yet:</b> the minimum-data experiments (how little telemetry still works). The table that answers your research question is still to come.")]
    t = Table([[tldr]], colWidths=[17.8 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.6, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S += [t, Paragraph("Method in brief", h2),
          B("<b>Data and cleaning.</b> 619.5M rows (one reading a minute). Failed batteries are cut at the repair date; impossible voltages (under 9 V, over 16 V) are dropped; "
            "charging sessions are selected with Aitio's rules (over 100 min, up to 14 V), giving about 320 sessions per battery."),
            B("<b>Battery state.</b> Charge level is tracked by counting current in and out, anchored to a lab open-circuit-voltage curve measured on a new battery. "
              "Resistance at each point is (voltage − resting voltage) / current."),
            B("<b>Resistance model.</b> BattGP's recursive Gaussian process separates slow ageing from the effect of current, charge level and temperature, "
              "and reports resistance at one common operating point. Only data up to the decision time is used (no look-ahead). Hyperparameters are re-fitted inside each training fold."),
            B("<b>Prediction.</b> Features at the decision time: resistance level, 2/4/8-week slopes, level relative to the battery's own early life, plus six usage factors. "
              "A Gaussian-process classifier (as in the paper) with logistic-regression and gradient-boosting checks. Five-fold cross-validation, always split by battery. "
              "Decision times are 0, 14, 28, 42 and 56 days before the end of each series (failed: repair date; alive: last telemetry)."),
            B("<b>Deviations from the papers.</b> Added a per-session offset term (otherwise the model chases every session); no prior on process noise; assumed generic gassing constants; "
              "5-minute grid; hyperparameters from 8 batteries per fold. Full list in docs/phase2_report.md.")]

    rows = [["Days before end", "0", "14", "28", "42", "56"],
            ["This work: resistance + usage", "81.3", "79.1", "78.6", "76.2", "74.2"],
            ["   95% interval", "78.8–83.6", "76.7–81.5", "76.1–80.9", "73.6–78.6", "71.4–76.8"],
            ["Paper: resistance + usage", "82", "80", "76", "76", "73"],
            ["This work: usage only", "71.6", "70.5", "70.7", "70.5", "70.5"],
            ["This work: resistance only", "73.8", "70.4", "69.9", "67.6", "64.3"],
            ["Paper: resistance only", "76", "73", "67", "64", "63"],
            ["Failures caught (%)", "78.8", "77.4", "76.4", "74.1", "71.9"],
            ["Healthy flagged per 1,000", "162", "192", "192", "216", "235"]]
    tb = Table(rows, colWidths=[5.6 * cm] + [2.44 * cm] * 5)
    tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.8), ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 7.8),
                            ("TEXTCOLOR", (0, 2), (-1, 2), MUTED), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
                            ("LINEBELOW", (0, -1), (-1, -1), 0.6, INK), ("BACKGROUND", (0, 1), (-1, 1), LIGHT), ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]))
    S += [KeepTogether([Paragraph("Results: balanced accuracy (%) by warning time", h2), tb,
                        Paragraph("Balanced accuracy = average of the share of failures caught and the share of healthy batteries cleared. Out-of-fold predictions for all 1,027 batteries. "
                                  "Paper values are from the supplement (Table 1). 'Resistance only' and 'usage only' use the Gaussian-process classifier.", small), Spacer(1, 3),
                        Image(str(ROOT / "results" / "figures" / "phase3_summary.png"), width=16.4 * cm, height=16.4 * cm * 2.9 / 7.4)])]

    S += [Paragraph("What this means and what to be careful about", h2),
          B("<b>The published result reproduces</b> with an independent implementation, with a modest advantage at 4 and 8 weeks that is within the uncertainty. It is not evidence of a better method."),
          B("<b>Honest reading of the 70% threshold.</b> Usage history alone already reaches 70–72% at every horizon (the paper found 66–71%). A 70% bar is therefore not evidence that the "
            "resistance estimate is needed; the useful comparison is the 8-point gain it adds at 4 weeks."),
          B("<b>Age alone gives 62%</b>, probably because failed batteries have somewhat shorter lifetimes (median 500 vs 528 days) even though the paper matched the groups; a weak age signal remains."),
          B("<b>Labels are customer repairs,</b> not measured end of life, so some 'healthy' batteries are failing unrecorded and some repairs are not ageing."),
          B("<b>Single-battery resistance curves are noisy</b> (spikes in both groups); the clear knee about 50 days before repair shows in group medians. The model hyperparameters also varied between folds "
            "(process noise 0.6–6.4), so a larger fitting set is a cheap improvement."),
          B("<b>Limits:</b> lead-acid, one manufacturer, 400+ day batteries only, alive batteries are censored, and the data licence is non-commercial. These results say nothing yet about LFP packs."),
          Paragraph("Next steps", h2),
          B("<b>Minimum-data experiments</b> (the core of your question): thin the telemetry to 1/5/15/60-minute sampling and daily summaries, drop temperature, shorten history to 30/90/180 days, "
            "round voltage and current, drop 10–50% of rows. Each run reports 4- and 8-week accuracy, false alarms, and a usage-only baseline. Aitio's own supplement already suggests 2–10-minute "
            "sampling and 50 charging sessions cost under 3 points, but that thinned already-selected sessions, not raw telemetry."),
          B("<b>Operating point:</b> choose a threshold for an acceptable false-alarm rate, and report precision at realistic failure rates."),
          B("<b>LFP validation:</b> repeat on the BattGP LFP dataset (Zenodo 13715694), then on operator data."),
          Paragraph("Code, figures and full reports are in the SunFix_coding repository (docs/, results/). Reproduce: python -m sunfix.phase3_features; python -m sunfix.classify.", small)]
    doc.build(S)
    return path


if __name__ == "__main__":
    print(build())
