"""
pdf_generator.py
----------------
Generates a professional multi-page PDF report using ReportLab.
Embeds matplotlib charts as PNG buffers — no external files needed.
"""

import io
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.colors import HexColor

C_BG       = HexColor("#1a1a2e")
C_PANEL    = HexColor("#1e2a3a")
C_RED      = HexColor("#e94560")
C_GREEN    = HexColor("#48bb78")
C_BLUE     = HexColor("#63b3ed")
C_ORANGE   = HexColor("#f6ad55")
C_DARK     = HexColor("#0f3460")
C_TEXT     = HexColor("#e2e8f0")
C_SUBTEXT  = HexColor("#718096")
C_WHITE    = colors.white

def _fig_to_image(fig, width_mm=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_mm * mm)


def _sentiment_colors(sentiment):
    return {
        "Positive": "#48bb78",
        "Negative": "#fc8181",
        "Neutral":  "#f6ad55",
    }.get(sentiment, "#a0aec0")

def _build_pie_chart(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values.tolist()
    clrs   = [_sentiment_colors(l) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1e2a3a")
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=clrs,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(width=0.6, edgecolor="#1a1a2e", linewidth=2),
    )
    for t in texts:
        t.set_color("#e2e8f0"); t.set_fontsize(9)
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(8); at.set_fontweight("bold")
    ax.set_facecolor("#1e2a3a")
    fig.patch.set_facecolor("#1e2a3a")
    return fig


def _build_timeline_chart(df: pd.DataFrame):
    if "date" not in df.columns or df["date"].isna().all():
        return None
    timeline = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    palette  = {"Positive": "#48bb78", "Negative": "#fc8181", "Neutral": "#f6ad55"}

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#1e2a3a")
    for sent in ["Positive", "Neutral", "Negative"]:
        if sent in timeline.columns:
            ax.fill_between(
                timeline.index, timeline[sent], alpha=0.6,
                color=palette[sent], label=sent,
            )
            ax.plot(timeline.index, timeline[sent], color=palette[sent], linewidth=1.5)

    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#a0aec0", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.set_ylabel("Post Count", color="#a0aec0", fontsize=8)
    ax.legend(facecolor="#1e2a3a", labelcolor="#e2e8f0", fontsize=8, loc="upper right")
    fig.patch.set_facecolor("#1e2a3a")
    plt.xticks(rotation=30)
    return fig


def _build_keyword_chart(top_keywords):
    if not top_keywords:
        return None
    kws   = [k for k, _ in top_keywords[:10]]
    freqs = [f for _, f in top_keywords[:10]]
    clrs  = ["#e94560" if i < 3 else "#63b3ed" for i in range(len(kws))]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#1e2a3a")
    bars = ax.barh(kws[::-1], freqs[::-1], color=clrs[::-1], edgecolor="#1a1a2e", linewidth=0.5)
    for bar, val in zip(bars, freqs[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color="#a0aec0", fontsize=8)

    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#a0aec0", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.set_xlabel("Frequency", color="#a0aec0", fontsize=8)
    fig.patch.set_facecolor("#1e2a3a")
    return fig


def _build_score_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#1e2a3a")
    for sent, color in [("Positive", "#48bb78"), ("Neutral", "#f6ad55"), ("Negative", "#fc8181")]:
        sub = df[df["sentiment"] == sent]["sentiment_score"]
        if not sub.empty:
            ax.hist(sub, bins=25, alpha=0.7, color=color, label=sent, edgecolor="#1a1a2e")
    avg = df["sentiment_score"].mean()
    ax.axvline(avg, color="#e94560", linewidth=2, linestyle="--", label=f"Avg {avg:+.3f}")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#a0aec0", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.set_xlabel("Compound Score", color="#a0aec0", fontsize=8)
    ax.legend(facecolor="#1e2a3a", labelcolor="#e2e8f0", fontsize=8)
    fig.patch.set_facecolor("#1e2a3a")
    return fig

class PDFGenerator:

    def generate(self, report_data: dict) -> bytes:
        buf     = io.BytesIO()
        df      = report_data["df"]
        td      = report_data["trend_data"]
        topic   = report_data["topic"]
        gen_at  = report_data["generated_at"]

        total   = len(df)
        pos_pct = round(len(df[df.sentiment == "Positive"]) / total * 100, 1)
        neg_pct = round(len(df[df.sentiment == "Negative"]) / total * 100, 1)
        neu_pct = round(100 - pos_pct - neg_pct, 1)
        avg_sc  = round(df["sentiment_score"].mean(), 4)

        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=15*mm, rightMargin=15*mm,
            topMargin=15*mm, bottomMargin=15*mm,
        )
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle("Title", parent=styles["Normal"],
            fontSize=26, textColor=C_RED, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=4)
        sub_style = ParagraphStyle("Sub", parent=styles["Normal"],
            fontSize=10, textColor=C_SUBTEXT, alignment=TA_CENTER, spaceAfter=2)
        section_style = ParagraphStyle("Section", parent=styles["Normal"],
            fontSize=13, textColor=C_RED, fontName="Helvetica-Bold",
            spaceBefore=12, spaceAfter=6)
        body_style = ParagraphStyle("Body", parent=styles["Normal"],
            fontSize=9, textColor=C_TEXT, leading=14, spaceAfter=8)
        caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
            fontSize=7.5, textColor=C_SUBTEXT, alignment=TA_CENTER, spaceAfter=4)

        story.append(Spacer(1, 20*mm))
        story.append(Paragraph("🌊 OpinionFlow", title_style))
        story.append(Paragraph("Social Media Sentiment & Intelligence Report", sub_style))
        story.append(Spacer(1, 6*mm))
        story.append(HRFlowable(width="100%", thickness=1, color=C_RED))
        story.append(Spacer(1, 6*mm))

        story.append(Paragraph(
            f'<font color="#e94560"><b>Topic:</b></font>  {topic}', body_style))
        story.append(Paragraph(
            f'<font color="#e94560"><b>Period:</b></font>  {report_data["date_from"]} to {report_data["date_to"]}', body_style))
        story.append(Paragraph(
            f'<font color="#e94560"><b>Platforms:</b></font>  {", ".join(report_data["platforms"])}', body_style))
        story.append(Paragraph(
            f'<font color="#e94560"><b>Model:</b></font>  {report_data["model"]}', body_style))
        story.append(Paragraph(
            f'<font color="#e94560"><b>Generated:</b></font>  {gen_at}', body_style))

        story.append(Spacer(1, 8*mm))

        kpi_data = [
            ["Total Posts", "Positive", "Negative", "Neutral", "Sentiment Score"],
            [f"{total:,}", f"{pos_pct}%", f"{neg_pct}%", f"{neu_pct}%", f"{avg_sc:+.4f}"],
        ]
        kpi_table = Table(kpi_data, colWidths=[35*mm]*5)
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), C_DARK),
            ("BACKGROUND",  (0,1), (-1,1), C_PANEL),
            ("TEXTCOLOR",   (0,0), (-1,0), C_RED),
            ("TEXTCOLOR",   (0,1), (0,1),  C_BLUE),
            ("TEXTCOLOR",   (1,1), (1,1),  HexColor("#48bb78")),
            ("TEXTCOLOR",   (2,1), (2,1),  HexColor("#fc8181")),
            ("TEXTCOLOR",   (3,1), (3,1),  C_ORANGE),
            ("TEXTCOLOR",   (4,1), (4,1),  C_RED),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME",    (0,1), (-1,1), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,0), 9),
            ("FONTSIZE",    (0,1), (-1,1), 14),
            ("ALIGN",       (0,0), (-1,-1), "CENTER"),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_DARK, C_PANEL]),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("GRID",        (0,0), (-1,-1), 0.5, HexColor("#2d3748")),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 8*mm))

        dominant = "Positive" if pos_pct > neg_pct else ("Negative" if neg_pct > pos_pct else "Neutral")
        top_kws  = ", ".join([k for k, _ in td.get("top_keywords", [])[:5]]) or "N/A"
        summary  = (
            f"This report analyzed {total:,} posts about '{topic}' across "
            f"{', '.join(report_data['platforms'])} between "
            f"{report_data['date_from']} and {report_data['date_to']}. "
            f"Public sentiment is predominantly {dominant} ({pos_pct}% positive, "
            f"{neg_pct}% negative, {neu_pct}% neutral), with an aggregate sentiment "
            f"score of {avg_sc:+.4f} on a scale from -1.0 to +1.0. "
            f"The most prominent keywords in this discourse are: {top_kws}. "
            f"These topics represent the primary narrative drivers and areas of public focus."
        )
        story.append(Paragraph("Executive Summary", section_style))
        story.append(Paragraph(summary, body_style))

        story.append(PageBreak())
        story.append(Paragraph("Sentiment Distribution", section_style))

        pie_fig  = _build_pie_chart(df)
        story.append(_fig_to_image(pie_fig, width_mm=100))
        story.append(Paragraph("Figure 1 — Sentiment distribution of all extracted posts.", caption_style))

        tl_fig = _build_timeline_chart(df)
        if tl_fig:
            story.append(Spacer(1, 6*mm))
            story.append(Paragraph("Sentiment Over Time", section_style))
            story.append(_fig_to_image(tl_fig, width_mm=170))
            story.append(Paragraph("Figure 2 — Temporal trend of sentiment across the selected date range.", caption_style))

        story.append(PageBreak())
        story.append(Paragraph("Top Keywords & Hashtags", section_style))
        kw_fig = _build_keyword_chart(td.get("top_keywords", []))
        if kw_fig:
            story.append(_fig_to_image(kw_fig, width_mm=170))
            story.append(Paragraph("Figure 3 — Most frequently co-occurring terms in the dataset.", caption_style))

        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("Sentiment Score Distribution", section_style))
        hist_fig = _build_score_histogram(df)
        story.append(_fig_to_image(hist_fig, width_mm=170))
        story.append(Paragraph("Figure 4 — Distribution of compound sentiment scores (-1 = most negative, +1 = most positive).", caption_style))

        story.append(PageBreak())
        story.append(Paragraph("Top 20 Posts by Engagement", section_style))

        top_posts = df.nlargest(20, "engagement") if "engagement" in df.columns else df.head(20)
        table_data = [["#", "Platform", "Sentiment", "Score", "Engagement", "Text"]]
        for idx, row in enumerate(top_posts.itertuples(), 1):
            text = str(getattr(row, "text", ""))[:80] + ("…" if len(str(getattr(row, "text", ""))) > 80 else "")
            table_data.append([
                str(idx),
                str(getattr(row, "platform", "N/A")),
                str(getattr(row, "sentiment", "")),
                f"{getattr(row, 'sentiment_score', 0):+.3f}",
                f"{int(getattr(row, 'engagement', 0)):,}",
                text,
            ])

        post_table = Table(table_data, colWidths=[8*mm, 22*mm, 22*mm, 18*mm, 22*mm, 78*mm])
        sent_colors = {
            "Positive": HexColor("#1a4731"),
            "Negative": HexColor("#4a1a1a"),
            "Neutral":  HexColor("#2a3a4a"),
        }
        ts = [
            ("BACKGROUND",    (0,0), (-1,0), C_DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), C_RED),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("ALIGN",         (5,0), (5,-1), "LEFT"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("GRID",          (0,0), (-1,-1), 0.5, HexColor("#2d3748")),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#1e2a3a"), HexColor("#172030")]),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]
        for i, row in enumerate(top_posts.itertuples(), 1):
            sent = str(getattr(row, "sentiment", ""))
            bg   = sent_colors.get(sent, HexColor("#1e2a3a"))
            ts.append(("BACKGROUND", (2, i), (2, i), bg))
            tc = {"Positive": HexColor("#48bb78"), "Negative": HexColor("#fc8181")}.get(sent, C_ORANGE)
            ts.append(("TEXTCOLOR",  (2, i), (2, i), tc))

        post_table.setStyle(TableStyle(ts))
        story.append(post_table)

        story.append(Spacer(1, 10*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_SUBTEXT))
        story.append(Paragraph(
            f"Generated by OpinionFlow · {gen_at} · Confidential",
            ParagraphStyle("footer", parent=styles["Normal"],
                fontSize=7, textColor=C_SUBTEXT, alignment=TA_CENTER, spaceBefore=4)
        ))

        doc.build(story)
        buf.seek(0)
        return buf.read()
