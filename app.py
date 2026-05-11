import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_extractor import DataExtractor
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.trend_analyzer import TrendAnalyzer
from utils.pdf_generator import PDFGenerator

st.set_page_config(
    page_title="OpinionFlow – Social Media Insights",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.8rem; font-weight: 700; margin: 0; }
    .main-header p  { color: #a0aec0; font-size: 1rem; margin: 0.4rem 0 0; }
    .metric-card {
        background: #1e2a3a;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.8rem; color: #718096; margin-top: 0.2rem; }
    .positive { color: #48bb78; }
    .negative { color: #fc8181; }
    .neutral  { color: #63b3ed; }
    .section-header {
        font-size: 1.2rem; font-weight: 600;
        color: #e2e8f0; border-left: 4px solid #e94560;
        padding-left: 0.8rem; margin: 1.5rem 0 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c53030);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }
    .sidebar .stSelectbox label, .sidebar .stTextInput label { color: #a0aec0; }
    .post-card {
        background: #1e2a3a; border: 1px solid #2d3748;
        border-radius: 8px; padding: 1rem; margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌊 OpinionFlow</h1>
    <p>Social Media Sentiment & Reporting Engine — Real-time public discourse analytics</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Query Configuration")
    st.markdown("---")

    topic = st.text_input("🔍 Keyword / Hashtag", placeholder="e.g. #AI, climate change")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From", value=datetime.now() - timedelta(days=7))
    with col2:
        date_to = st.date_input("To", value=datetime.now())

    platforms = st.multiselect(
        "📡 Platforms",
        ["Reddit", "Twitter/X (Demo)", "HackerNews"],
        default=["Reddit", "Twitter/X (Demo)"],
    )
    record_limit = st.slider("📊 Max Records", 100, 1000, 500, 50)
    analysis_model = st.selectbox(
        "🤖 Sentiment Model",
        ["VADER (Fast, No Training)", "Fine-tuned DistilBERT (Upload .pt)"],
    )

    model_file = None
    if analysis_model == "Fine-tuned DistilBERT (Upload .pt)":
        model_file = st.file_uploader("Upload model (.pt)", type=["pt", "bin"])

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("**How to get API keys:**")
    with st.expander("Reddit (PRAW)"):
        st.markdown("1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)\n2. Create a 'script' app\n3. Copy `client_id`, `client_secret`, `user_agent`")
    with st.expander("Twitter / X"):
        st.markdown("1. Go to [developer.twitter.com](https://developer.twitter.com)\n2. Create a project & app\n3. Copy **Bearer Token**")

    st.markdown("---")
    st.caption("OpinionFlow v1.0 · Built with Streamlit")

if "results" not in st.session_state:
    st.session_state.results = None
if "report_data" not in st.session_state:
    st.session_state.report_data = None

if run_btn:
    if not topic.strip():
        st.error("⚠️ Please enter a keyword or hashtag to analyze.")
    elif not platforms:
        st.error("⚠️ Please select at least one platform.")
    else:
        progress = st.progress(0, text="Initializing...")
        status  = st.empty()

        try:
            status.info("📡 Step 1/4 — Fetching posts from selected platforms...")
            extractor = DataExtractor()
            raw_posts = extractor.fetch(
                topic=topic,
                platforms=platforms,
                date_from=date_from,
                date_to=date_to,
                limit=record_limit,
            )
            progress.progress(25, text="Data fetched ✓")

            if raw_posts.empty:
                st.warning("No posts found for this query. Try a broader keyword or different date range.")
                progress.empty(); status.empty()
                st.stop()

            status.info("🧠 Step 2/4 — Running sentiment analysis...")
            analyzer = SentimentAnalyzer(model_type=analysis_model, model_file=model_file)
            df = analyzer.analyze(raw_posts)
            progress.progress(50, text="Sentiment analysis complete ✓")

            status.info("📈 Step 3/4 — Mapping trends & keywords...")
            trend_analyzer = TrendAnalyzer()
            trend_data = trend_analyzer.analyze(df, topic)
            progress.progress(75, text="Trend mapping complete ✓")

            status.info("📊 Step 4/4 — Generating dashboard...")
            report_data = {
                "topic": topic,
                "platforms": platforms,
                "date_from": str(date_from),
                "date_to": str(date_to),
                "df": df,
                "trend_data": trend_data,
                "model": analysis_model,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.results = True
            st.session_state.report_data = report_data
            progress.progress(100, text="Done! ✅")
            time.sleep(0.5)
            progress.empty()
            status.empty()
            st.success(f"✅ Analysis complete! Processed **{len(df):,}** posts.")

        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("💡 Tip: Check your API credentials in the `utils/data_extractor.py` file, or the README for setup instructions.")

if st.session_state.results and st.session_state.report_data:
    rd  = st.session_state.report_data
    df  = rd["df"]
    td  = rd["trend_data"]

    total      = len(df)
    pos_pct    = round(len(df[df.sentiment == "Positive"]) / total * 100, 1)
    neg_pct    = round(len(df[df.sentiment == "Negative"]) / total * 100, 1)
    neu_pct    = round(100 - pos_pct - neg_pct, 1)
    avg_score  = round(df["sentiment_score"].mean(), 3)
    engagement = int(df["engagement"].sum()) if "engagement" in df.columns else "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, f"{total:,}", "Total Posts", "#63b3ed"),
        (c2, f"{pos_pct}%", "Positive", "#48bb78"),
        (c3, f"{neg_pct}%", "Negative", "#fc8181"),
        (c4, f"{neu_pct}%", "Neutral", "#f6ad55"),
        (c5, f"{avg_score:+.3f}", "Sentiment Score", "#e94560"),
    ]
    for col, val, lbl, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{color}">{val}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="section-header">📊 Sentiment Overview</div>', unsafe_allow_html=True)
    col_pie, col_line = st.columns([1, 2])

    with col_pie:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        color_map = {"Positive": "#48bb78", "Negative": "#fc8181", "Neutral": "#f6ad55"}
        fig_pie = px.pie(
            counts, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=color_map,
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_line:
        if "date" in df.columns and not df["date"].isna().all():
            timeline = (
                df.groupby(["date", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            fig_line = px.area(
                timeline, x="date", y="count", color="sentiment",
                color_discrete_map=color_map,
                labels={"count": "Posts", "date": "Date"},
            )
            fig_line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="#2d3748"),
                yaxis=dict(gridcolor="#2d3748"),
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Timeline chart requires date information from the API.")

    st.markdown('<div class="section-header">🔍 Trend & Keyword Analysis</div>', unsafe_allow_html=True)
    col_kw, col_plat = st.columns([2, 1])

    with col_kw:
        if td.get("top_keywords"):
            kw_df = pd.DataFrame(td["top_keywords"], columns=["Keyword", "Frequency"])
            fig_bar = px.bar(
                kw_df.sort_values("Frequency"), x="Frequency", y="Keyword",
                orientation="h", color="Frequency",
                color_continuous_scale=["#2d3748", "#e94560"],
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="#2d3748"),
                yaxis=dict(gridcolor="#2d3748"),
                coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_plat:
        if "platform" in df.columns:
            plat_df = df["platform"].value_counts().reset_index()
            plat_df.columns = ["Platform", "Count"]
            fig_plat = px.bar(
                plat_df, x="Platform", y="Count",
                color="Platform",
                color_discrete_sequence=["#e94560", "#63b3ed", "#f6ad55"],
            )
            fig_plat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="#2d3748"),
                yaxis=dict(gridcolor="#2d3748"),
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_plat, use_container_width=True)

    st.markdown('<div class="section-header">📉 Score Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        df, x="sentiment_score", color="sentiment",
        color_discrete_map=color_map, nbins=40,
        labels={"sentiment_score": "Compound Score (-1 → +1)"},
        barmode="overlay", opacity=0.75,
    )
    fig_hist.add_vline(x=avg_score, line_dash="dash", line_color="#e94560",
                       annotation_text=f"  Avg: {avg_score:+.3f}", annotation_font_color="#e94560")
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748"),
        margin=dict(t=20, b=20, l=20, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-header">📝 AI Executive Summary</div>', unsafe_allow_html=True)
    dominant = "Positive" if pos_pct > neg_pct else ("Negative" if neg_pct > pos_pct else "Neutral")
    dominant_color = color_map.get(dominant, "#e2e8f0")
    top_kws = ", ".join([k for k, _ in td.get("top_keywords", [])[:5]]) if td.get("top_keywords") else "N/A"

    summary_html = f"""
    <div style="background:#1e2a3a;border:1px solid #2d3748;border-radius:12px;padding:1.5rem;">
        <p style="color:#e2e8f0;line-height:1.8;font-size:0.95rem;">
        This report analyzes public discourse around <strong style="color:#e94560">"{rd['topic']}"</strong>
        across <strong>{', '.join(rd['platforms'])}</strong> from <strong>{rd['date_from']}</strong>
        to <strong>{rd['date_to']}</strong>, covering a total of <strong>{total:,} posts</strong>.<br><br>
        The overall public sentiment is predominantly
        <strong style="color:{dominant_color}">{dominant}</strong>,
        with <strong style="color:#48bb78">{pos_pct}%</strong> positive,
        <strong style="color:#fc8181">{neg_pct}%</strong> negative, and
        <strong style="color:#f6ad55">{neu_pct}%</strong> neutral posts.
        The aggregate sentiment score is <strong style="color:#e94560">{avg_score:+.3f}</strong>
        on a scale of -1.0 (most negative) to +1.0 (most positive).<br><br>
        The most frequently associated keywords and themes include:
        <strong style="color:#63b3ed">{top_kws}</strong>.
        These terms reflect the primary narrative drivers within the dataset.
        </p>
        <p style="color:#718096;font-size:0.78rem;margin-top:0.5rem;">
        Generated at {rd['generated_at']} · Model: {rd['model']}
        </p>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏆 Top Performing Posts</div>', unsafe_allow_html=True)
    display_cols = ["text", "platform", "sentiment", "sentiment_score", "engagement"]
    display_cols = [c for c in display_cols if c in df.columns]
    top_posts = df.nlargest(20, "engagement") if "engagement" in df.columns else df.head(20)
    st.dataframe(
        top_posts[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=340,
    )

    st.markdown('<div class="section-header">📥 Export Report</div>', unsafe_allow_html=True)
    col_dl, col_csv = st.columns(2)

    with col_dl:
        if st.button("📄 Generate & Download PDF Report"):
            with st.spinner("Building PDF..."):
                try:
                    gen = PDFGenerator()
                    pdf_bytes = gen.generate(rd)
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"opinionflow_{rd['topic'].replace(' ','_')}_{rd['generated_at'][:10]}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    with col_csv:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Download Raw CSV",
            data=csv_buf.getvalue(),
            file_name=f"opinionflow_raw_{rd['topic'].replace(' ','_')}.csv",
            mime="text/csv",
        )

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem;">🌊</div>
        <h3 style="color:#e2e8f0;margin-top:1rem;">Ready to Analyze Public Opinion</h3>
        <p style="color:#718096;max-width:480px;margin:0.8rem auto 0;">
            Enter a keyword or hashtag in the sidebar, configure your date range and platforms,
            then hit <strong>Run Analysis</strong> to generate your sentiment report.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        ("📡", "Multi-Platform", "Pulls data from Reddit, Twitter/X, and HackerNews simultaneously"),
        ("🧠", "Sentiment AI", "VADER for instant analysis or fine-tuned DistilBERT for higher accuracy"),
        ("📄", "PDF Export", "One-click professional PDF report with charts and executive summary"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:1.5rem">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin:.5rem 0 .3rem">{title}</div>
                <div style="font-size:.8rem;color:#718096">{desc}</div>
            </div>""", unsafe_allow_html=True)
