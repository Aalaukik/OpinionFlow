# 🌊 OpinionFlow — Social Media Sentiment & Reporting Engine

> **Turn public opinion into actionable intelligence.** OpinionFlow is an end-to-end sentiment analytics dashboard that extracts real-time social media data, analyzes it with AI, and delivers a downloadable executive report — all in under 3 minutes.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?style=flat&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🎯 The Problem

Brands, researchers, analysts, and students need to understand how people feel about a topic online — but doing this manually means scrolling through thousands of posts across multiple platforms, copy-pasting into spreadsheets, and still ending up with no real insight. Existing tools are either too expensive, too complex, or locked behind enterprise paywalls.

## 💡 The Solution

OpinionFlow is a **free, open-source, one-click analytics engine** that does the heavy lifting for you. Enter any keyword or hashtag, select your platforms, and in seconds you get:

- A fully visualized sentiment dashboard
- AI-powered classification of every post (Positive / Neutral / Negative)
- Top trending keywords and hashtags
- A professional PDF report you can share with anyone

No data science knowledge required. No expensive subscriptions. Just answers.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📡 **Multi-Platform Extraction** | Pulls posts from Reddit, Twitter/X, and HackerNews simultaneously |
| 🧠 **Dual Sentiment Engine** | VADER for instant results, or fine-tuned DistilBERT for higher accuracy |
| 📈 **Trend Mapping** | Identifies the top 10 most frequent keywords and hashtags in the dataset |
| 📊 **Interactive Dashboard** | Sentiment pie charts, timeline area charts, score histograms — all interactive |
| 📝 **AI Executive Summary** | Auto-generated plain-English summary of findings |
| 📄 **One-Click PDF Export** | Professional multi-page PDF with charts, tables, and summary — ready to share |
| 🔒 **Privacy First** | Zero PII stored; all data lives only in your current session |
| 🚀 **Demo Mode** | Works without any API keys using realistic synthetic data |

---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────────────────────────┐
│  🌊 OpinionFlow                                              │
│  Social Media Sentiment & Reporting Engine                  │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│  2,847   │  61.2%   │  24.3%   │  14.5%   │   +0.312       │
│  Posts   │ Positive │ Negative │  Neutral │ Sentiment Score│
├──────────┴──────────┴──────────┴──────────┴────────────────┤
│  [Sentiment Donut Chart]    [Sentiment Timeline Area Chart] │
├─────────────────────────────────────────────────────────────┤
│  [Top Keywords Bar Chart]   [Platform Distribution Chart]  │
├─────────────────────────────────────────────────────────────┤
│  📝 Executive Summary                                       │
│  Analysis of "artificial intelligence" across Reddit and   │
│  HackerNews shows predominantly Positive sentiment...       │
├─────────────────────────────────────────────────────────────┤
│  [Download PDF]   [Download CSV]                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
opinionflow/
│
├── app.py                          ← Streamlit UI & dashboard
│
├── utils/
│   ├── data_extractor.py           ← Reddit / Twitter / HackerNews fetchers
│   ├── sentiment_analyzer.py       ← VADER + DistilBERT scoring engine
│   ├── trend_analyzer.py           ← Keyword & hashtag extraction
│   └── pdf_generator.py            ← ReportLab PDF report builder
│
├── notebooks/
│   └── OpinionFlow_DistilBERT_Training.ipynb   ← Google Colab training
│
├── models/                         ← Drop your fine-tuned .pt here
├── .streamlit/                     ← Theme + secrets config
├── requirements.txt
└── README.md
```

**Data Flow:**
```
User Input (keyword + date range + platforms)
        ↓
DataExtractor  →  raw posts DataFrame
        ↓
SentimentAnalyzer  →  scored + labelled DataFrame
        ↓
TrendAnalyzer  →  top keywords + trend signals
        ↓
Streamlit Dashboard  →  interactive charts + executive summary
        ↓
PDFGenerator  →  downloadable report
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Web UI, interactive charts |
| Data | PRAW, Tweepy, Requests | Social media APIs |
| NLP (fast) | VADER Sentiment | Rule-based scoring, no training |
| NLP (accurate) | DistilBERT + HuggingFace | Fine-tuned transformer model |
| Training | PyTorch + Google Colab | GPU training on Sentiment140 |
| Visualization | Plotly, Matplotlib | Interactive + static charts |
| PDF | ReportLab | Professional report generation |
| Deployment | Streamlit Cloud | Free hosting |

---

## 🚀 Quick Start

### Option A — No API Keys (works immediately)
```bash
git clone https://github.com/YOUR_USERNAME/opinionflow.git
cd opinionflow
pip install -r requirements.txt
streamlit run app.py
```
Select **HackerNews** in the sidebar — it's a public API requiring zero setup.

### Option B — With Reddit & Twitter Keys
```bash
cp .env.example .env
# Edit .env with your credentials
streamlit run app.py
```

---

## 🔑 API Keys Setup

### Reddit (Free)
1. Go to [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
2. Create a **script** type app
3. Copy `client_id` and `client_secret` into `.env`

### Twitter/X (Free tier)
1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Create a project and app
3. Copy the **Bearer Token** into `.env`

### HackerNews
No key needed — Algolia's public API is free with no limits.

---

## 🤖 Training the DistilBERT Model

The project ships with VADER by default. For higher accuracy, fine-tune DistilBERT on the Sentiment140 dataset using the included Colab notebook.

**Dataset:** Stanford Sentiment140 — 1.6 million labelled tweets (auto-downloaded)

**Steps:**
1. Open `notebooks/OpinionFlow_DistilBERT_Training.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (free tier)
3. Run all cells (~40 minutes)
4. Download `opinionflow_sentiment.pt`
5. Upload it in the Streamlit sidebar

**Expected results:** ~87–90% accuracy on 3-class sentiment classification

---

## 📊 Performance Targets

| Metric | Target | Notes |
|---|---|---|
| Query to dashboard | < 60 seconds | For 500 records |
| Keyword to PDF download | < 3 minutes | Full end-to-end |
| Extraction success rate | > 95% | With valid API keys |
| PDF export fidelity | 100% | Charts match dashboard |

---

## 🔒 Privacy & Data Policy

- No user data or post content is stored to disk or any database
- All extracted data lives only in the current Streamlit session memory
- Sessions are isolated — no data persists between runs
- No personally identifiable information (PII) is collected or logged

---

## 🤝 Contributing

Contributions are welcome! Some ideas for extensions:

- Add Instagram or LinkedIn data sources
- Multilingual sentiment support
- Scheduled recurring reports via email
- Topic clustering with BERTopic
- Real-time streaming with websockets

```bash
# Fork the repo, create your branch
git checkout -b feature/your-feature-name

# Make changes, then
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built with ❤️ as a personal project to make social media intelligence accessible to everyone.

If you found this useful, consider giving it a ⭐ on GitHub!

---

*OpinionFlow v1.0 · Powered by Streamlit, HuggingFace & ReportLab*
