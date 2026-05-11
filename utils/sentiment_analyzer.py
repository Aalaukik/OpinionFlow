"""
sentiment_analyzer.py
---------------------
Provides two sentiment analysis modes:
  1. VADER  – rule-based, fast, no training required.
  2. Fine-tuned DistilBERT – upload a .pt model trained in Google Colab.
"""

import re
import pandas as pd
import numpy as np

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SentimentAnalyzer:
    """
    Analyzes sentiment of a DataFrame column 'text'.

    Parameters
    ----------
    model_type : str
        "VADER (Fast, No Training)" or "Fine-tuned DistilBERT (Upload .pt)"
    model_file : file-like object | None
        Uploaded .pt state-dict when using DistilBERT.
    """

    LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def __init__(self, model_type: str = "VADER (Fast, No Training)", model_file=None):
        self.model_type = model_type
        self.model_file = model_file
        self._vader = None
        self._bert_model = None
        self._tokenizer = None

        if "VADER" in model_type or not TORCH_AVAILABLE:
            self._init_vader()
        else:
            self._init_bert()

    def _init_vader(self):
        if not VADER_AVAILABLE:
            raise RuntimeError(
                "vaderSentiment is not installed.\n"
                "Run: pip install vaderSentiment"
            )
        self._vader = SentimentIntensityAnalyzer()

    def _init_bert(self):
        if not TORCH_AVAILABLE:
            print("[SentimentAnalyzer] torch/transformers not available — falling back to VADER.")
            self._init_vader()
            return
        model_name = "distilbert-base-uncased"
        self._tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self._bert_model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        if self.model_file is not None:
            try:
                import io
                state_dict = torch.load(io.BytesIO(self.model_file.read()), map_location="cpu")
                self._bert_model.load_state_dict(state_dict)
                print("[SentimentAnalyzer] Custom DistilBERT weights loaded.")
            except Exception as e:
                print(f"[SentimentAnalyzer] Failed to load custom weights: {e}. Using base model.")
        self._bert_model.eval()

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["clean_text"] = df["text"].apply(self._clean_text)

        if self._vader is not None:
            scores, labels = zip(*df["clean_text"].apply(self._score_vader))
        else:
            scores, labels = self._score_bert_batch(df["clean_text"].tolist())

        df["sentiment_score"] = list(scores)
        df["sentiment"]       = list(labels)
        return df

    def _score_vader(self, text: str):
        vs = self._vader.polarity_scores(text)
        compound = vs["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return compound, label

    def _score_bert_batch(self, texts: list, batch_size: int = 32):
        import torch
        all_scores, all_labels = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = self._tokenizer(
                batch, truncation=True, padding=True,
                max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self._bert_model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).numpy()
            preds  = np.argmax(probs, axis=1)
            scores = (probs[:, 2] - probs[:, 0]).tolist()
            labels = [self.LABEL_MAP[p] for p in preds]
            all_scores.extend(scores)
            all_labels.extend(labels)
        return all_scores, all_labels

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+|www\S+", "", text)       
        text = re.sub(r"@\w+", "", text)                 
        text = re.sub(r"<[^>]+>", "", text)              
        text = re.sub(r"[^\w\s#.,!?'-]", " ", text)       
        text = re.sub(r"\s+", " ", text).strip()
        return text[:512]
