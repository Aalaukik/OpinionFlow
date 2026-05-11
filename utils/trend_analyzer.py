"""
trend_analyzer.py
-----------------
Extracts top 10 keywords / hashtags from the dataset and other trend signals.
"""

import re
import string
from collections import Counter
import pandas as pd


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "this", "that", "these",
    "those", "it", "its", "he", "she", "they", "we", "i", "you", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "when", "where", "how", "why", "if", "as",
    "about", "just", "more", "also", "than", "then", "there", "here",
    "like", "get", "got", "go", "make", "know", "think", "see", "say",
    "said", "new", "one", "two", "all", "some", "any", "each", "every",
    "rt", "amp", "via", "re", "s", "t", "ll", "ve", "d", "m",
}


class TrendAnalyzer:
    """Identifies top keywords, hashtags, and trend signals from a post DataFrame."""

    def analyze(self, df: pd.DataFrame, topic: str) -> dict:
        topic_tokens = set(topic.lower().replace("#", "").split())

        hashtags = self._extract_hashtags(df["text"])
        keywords = self._extract_keywords(df["text"], topic_tokens)
       
        combined = {}
        for tag, cnt in hashtags[:8]:
            combined[tag] = combined.get(tag, 0) + cnt * 2   
        for word, cnt in keywords[:15]:
            if word not in combined:
                combined[word] = cnt

        top_10 = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "top_keywords": top_10,
            "hashtag_count": len(hashtags),
            "unique_keywords": len(keywords),
        }

    def _extract_hashtags(self, texts: pd.Series) -> list:
        counter: Counter = Counter()
        for text in texts.dropna():
            for tag in re.findall(r"#(\w+)", text.lower()):
                if len(tag) > 1:
                    counter[f"#{tag}"] += 1
        return counter.most_common(20)

    def _extract_keywords(self, texts: pd.Series, exclude: set) -> list:
        counter: Counter = Counter()
        punct = str.maketrans("", "", string.punctuation.replace("#", ""))

        for text in texts.dropna():
            text = text.lower().translate(punct)
            for token in text.split():
                token = token.strip("'\"")
                if (
                    token
                    and len(token) > 2
                    and token not in STOPWORDS
                    and token not in exclude
                    and not token.isdigit()
                ):
                    counter[token] += 1

        return counter.most_common(30)
