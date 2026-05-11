"""
data_extractor.py
-----------------
Fetches posts from Reddit (PRAW), Twitter/X (Tweepy v2), and HackerNews (Algolia API).

API KEYS — set via environment variables or a .env file:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT
    TWITTER_BEARER_TOKEN
"""

import os
import re
import time
import random
import requests
import pandas as pd
from datetime import datetime, timedelta

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class DataExtractor:
    """Unified multi-platform data extractor."""

    SAMPLE_SENTENCES = [
        "This is absolutely amazing progress in the field!",
        "I really love how this technology is developing.",
        "What a fantastic achievement for everyone involved.",
        "Completely disappointed with the latest updates here.",
        "This is terrible and makes everything worse.",
        "Not sure what to think about these recent changes.",
        "Just another ordinary day of news and updates.",
        "The results are interesting but need more analysis.",
        "People are really divided on this important topic.",
        "Some great points raised in this ongoing discussion.",
        "I cannot believe how bad things have gotten recently.",
        "Truly inspiring work being done by dedicated teams.",
        "Meh, nothing special about any of this honestly.",
        "Huge breakthrough that will change everything we know!",
        "Deeply concerning trends that we should all watch carefully.",
        "Neutral observer here, just sharing what I observed.",
        "Best thing I have seen all week without a doubt.",
        "Worst possible outcome for everyone affected by this.",
        "Moderate take: there are pros and cons to consider.",
        "Very thoughtful analysis of a complex and evolving situation.",
    ]

    def __init__(self):
        self.reddit_id     = os.getenv("REDDIT_CLIENT_ID", "")
        self.reddit_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.reddit_agent  = os.getenv("REDDIT_USER_AGENT", "OpinionFlow/1.0")
        self.twitter_token = os.getenv("TWITTER_BEARER_TOKEN", "")

    # ─── Public entry point ──────────────────────────────────────────────────
    def fetch(
        self,
        topic: str,
        platforms: list,
        date_from,
        date_to,
        limit: int = 500,
    ) -> pd.DataFrame:
        frames = []
        per_platform = max(50, limit // max(len(platforms), 1))

        for plat in platforms:
            try:
                if plat == "Reddit":
                    df = self._fetch_reddit(topic, date_from, date_to, per_platform)
                elif plat == "Twitter/X (Demo)":
                    df = self._fetch_twitter(topic, date_from, date_to, per_platform)
                elif plat == "HackerNews":
                    df = self._fetch_hackernews(topic, date_from, date_to, per_platform)
                else:
                    continue
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"[DataExtractor] {plat} error: {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
        return combined

    def _fetch_reddit(self, topic, date_from, date_to, limit) -> pd.DataFrame:
        if not PRAW_AVAILABLE:
            print("[DataExtractor] praw not installed — using demo data for Reddit.")
            return self._demo_data("Reddit", topic, limit)

        if not self.reddit_id or not self.reddit_secret:
            print("[DataExtractor] Reddit API keys not set — using demo data.")
            return self._demo_data("Reddit", topic, limit)

        try:
            reddit = praw.Reddit(
                client_id=self.reddit_id,
                client_secret=self.reddit_secret,
                user_agent=self.reddit_agent,
            )
            posts = []
            ts_from = int(datetime.combine(date_from, datetime.min.time()).timestamp())
            ts_to   = int(datetime.combine(date_to,   datetime.max.time()).timestamp())

            for sub in reddit.subreddit("all").search(topic, limit=limit, time_filter="month"):
                if not (ts_from <= sub.created_utc <= ts_to):
                    continue
                posts.append({
                    "text":       (sub.title + " " + (sub.selftext or ""))[:500],
                    "platform":   "Reddit",
                    "engagement": sub.score + sub.num_comments,
                    "date":       pd.to_datetime(sub.created_utc, unit="s").date(),
                    "url":        f"https://reddit.com{sub.permalink}",
                })
                if len(posts) >= limit:
                    break

            if not posts:
                return self._demo_data("Reddit", topic, limit)

            return pd.DataFrame(posts)
        except Exception as e:
            print(f"[DataExtractor] Reddit API error: {e} — falling back to demo.")
            return self._demo_data("Reddit", topic, limit)

    def _fetch_twitter(self, topic, date_from, date_to, limit) -> pd.DataFrame:
        if not TWEEPY_AVAILABLE:
            print("[DataExtractor] tweepy not installed — using demo data.")
            return self._demo_data("Twitter/X", topic, limit)

        if not self.twitter_token:
            print("[DataExtractor] Twitter Bearer Token not set — using demo data.")
            return self._demo_data("Twitter/X", topic, limit)

        try:
            client = tweepy.Client(bearer_token=self.twitter_token, wait_on_rate_limit=True)
            query  = f"{topic} lang:en -is:retweet"
            start  = datetime.combine(date_from, datetime.min.time()).isoformat() + "Z"
            end    = datetime.combine(date_to,   datetime.max.time()).isoformat() + "Z"

            response = client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                tweet_fields=["created_at", "public_metrics", "text"],
                start_time=start,
                end_time=end,
            )

            if not response.data:
                return self._demo_data("Twitter/X", topic, limit)

            rows = []
            for tweet in response.data:
                m = tweet.public_metrics or {}
                rows.append({
                    "text":       tweet.text[:500],
                    "platform":   "Twitter/X",
                    "engagement": m.get("like_count", 0) + m.get("retweet_count", 0),
                    "date":       pd.to_datetime(tweet.created_at).date() if tweet.created_at else None,
                    "url":        f"https://twitter.com/i/web/status/{tweet.id}",
                })
            return pd.DataFrame(rows) if rows else self._demo_data("Twitter/X", topic, limit)

        except Exception as e:
            print(f"[DataExtractor] Twitter API error: {e} — falling back to demo.")
            return self._demo_data("Twitter/X", topic, limit)

    def _fetch_hackernews(self, topic, date_from, date_to, limit) -> pd.DataFrame:
        try:
            ts_from = int(datetime.combine(date_from, datetime.min.time()).timestamp())
            ts_to   = int(datetime.combine(date_to,   datetime.max.time()).timestamp())
            url = (
                f"https://hn.algolia.com/api/v1/search?"
                f"query={requests.utils.quote(topic)}&tags=comment"
                f"&numericFilters=created_at_i>{ts_from},created_at_i<{ts_to}"
                f"&hitsPerPage={min(limit, 200)}"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])

            if not hits:
                return self._demo_data("HackerNews", topic, min(limit, 50))

            rows = []
            for h in hits[:limit]:
                text = (h.get("comment_text") or h.get("title") or "")
                text = re.sub(r"<[^>]+>", "", text)[:500]
                if not text.strip():
                    continue
                rows.append({
                    "text":       text,
                    "platform":   "HackerNews",
                    "engagement": h.get("points") or 0,
                    "date":       pd.to_datetime(h.get("created_at")).date() if h.get("created_at") else None,
                    "url":        h.get("url") or "",
                })
            return pd.DataFrame(rows) if rows else self._demo_data("HackerNews", topic, min(limit, 50))

        except Exception as e:
            print(f"[DataExtractor] HackerNews API error: {e} — falling back to demo.")
            return self._demo_data("HackerNews", topic, min(limit, 50))

    def _demo_data(self, platform: str, topic: str, n: int) -> pd.DataFrame:
        """Generates realistic-looking synthetic data for demonstration / fallback."""
        random.seed(hash(topic + platform) % 2**31)
        base_date = datetime.now()
        rows = []
        templates = [
            f"Just heard about {topic} and it's {{adj}}!",
            f"The latest news on {topic} is {{adj}}.",
            f"Can't stop thinking about {topic} — truly {{adj}} stuff.",
            f"My honest take: {topic} is {{adj}} and here is why.",
            f"Everyone is talking about {topic} but I find it {{adj}}.",
            f"{topic} discussion: {{adj}} outcomes for the community.",
            f"Read a long thread about {topic}. Feeling {{adj}} about it.",
            f"Hot take: {topic} is more {{adj}} than people realize.",
        ]
        adj_pool = {
            "positive": ["amazing", "incredible", "wonderful", "brilliant", "exciting", "great", "fantastic"],
            "negative": ["terrible", "awful", "horrifying", "disappointing", "dreadful", "disastrous"],
            "neutral":  ["interesting", "complex", "nuanced", "debatable", "mixed", "uncertain", "varied"],
        }

        for i in range(n):
            bucket = random.choices(["positive", "negative", "neutral"], weights=[0.45, 0.30, 0.25])[0]
            adj    = random.choice(adj_pool[bucket])
            tmpl   = random.choice(templates)
            text   = tmpl.format(adj=adj) + " " + random.choice(self.SAMPLE_SENTENCES)
            days_back = random.randint(0, 14)
            rows.append({
                "text":       text[:500],
                "platform":   platform,
                "engagement": random.randint(0, 5000),
                "date":       (base_date - timedelta(days=days_back)).date(),
                "url":        "",
            })

        return pd.DataFrame(rows)
