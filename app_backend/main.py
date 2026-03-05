import sys
import os
import json
import redis
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import timedelta

# ── Path Setup ─────────────────────────────────────────────────────────────────
# When running inside Docker (WORKDIR /app) the project root is /app.
# When running locally (app_backend/main.py) the parent is the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)   # one level up from app_backend/

# Insert at position 0 so our modules take priority
sys.path.insert(0, _PROJECT_ROOT)

# ── Imports from existing project scripts ──────────────────────────────────────
# These are kept outside try-except so any ImportError surfaces immediately
# in the container logs rather than being silently swallowed.
from real_time_inference import (        # noqa: E402
    fetch_apple_news,
    calculate_real_time_sentiment,
    get_real_data,
    NEWS_DATA_API_KEY,
    OLLAMA_MODEL,                        # kept for reference
)
from modelInference import (             # noqa: E402
    load_best_model,
    predict_with_model,
    MODELS_DIR,
    SCALERS_DIR,
    RESULTS_CSV,
    DEVICE,
)

# time_features lives in informerMLopsUpdated; provide a fallback
try:
    from informerMLopsUpdated import time_features  # noqa: E402
except ImportError:
    def time_features(dates):  # type: ignore[no-redef]
        """Minimal fallback when informerMLopsUpdated is unavailable."""
        return np.column_stack([
            dates.month.values,
            dates.day.values,
            dates.weekday.values,
            np.zeros(len(dates)),
        ])

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Real-Time Stock Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis ──────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)


# ── Response Models ────────────────────────────────────────────────────────────
class ForecastItem(BaseModel):
    date: str
    price: float


class SentimentLog(BaseModel):
    article_preview: str
    score: float
    confidence: float
    driver: str

class PredictionResponse(BaseModel):
    current_price: float
    forecast: List[ForecastItem]
    sentiment_score: float
    sentiment_confidence: float
    article_volume: int
    summary: str
    cached: bool
    sentiment_logs: List[SentimentLog] = []


class NewsResponse(BaseModel):
    articles: List[str]
    article_count: int
    cached: bool
    cache_expires_in: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    try:
        redis_ok = redis_client.ping()
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok}


@app.get("/news", response_model=NewsResponse)
def get_news(force_refresh: bool = False):
    """
    Return the latest Apple Inc. news articles.

    Results are cached in Redis for 24 hours (daily refresh).
    Pass `?force_refresh=true` to bypass the cache and hit the News API immediately.
    """
    news_cache_key = "news:aapl"
    NEWS_TTL = 86400  # 24 hours – refresh once per day

    # ── Serve from cache ───────────────────────────────────────────────────────
    if not force_refresh:
        cached_news = redis_client.get(news_cache_key)
        if cached_news:
            ttl_remaining = redis_client.ttl(news_cache_key)
            hours, rem = divmod(ttl_remaining, 3600)
            mins = rem // 60
            data = json.loads(cached_news)
            data["cached"] = True
            data["cache_expires_in"] = f"{hours}h {mins}m"
            return data

    if not NEWS_DATA_API_KEY or NEWS_DATA_API_KEY in ("YOUR_API_KEY_HERE", ""):
        raise HTTPException(status_code=503, detail="News API key not configured.")

    try:
        articles = fetch_apple_news(NEWS_DATA_API_KEY)
        if not articles:
            raise HTTPException(status_code=502, detail="No articles returned from News API.")

        payload = {
            "articles": articles,
            "article_count": len(articles),
            "cached": False,
            "cache_expires_in": "24h 0m",
        }

        # Store with daily TTL
        cached_payload = dict(payload)
        cached_payload["cached"] = True
        redis_client.setex(news_cache_key, NEWS_TTL, json.dumps(cached_payload))

        return payload

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict", response_model=PredictionResponse)
def get_prediction(force_refresh: bool = False):
    """
    Return AAPL 16-day price forecast + live sentiment.

    Results are cached in Redis for 1 hour.
    Pass `?force_refresh=true` to bypass the cache.
    """
    cache_key = "prediction:aapl"

    # ── Serve from cache ───────────────────────────────────────────────────────
    if not force_refresh:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

    try:
        # 1. Load best model
        horizon = 16
        scenario = "With Sentiment"
        model, scaler_target, scaler_cov, config = load_best_model(
            horizon, scenario, RESULTS_CSV, MODELS_DIR, SCALERS_DIR, DEVICE
        )

        # 2. Fetch market data
        model_feature_cols = config.get("feature_cols", None)
        df_features, df_full = get_real_data(feature_cols=model_feature_cols)

        # 3. Live news sentiment — reuse the Redis news cache to avoid extra API calls
        sent_score = 0.0
        sent_std   = 0.0
        vol        = 0
        conf_ratio = 0.0
        sentiment_logs = []

        if NEWS_DATA_API_KEY and NEWS_DATA_API_KEY not in ("YOUR_API_KEY_HERE", ""):
            # Try the daily news cache first; fall back to a fresh API call
            _cached_news = redis_client.get("news:aapl")
            if _cached_news:
                articles = json.loads(_cached_news).get("articles", [])
            else:
                articles = fetch_apple_news(NEWS_DATA_API_KEY)
                # Seed the cache so future /news calls also benefit
                if articles:
                    _payload = {"articles": articles, "article_count": len(articles),
                                "cached": True, "cache_expires_in": "24h 0m"}
                    redis_client.setex("news:aapl", 86400, json.dumps(_payload))

            if articles:
                vol = len(articles)
                sent_score, sent_std, _vol, conf_ratio, sentiment_logs = calculate_real_time_sentiment(articles)
                last_idx = df_features.index[-1]
                df_features.at[last_idx, "sentiment_score"]        = sent_score
                df_features.at[last_idx, "sentiment_std"]          = sent_std
                df_features.at[last_idx, "article_volume"]         = vol
                df_features.at[last_idx, "high_confidence_ratio"]  = conf_ratio

        # 4. Prepare tensors
        features   = df_features.values.astype(np.float32)
        dates      = df_full.index
        time_marks = time_features(dates.dt if hasattr(dates, "dt") else dates)

        # 5. Run inference
        predictions = predict_with_model(
            model, scaler_target, scaler_cov, features, time_marks,
            config["seq_len"], config["label_len"], config["pred_len"],
            config["input_dim"], DEVICE,
        )

        # 6. Build response
        last_date       = df_full.index[-1]
        flat            = predictions.flatten()
        forecast_items  = [
            ForecastItem(date=(last_date + timedelta(days=i)).strftime("%Y-%m-%d"), price=float(p))
            for i, p in enumerate(flat, 1)
        ]

        last_price = float(features[-1, 0])
        avg_pred   = float(flat.mean())
        change     = (avg_pred - last_price) / last_price * 100

        response = PredictionResponse(
            current_price        = last_price,
            forecast             = forecast_items,
            sentiment_score      = float(sent_score),
            sentiment_confidence = float(conf_ratio),
            article_volume       = int(vol),
            summary              = f"Current: ${last_price:.2f} | Avg Pred: ${avg_pred:.2f} | Change: {change:+.2f}%",
            cached               = False,   # False for the live response; cached copy stored as True
            sentiment_logs       = sentiment_logs,
        )

        # 7. Store in Redis (TTL = 1 hour)
        cached_dict = response.dict()
        cached_dict["cached"] = True
        redis_client.setex(cache_key, 3600, json.dumps(cached_dict))

        return response

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
