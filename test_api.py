import datetime
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def separator(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'-' * pad} {title} {'-' * pad}")
    else:
        print(f"\n{'-' * width}")

def test_api():
    print(f"Testing API at {BASE_URL}...")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Health Check ────────────────────────────────────────────────────────
    separator("HEALTH CHECK")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            health = resp.json()
            print(f"[PASS] Status : {health['status']}")
            print(f"       Redis  : {'[YES] connected' if health.get('redis') else '[NO] not connected'}")
        else:
            print(f"[FAIL] Health Check status: {resp.status_code}")
            print(f"       Response: {resp.text}")
            return
    except Exception as e:
        print("[FAIL] Could not connect to API. Is it running?", e)
        return

    # ── 2. News Fetch (with Redis daily cache) ─────────────────────────────────
    separator("APPLE NEWS  (cached 24 h / day)")
    print("[INFO] Requesting /news ... may take a moment on first call ...")
    start_time = time.time()
    try:
        resp = requests.get(f"{BASE_URL}/news")
        duration = time.time() - start_time

        if resp.status_code == 200:
            news = resp.json()
            cached_label = "CACHE HIT" if news.get("cached") else "LIVE FETCH"
            print(f"[PASS] {cached_label}  ({duration:.2f}s)")
            print(f"       Articles   : {news['article_count']}")
            print(f"       Cache TTL  : {news.get('cache_expires_in', 'n/a')}")
            print()

            articles = news.get("articles", [])
            for idx, article in enumerate(articles, 1):
                # Print the first 120 chars as a preview
                preview = article[:120].replace("\n", " ").strip()
                if len(article) > 120:
                    preview += "…"
                print(f"  [{idx:02d}] {preview}")
        elif resp.status_code == 503:
            print("[WARN] News API key not configured — skipping news section.")
        else:
            print(f"[FAIL] /news returned {resp.status_code}:")
            print(f"       {resp.text}")
    except Exception as e:
        print(f"[FAIL] /news (Exception): {e}")

    # ── 3. First Inference (Cold Start) ────────────────────────────────────────
    separator("PREDICTION  — Run 1 (Cold Start)")
    print("[INFO] Requesting /predict ... please wait (model inference) ...")
    start_time = time.time()
    try:
        resp = requests.get(f"{BASE_URL}/predict")
        duration = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            print(f"[PASS] Run 1 Success ({duration:.2f}s)")
            print(f"       Price      : ${data['current_price']:.2f}")
            print(f"       Sentiment  : {data['sentiment_score']:.4f}  "
                  f"(Conf: {data['sentiment_confidence']:.2%})")
            print(f"       Articles   : {data['article_volume']}")
            print(f"       Summary    : {data['summary']}")
            print(f"       Cached     : {data.get('cached')}")
            
            if "sentiment_logs" in data and data["sentiment_logs"]:
                print("\n  Sentiment Calculation Logs:")
                for log in data["sentiment_logs"]:
                    print(f"    [Score: {log['score']:>5.2f} | Conf: {log['confidence']:.2f} | Driver: {log['driver']:<12}] -> {log['article_preview']}")

            print(f"\n  Forecast (next {len(data.get('forecast', []))} days):")
            print(f"  {'Date':<14} {'Price':>10}")
            print(f"  {'-'*14} {'-'*10}")
            for item in data.get("forecast", []):
                print(f"  {item['date']:<14} ${item['price']:>9.2f}")
        else:
            print(f"[FAIL] Run 1 failed with status {resp.status_code}:")
            print(f"       {resp.text}")
            return
    except Exception as e:
        print(f"[FAIL] Run 1 (Exception): {e}")
        return

    # ── 4. Second Inference (Cached) ───────────────────────────────────────────
    separator("PREDICTION  — Run 2 (Cached)")
    print("[INFO] Requesting /predict again ...")
    start_time = time.time()
    try:
        resp = requests.get(f"{BASE_URL}/predict")
        duration = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            print(f"[PASS] Run 2 Success ({duration:.2f}s)")
            print(f"       Cached     : {data.get('cached')}")

            if duration < 1.0 and data.get("cached") is True:
                print("[PASS] Cache is working correctly [YES]")
            else:
                print("[WARN] Response might not be cached properly.")
        else:
            print(f"[FAIL] Run 2 failed: {resp.status_code}")
            print(f"       {resp.text}")
    except Exception as e:
        print(f"[FAIL] Run 2 (Exception): {e}")

    separator()
    print("All tests complete.")

if __name__ == "__main__":
    test_api()
