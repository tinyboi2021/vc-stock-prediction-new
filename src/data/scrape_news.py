# ============================================================
# NVIDIA (NVDA) NEWS SCRAPER — STRATEGIC VERSION
# DUAL MODE: BULK OPTIMIZED + SMART BACKFILL
# FOLLOWS: NYT Skeleton → Guardian Flesh → Finnhub Skin → AlphaVantage Validation
# ============================================================

import pandas as pd
import requests
import time
import os
import re
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# ================= CONFIG =================
COMPANY = "Apple"
TICKER = "AAPL"

START_DATE = datetime(2017, 1, 1)
END_DATE   = datetime(2025, 12, 31)

INPUT_DATASET = "data/raw/AAPL_2017_2026_LOGGED_BULK.csv"
OUTPUT_FILE = "data/raw/apple_news_data.csv"

# Sleep times (in seconds)
NYT_SLEEP = 7       # NYT rate limit: 10 requests/min
SLEEP = 4           # Default for other APIs
ALPHA_SLEEP = 15    # Alpha Vantage is strict

# Strategic thresholds
HIGH_ACTIVITY_THRESHOLD = 3  # Guardian only queries dates with 3+ NYT articles
TOP_VALIDATION_DAYS = 25     # Alpha Vantage only for top 25 highest-activity days

# ================= API KEYS =================
NYT_KEY = "LcToStxXqsfxZmLwOhPKWgjo0N75VbGAek2PWTR2v7GEmIn0"
GNEWS_KEY = "221ec401857485a7b0228c06cff996dd"
GUARDIAN_KEY = "b5abef80-6e5b-4386-867a-77d4c211c607"
NEWSDATA_KEY = "pub_980912c4ddea4f60b1f3933c7c8e8784"
FINNHUB_KEY = "d5tjm4pr01qt62njpa10d5tjm4pr01qt62njpa1g"
ALPHAVANTAGE_KEY = "OF29MM764TWVLV60"
# NewsData excluded - not useful for historical collection

# ================= FREE-TIER LIMITS =================
LIMITS = {
    "NYT": 4000,        # Can fetch 9 years in ~108 calls
    "GNEWS": 100,       # NEW: Gap filler - 100 calls/day
    "GUARDIAN": 500,    # Use strategically on high-activity dates
    "ALPHAVANTAGE": 25, # Strict - only for validation
    "FINNHUB": 300      # Last year, high volume
}

api_usage = defaultdict(int)
api_errors = defaultdict(list)
api_articles_added = defaultdict(int)
start_time = time.time()

# ====================================================
# EMPTY / PLACEHOLDER DETECTION
# ====================================================

EMPTY_MARKERS = {
    "", " ", "no news", "no significant news",
    "no significant news reported", "none", "nan"
}

def is_effectively_empty(text):
    if text is None:
        return True
    t = str(text).strip().lower()
    if not t or t in EMPTY_MARKERS:
        return True
    if set(t) == {"|"}:
        return True
    return False

# ====================================================
# ENHANCED LOGGING
# ====================================================

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    """Print a formatted section header"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def log_api(date, api, status, message=""):
    """Enhanced API logging with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icon = {
        "SUCCESS": "✅",
        "FAILED": "❌",
        "ERROR": "⚠️",
        "SKIPPED": "⏭️",
        "EMPTY_PAYLOAD": "📭",
        "OUT_OF_RANGE": "📅",
        "STRATEGIC_SKIP": "🎯"
    }.get(status, "ℹ️")
    
    print(f"   [{timestamp}] [{api:12s}] {status_icon} {status:15s} {message}")
    
    if status in ["FAILED", "ERROR"]:
        api_errors[api].append({
            "date": date,
            "status": status,
            "message": message
        })

def check_api_connectivity():
    """Test connectivity to all APIs"""
    print_section("🔌 CHECKING API CONNECTIVITY")
    
    api_tests = {
        "NYT": {
            "url": "https://api.nytimes.com/svc/search/v2/articlesearch.json",
            "params": {"q": "test", "api-key": NYT_KEY},
            "key_check": NYT_KEY
        },
        "GNEWS": {
            "url": "https://gnews.io/api/v4/search",
            "params": {"q": "test", "token": GNEWS_KEY, "lang": "en"},
            "key_check": GNEWS_KEY
        },
        "GUARDIAN": {
            "url": "https://content.guardianapis.com/search",
            "params": {"q": "test", "api-key": GUARDIAN_KEY},
            "key_check": GUARDIAN_KEY
        },
        "FINNHUB": {
            "url": "https://finnhub.io/api/v1/company-news",
            "params": {"symbol": "NVDA", "from": "2024-01-01", "to": "2024-01-01", "token": FINNHUB_KEY},
            "key_check": FINNHUB_KEY
        },
        "ALPHAVANTAGE": {
            "url": "https://www.alphavantage.co/query",
            "params": {"function": "NEWS_SENTIMENT", "tickers": "NVDA", "apikey": ALPHAVANTAGE_KEY},
            "key_check": ALPHAVANTAGE_KEY
        }
    }
    
    connectivity_status = {}
    
    for api_name, config in api_tests.items():
        if not config["key_check"] or config["key_check"].strip() == "":
            print(f"   [{api_name:12s}] 🔑 NO API KEY PROVIDED")
            connectivity_status[api_name] = "NO_KEY"
            continue
        
        try:
            response = requests.get(config["url"], params=config["params"], timeout=10)
            
            if response.status_code == 200:
                print(f"   [{api_name:12s}] ✅ CONNECTED (HTTP 200)")
                connectivity_status[api_name] = "CONNECTED"
            elif response.status_code == 401:
                print(f"   [{api_name:12s}] 🔐 AUTHENTICATION FAILED (HTTP 401)")
                connectivity_status[api_name] = "AUTH_FAILED"
            elif response.status_code == 429:
                print(f"   [{api_name:12s}] ⏸️  RATE LIMITED (HTTP 429)")
                connectivity_status[api_name] = "RATE_LIMITED"
            else:
                print(f"   [{api_name:12s}] ⚠️  HTTP {response.status_code}")
                connectivity_status[api_name] = f"HTTP_{response.status_code}"
                
        except requests.exceptions.Timeout:
            print(f"   [{api_name:12s}] ⏱️  TIMEOUT")
            connectivity_status[api_name] = "TIMEOUT"
        except requests.exceptions.ConnectionError:
            print(f"   [{api_name:12s}] 🔌 CONNECTION ERROR")
            connectivity_status[api_name] = "CONNECTION_ERROR"
        except Exception as e:
            print(f"   [{api_name:12s}] ❌ ERROR: {str(e)[:50]}")
            connectivity_status[api_name] = "ERROR"
    
    return connectivity_status

def print_quota_status():
    """Print current API quota status"""
    print_section("📊 INITIAL API QUOTA STATUS")
    print(f"   {'API':12s} {'Daily Limit':>12s} {'Used':>8s} {'Remaining':>12s}")
    print(f"   {'-' * 50}")
    
    for api, limit in LIMITS.items():
        used = api_usage[api]
        remaining = limit - used
        print(f"   {api:12s} {limit:>12,} {used:>8,} {remaining:>12,}")

# ====================================================
# UTILITY FUNCTIONS
# ====================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]

def article_hash(text):
    return hashlib.sha1(normalize_text(text).encode()).hexdigest()

def article_pack(*parts):
    return " ".join([p for p in parts if p])

def count_articles_in_text(text):
    """Count how many articles are in a pipe-separated string"""
    if is_effectively_empty(text):
        return 0
    return len([a for a in str(text).split("|") if a.strip()])

# ====================================================
# SAFE REQUEST
# ====================================================

def safe_request(url, params, api, identifier):
    """Safe API request with quota checking and better error handling"""
    
    if api_usage[api] >= LIMITS[api]:
        log_api(identifier, api, "SKIPPED", "Quota exhausted")
        return {}
    
    try:
        r = requests.get(url, params=params, timeout=30)
        api_usage[api] += 1
        
        if r.status_code == 429:
            log_api(identifier, api, "FAILED", "RATE LIMITED - Too many requests")
            return {}
        elif r.status_code == 401:
            log_api(identifier, api, "FAILED", "AUTH FAILED - Check API key")
            return {}
        elif r.status_code != 200:
            log_api(identifier, api, "FAILED", f"HTTP {r.status_code}")
            return {}
        
        data = r.json()
        if data is None:
            log_api(identifier, api, "EMPTY_PAYLOAD")
            return {}
        
        log_api(identifier, api, "SUCCESS")
        return data if isinstance(data, (dict, list)) else {}
    
    except Exception as e:
        log_api(identifier, api, "ERROR", str(e)[:50])
        return {}

# ====================================================
# BULK FETCH FUNCTIONS
# ====================================================

def fetch_nyt_bulk(start_date, end_date):
    """Fetch NYT articles - THE SKELETON"""
    articles_by_date = defaultdict(list)
    
    print(f"   📦 Fetching NYT (THE SKELETON): {start_date.date()} to {end_date.date()}")
    
    page = 0
    total_articles = 0
    
    while page < 100:
        data = safe_request(
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
            {
                "q": "Apple OR AAPL OR iPhone OR MacBook OR \"Tim Cook\"",
                "begin_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
                "page": page,
                "api-key": NYT_KEY
            },
            "NYT", f"Page {page}"
        )
        
        if not data:
            break
        
        docs = (data.get("response") or {}).get("docs") or []
        if not docs:
            break
        
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            
            pub_date = doc.get("pub_date", "")
            try:
                art_date = datetime.strptime(pub_date[:10], "%Y-%m-%d").date().isoformat()
            except:
                continue
            
            article_text = article_pack(
                doc.get("headline", {}).get("main", ""),
                doc.get("abstract", ""),
                doc.get("lead_paragraph", "")
            )
            
            if article_text:
                articles_by_date[art_date].append(article_text)
                total_articles += 1
        
        page += 1
        time.sleep(NYT_SLEEP)
        
        if page % 10 == 0:
            print(f"   📊 Progress: Page {page}, {total_articles} articles so far")
    
    print(f"   ✅ NYT: Fetched {total_articles} articles across {len(articles_by_date)} dates")
    return articles_by_date

def fetch_guardian_for_high_activity_dates(articles_by_date, high_activity_dates):
    """Fetch Guardian articles - THE FLESH (only for high-activity dates)"""
    
    print(f"\n   📦 Fetching Guardian (THE FLESH): {len(high_activity_dates)} high-activity dates")
    print(f"   🎯 Strategy: Only dates with {HIGH_ACTIVITY_THRESHOLD}+ NYT articles")
    
    total_articles = 0
    dates_enhanced = 0
    
    for date_str in high_activity_dates:
        if api_usage["GUARDIAN"] >= LIMITS["GUARDIAN"]:
            print(f"   ⏭️  Guardian quota exhausted at {dates_enhanced} dates")
            break
        
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        data = safe_request(
            "https://content.guardianapis.com/search",
            {
                "q": "Apple OR AAPL OR iPhone OR Mac OR \"Tim Cook\"",
                "from-date": date_str,
                "to-date": date_str,
                "show-fields": "bodyText",
                "api-key": GUARDIAN_KEY
            },
            "GUARDIAN", date_str
        )
        
        if not data:
            time.sleep(1)
            continue
        
        results = (data.get("response") or {}).get("results") or []
        
        for r in results:
            if not isinstance(r, dict):
                continue
            
            article_text = article_pack(
                r.get("webTitle", ""),
                r.get("fields", {}).get("bodyText", "")
            )
            
            if article_text:
                articles_by_date[date_str].append(article_text)
                total_articles += 1
        
        if results:
            dates_enhanced += 1
        
        time.sleep(1)
    
    print(f"   ✅ Guardian: Added {total_articles} rich articles to {dates_enhanced} dates")
    return articles_by_date

def fetch_finnhub_recent():
    """Fetch Finnhub articles - THE SKIN (last 365 days only)"""
    articles_by_date = defaultdict(list)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"\n   📦 Fetching Finnhub (THE SKIN): {start_date.date()} to {end_date.date()}")
    print(f"   🎯 Strategy: Last 365 days, high volume")
    
    data = safe_request(
        "https://finnhub.io/api/v1/company-news",
        {
            "symbol": TICKER,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": FINNHUB_KEY
        },
        "FINNHUB", f"{start_date.date()} to {end_date.date()}"
    )
    
    if not data or not isinstance(data, list):
        return articles_by_date
    
    total_articles = 0
    
    for art in data:
        if not isinstance(art, dict):
            continue
        
        timestamp = art.get("datetime", 0)
        try:
            art_date = datetime.fromtimestamp(timestamp).date().isoformat()
        except:
            continue
        
        article_text = article_pack(
            art.get("headline", ""),
            art.get("summary", "")
        )
        
        if article_text:
            articles_by_date[art_date].append(article_text)
            total_articles += 1
    
    print(f"   ✅ Finnhub: Fetched {total_articles} articles across {len(articles_by_date)} dates")
    return articles_by_date

def fetch_alphavantage_for_top_days(top_dates):
    """Fetch Alpha Vantage - THE SPOT-CHECK (validation only)"""
    articles_by_date = defaultdict(list)
    
    print(f"\n   📦 Fetching Alpha Vantage (VALIDATION): Top {len(top_dates)} highest-activity days")
    print(f"   🎯 Strategy: Sentiment scores for market-moving events")
    
    total_articles = 0
    dates_validated = 0
    
    for date_str in top_dates[:TOP_VALIDATION_DAYS]:  # Strict limit
        if api_usage["ALPHAVANTAGE"] >= LIMITS["ALPHAVANTAGE"]:
            print(f"   ⏭️  Alpha Vantage quota exhausted at {dates_validated} dates")
            break
        
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        data = safe_request(
            "https://www.alphavantage.co/query",
            {
                "function": "NEWS_SENTIMENT",
                "tickers": TICKER,
                "time_from": date_obj.strftime("%Y%m%dT0000"),
                "time_to": date_obj.strftime("%Y%m%dT2359"),
                "sort": "EARLIEST",
                "limit": 50,
                "apikey": ALPHAVANTAGE_KEY
            },
            "ALPHAVANTAGE", date_str
        )
        
        if not data:
            time.sleep(ALPHA_SLEEP)
            continue
        
        feed = data.get("feed") or []
        
        for art in feed:
            if not isinstance(art, dict):
                continue
            
            article_text = article_pack(
                art.get("title", ""),
                art.get("summary", ""),
                f"[Sentiment: {art.get('overall_sentiment_label', 'N/A')}]"
            )
            
            if article_text:
                articles_by_date[date_str].append(article_text)
                total_articles += 1
        
        if feed:
            dates_validated += 1
        
        time.sleep(ALPHA_SLEEP)
    
    print(f"   ✅ Alpha Vantage: Added {total_articles} sentiment-tagged articles to {dates_validated} dates")
    return articles_by_date

def fetch_gnews_for_gaps(articles_by_date, max_calls=100):
    """Fetch GNews - THE GAP FILLER (fill empty dates)"""
    
    # Find dates with no articles
    empty_dates = [
        date_str for date_str, articles in articles_by_date.items()
        if len(articles) == 0
    ]
    
    # Sort by date (oldest first for better historical coverage)
    empty_dates_sorted = sorted(empty_dates)
    
    print(f"\n   📦 Fetching GNews (GAP FILLER): {len(empty_dates_sorted)} empty dates")
    print(f"   🎯 Strategy: Fill dates other APIs missed (up to {max_calls} calls)")
    
    total_articles = 0
    dates_filled = 0
    
    for date_str in empty_dates_sorted:
        if api_usage["GNEWS"] >= LIMITS["GNEWS"]:
            print(f"   ⏭️  GNews quota exhausted at {dates_filled} dates")
            break
        
        if dates_filled >= max_calls:
            print(f"   ⏹️  Reached {max_calls} call limit")
            break
        
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        # GNews API format
        data = safe_request(
            "https://gnews.io/api/v4/search",
            {
                "q": "Apple OR AAPL OR iPhone OR Mac",
                "from": date_obj.strftime("%Y-%m-%dT00:00:00Z"),
                "to": date_obj.strftime("%Y-%m-%dT23:59:59Z"),
                "lang": "en",
                "token": GNEWS_KEY
            },
            "GNEWS", date_str
        )
        
        if not data:
            time.sleep(1)
            continue
        
        articles_list = data.get("articles") or []
        
        for art in articles_list:
            if not isinstance(art, dict):
                continue
            
            article_text = article_pack(
                art.get("title", ""),
                art.get("description", ""),
                art.get("content", "")
            )
            
            if article_text:
                articles_by_date[date_str].append(article_text)
                total_articles += 1
        
        if articles_list:
            dates_filled += 1
        
        time.sleep(1)
    
    print(f"   ✅ GNews: Filled {dates_filled} empty dates with {total_articles} articles")
    return articles_by_date

# ====================================================
# STRATEGIC BACKFILL
# ====================================================

def run_strategic_backfill(base_df):
    """STRATEGIC BACKFILL: Layer by layer following the table"""
    print_header("🧠 STRATEGIC BACKFILL MODE")
    print("   Strategy: NYT Skeleton → Guardian Flesh → Finnhub Skin → AlphaVantage Validation")
    
    # ========== ANALYZE EXISTING DATA ==========
    print_section("📊 ANALYZING EXISTING DATA")
    
    base_df["Date"] = base_df["Date"].astype(str)
    
    # Count articles per date
    date_article_counts = {}
    for _, row in base_df.iterrows():
        date_str = row["Date"]
        count = count_articles_in_text(row["news_articles"])
        date_article_counts[date_str] = count
    
    # Identify high-activity dates (3+ articles)
    high_activity_dates = [
        date for date, count in date_article_counts.items()
        if count >= HIGH_ACTIVITY_THRESHOLD
    ]
    
    # Sort by activity level (highest first)
    high_activity_dates_sorted = sorted(
        high_activity_dates,
        key=lambda d: date_article_counts[d],
        reverse=True
    )
    
    print(f"   📅 Total dates in dataset: {len(date_article_counts):,}")
    print(f"   🔥 High-activity dates ({HIGH_ACTIVITY_THRESHOLD}+ articles): {len(high_activity_dates):,}")
    print(f"   🏆 Top date: {high_activity_dates_sorted[0] if high_activity_dates_sorted else 'N/A'} "
          f"({date_article_counts.get(high_activity_dates_sorted[0], 0) if high_activity_dates_sorted else 0} articles)")
    
    # Convert to articles_by_date structure
    articles_by_date = {}
    for _, row in base_df.iterrows():
        date_str = row["Date"]
        articles_text = row["news_articles"]
        
        if is_effectively_empty(articles_text):
            articles_by_date[date_str] = []
        else:
            articles_by_date[date_str] = [a.strip() for a in str(articles_text).split("|") if a.strip()]
    
    # ========== LAYER 2: GNEWS GAP FILLER ==========
    print_section("🔌 LAYER 2: GNEWS - THE GAP FILLER")
    print(f"   Adding articles to empty dates (cost-effective coverage)")
    
    if GNEWS_KEY:
        articles_by_date = fetch_gnews_for_gaps(articles_by_date, max_calls=100)
    
    # ========== LAYER 2.5: HISTORICAL ENRICHMENT (2017-2021) ==========
    print_section("📚 LAYER 2.5: HISTORICAL ENRICHMENT (2017-2021)")
    print(f"   🎯 Strategy: Aggressively fill 2017-2021 dates with low coverage")
    
    # Identify 2017-2021 dates with 0-2 articles (need enrichment)
    historical_start = datetime(2017, 1, 1)
    historical_end = datetime(2021, 12, 31)
    
    dates_needing_historical_enrichment = []
    for date_str, articles in articles_by_date.items():
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if historical_start <= date_obj <= historical_end:
            if len(articles) <= 2:  # 0, 1, or 2 articles
                dates_needing_historical_enrichment.append(date_str)
    
    # Sort by article count (emptiest first, then by date)
    dates_needing_historical_enrichment.sort(key=lambda d: (len(articles_by_date[d]), d))
    
    print(f"   📅 Found {len(dates_needing_historical_enrichment)} dates from 2017-2021 with ≤2 articles")
    print(f"   🔄 Attempting to enrich with NYT and Guardian")
    
    if dates_needing_historical_enrichment:
        historical_filled = 0
        historical_articles_added = 0
        
        # Try NYT first for these dates (single-date queries)
        print(f"\n   🗞️  NYT Historical Pass:")
        for date_str in dates_needing_historical_enrichment[:150]:  # Limit to preserve quota
            if api_usage["NYT"] >= LIMITS["NYT"]:
                print(f"   ⏭️  NYT quota exhausted")
                break
            
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            data = safe_request(
                "https://api.nytimes.com/svc/search/v2/articlesearch.json",
                {
                    "q": "Apple OR AAPL OR iPhone OR MacBook OR \"Tim Cook\"",
                    "begin_date": date_obj.strftime("%Y%m%d"),
                    "end_date": date_obj.strftime("%Y%m%d"),
                    "api-key": NYT_KEY
                },
                "NYT", date_str
            )
            
            if data:
                docs = (data.get("response") or {}).get("docs") or []
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    
                    article_text = article_pack(
                        doc.get("headline", {}).get("main", ""),
                        doc.get("abstract", ""),
                        doc.get("lead_paragraph", "")
                    )
                    
                    if article_text:
                        articles_by_date[date_str].append(article_text)
                        api_articles_added["NYT"] += 1
                        historical_articles_added += 1
                
                if docs:
                    historical_filled += 1
            
            time.sleep(NYT_SLEEP)
        
        print(f"   ✅ NYT: Added {historical_articles_added} articles to {historical_filled} dates")
        
        # Then Guardian for dates still with ≤2 articles
        print(f"\n   📰 Guardian Historical Pass:")
        guardian_historical_filled = 0
        guardian_historical_articles = 0
        
        dates_still_needing = [d for d in dates_needing_historical_enrichment if len(articles_by_date[d]) <= 2]
        
        for date_str in dates_still_needing[:200]:  # More aggressive with Guardian
            if api_usage["GUARDIAN"] >= LIMITS["GUARDIAN"]:
                print(f"   ⏭️  Guardian quota exhausted")
                break
            
            data = safe_request(
                "https://content.guardianapis.com/search",
                {
                    "q": "Apple OR AAPL OR iPhone OR Mac OR \"Tim Cook\"",
                    "from-date": date_str,
                    "to-date": date_str,
                    "show-fields": "bodyText",
                    "api-key": GUARDIAN_KEY
                },
                "GUARDIAN", date_str
            )
            
            if data:
                results = (data.get("response") or {}).get("results") or []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    
                    article_text = article_pack(
                        r.get("webTitle", ""),
                        r.get("fields", {}).get("bodyText", "")
                    )
                    
                    if article_text:
                        articles_by_date[date_str].append(article_text)
                        api_articles_added["GUARDIAN"] += 1
                        guardian_historical_articles += 1
                
                if results:
                    guardian_historical_filled += 1
            
            time.sleep(1)
        
        print(f"   ✅ Guardian: Added {guardian_historical_articles} articles to {guardian_historical_filled} dates")
        print(f"\n   📊 Historical Enrichment Summary:")
        print(f"      Total articles added: {historical_articles_added + guardian_historical_articles}")
        print(f"      Dates enhanced: {historical_filled + guardian_historical_filled}")
    
    # Recalculate high-activity dates after historical enrichment
    high_activity_dates = [
        date_str for date_str, articles in articles_by_date.items()
        if len(articles) >= HIGH_ACTIVITY_THRESHOLD
    ]
    
    high_activity_dates_sorted = sorted(
        high_activity_dates,
        key=lambda d: len(articles_by_date[d]),
        reverse=True
    )
    
    # ========== LAYER 3: GUARDIAN FLESH ==========
    print_section("🥩 LAYER 3: GUARDIAN - THE FLESH")
    print(f"   Adding rich content to {len(high_activity_dates_sorted[:300])} highest-activity dates")
    print(f"   (Reduced from 500 to preserve quota for historical enrichment)")
    
    if GUARDIAN_KEY and high_activity_dates_sorted:
        # Reduce Guardian flesh quota to account for historical pass
        remaining_guardian_quota = LIMITS["GUARDIAN"] - api_usage["GUARDIAN"]
        guardian_flesh_limit = min(300, remaining_guardian_quota)
        
        articles_by_date = fetch_guardian_for_high_activity_dates(
            articles_by_date,
            high_activity_dates_sorted[:guardian_flesh_limit]
        )
    
    # ========== LAYER 4: FINNHUB SKIN ==========
    print_section("🌐 LAYER 4: FINNHUB - THE SKIN")
    print(f"   Adding company-specific tags for last 365 days")
    
    if FINNHUB_KEY:
        finnhub_articles = fetch_finnhub_recent()
        for date_str, articles_list in finnhub_articles.items():
            if date_str in articles_by_date:
                articles_by_date[date_str].extend(articles_list)
                api_articles_added["FINNHUB"] += len(articles_list)
    
    # ========== LAYER 5: ALPHA VANTAGE VALIDATION ==========
    print_section("🎯 LAYER 5: ALPHA VANTAGE - VALIDATION")
    print(f"   Adding sentiment scores to top {TOP_VALIDATION_DAYS} market-moving days")
    
    if ALPHAVANTAGE_KEY and high_activity_dates_sorted:
        alphavantage_articles = fetch_alphavantage_for_top_days(high_activity_dates_sorted)
        for date_str, articles_list in alphavantage_articles.items():
            if date_str in articles_by_date:
                articles_by_date[date_str].extend(articles_list)
                api_articles_added["ALPHAVANTAGE"] += len(articles_list)
    
    return articles_by_date

# ====================================================
# BULK MODE
# ====================================================

def run_bulk_mode():
    """Build the skeleton with NYT, then layer strategically"""
    print_header("🏗️  BULK MODE - BUILDING THE SKELETON")
    
    articles_by_date = {}
    all_dates = pd.date_range(START_DATE, END_DATE).date
    
    for date in all_dates:
        date_str = date.isoformat()
        articles_by_date[date_str] = []
    
    print(f"   📅 Total dates to fill: {len(all_dates):,}")
    
    # ========== LAYER 1: NYT SKELETON ==========
    print_section("🦴 LAYER 1: NYT - THE SKELETON")
    
    if NYT_KEY:
        nyt_articles = fetch_nyt_bulk(START_DATE, END_DATE)
        for date_str, articles_list in nyt_articles.items():
            if date_str in articles_by_date:
                articles_by_date[date_str] = articles_list
                api_articles_added["NYT"] += len(articles_list)
    
    # Identify high-activity dates for Guardian
    high_activity_dates = [
        date_str for date_str, articles in articles_by_date.items()
        if len(articles) >= HIGH_ACTIVITY_THRESHOLD
    ]
    
    high_activity_sorted = sorted(
        high_activity_dates,
        key=lambda d: len(articles_by_date[d]),
        reverse=True
    )
    
    print(f"\n   📊 Skeleton built: {sum(1 for a in articles_by_date.values() if a):,} dates with articles")
    print(f"   🔥 High-activity dates identified: {len(high_activity_dates):,}")
    
    # ========== LAYER 2: GNEWS GAP FILLER ==========
    print_section("🔌 LAYER 2: GNEWS - THE GAP FILLER")
    
    if GNEWS_KEY:
        articles_by_date = fetch_gnews_for_gaps(articles_by_date, max_calls=100)
    
    # ========== LAYER 2.5: HISTORICAL ENRICHMENT (2017-2021) ==========
    print_section("📚 LAYER 2.5: HISTORICAL ENRICHMENT (2017-2021)")
    print(f"   🎯 Strategy: Aggressively fill 2017-2021 dates with low coverage")
    
    # Identify 2017-2021 dates with 0-2 articles
    historical_start = datetime(2017, 1, 1)
    historical_end = datetime(2021, 12, 31)
    
    dates_needing_historical_enrichment = []
    for date_str, articles in articles_by_date.items():
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if historical_start <= date_obj <= historical_end:
            if len(articles) <= 2:
                dates_needing_historical_enrichment.append(date_str)
    
    dates_needing_historical_enrichment.sort(key=lambda d: (len(articles_by_date[d]), d))
    
    print(f"   📅 Found {len(dates_needing_historical_enrichment)} dates from 2017-2021 with ≤2 articles")
    print(f"   🔄 Enriching with NYT and Guardian")
    
    if dates_needing_historical_enrichment:
        historical_filled = 0
        historical_articles_added = 0
        
        # NYT historical pass
        print(f"\n   🗞️  NYT Historical Pass:")
        for date_str in dates_needing_historical_enrichment[:150]:
            if api_usage["NYT"] >= LIMITS["NYT"]:
                break
            
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            data = safe_request(
                "https://api.nytimes.com/svc/search/v2/articlesearch.json",
                {
                    "q": "Apple OR AAPL OR iPhone OR MacBook OR \"Tim Cook\"",
                    "begin_date": date_obj.strftime("%Y%m%d"),
                    "end_date": date_obj.strftime("%Y%m%d"),
                    "api-key": NYT_KEY
                },
                "NYT", date_str
            )
            
            if data:
                docs = (data.get("response") or {}).get("docs") or []
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    
                    article_text = article_pack(
                        doc.get("headline", {}).get("main", ""),
                        doc.get("abstract", ""),
                        doc.get("lead_paragraph", "")
                    )
                    
                    if article_text:
                        articles_by_date[date_str].append(article_text)
                        api_articles_added["NYT"] += 1
                        historical_articles_added += 1
                
                if docs:
                    historical_filled += 1
            
            time.sleep(NYT_SLEEP)
        
        print(f"   ✅ NYT: Added {historical_articles_added} articles to {historical_filled} dates")
        
        # Guardian historical pass
        print(f"\n   📰 Guardian Historical Pass:")
        guardian_historical_filled = 0
        guardian_historical_articles = 0
        
        dates_still_needing = [d for d in dates_needing_historical_enrichment if len(articles_by_date[d]) <= 2]
        
        for date_str in dates_still_needing[:200]:
            if api_usage["GUARDIAN"] >= LIMITS["GUARDIAN"]:
                break
            
            data = safe_request(
                "https://content.guardianapis.com/search",
                {
                    "q": "Apple OR AAPL OR iPhone OR Mac OR \"Tim Cook\"",
                    "from-date": date_str,
                    "to-date": date_str,
                    "show-fields": "bodyText",
                    "api-key": GUARDIAN_KEY
                },
                "GUARDIAN", date_str
            )
            
            if data:
                results = (data.get("response") or {}).get("results") or []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    
                    article_text = article_pack(
                        r.get("webTitle", ""),
                        r.get("fields", {}).get("bodyText", "")
                    )
                    
                    if article_text:
                        articles_by_date[date_str].append(article_text)
                        api_articles_added["GUARDIAN"] += 1
                        guardian_historical_articles += 1
                
                if results:
                    guardian_historical_filled += 1
            
            time.sleep(1)
        
        print(f"   ✅ Guardian: Added {guardian_historical_articles} articles to {guardian_historical_filled} dates")
        print(f"\n   📊 Historical Enrichment Summary:")
        print(f"      Total articles added: {historical_articles_added + guardian_historical_articles}")
        print(f"      Dates enhanced: {historical_filled + guardian_historical_filled}")
    
    # Recalculate high-activity after historical enrichment and GNews
    high_activity_dates = [
        date_str for date_str, articles in articles_by_date.items()
        if len(articles) >= HIGH_ACTIVITY_THRESHOLD
    ]
    
    high_activity_sorted = sorted(
        high_activity_dates,
        key=lambda d: len(articles_by_date[d]),
        reverse=True
    )
    
    # ========== LAYER 3: GUARDIAN FLESH ==========
    print_section("🥩 LAYER 3: GUARDIAN - THE FLESH")
    
    if GUARDIAN_KEY and high_activity_sorted:
        # Adjust Guardian flesh quota based on historical usage
        remaining_guardian_quota = LIMITS["GUARDIAN"] - api_usage["GUARDIAN"]
        guardian_flesh_limit = min(300, remaining_guardian_quota)
        
        print(f"   Adding rich content to {min(len(high_activity_sorted), guardian_flesh_limit)} highest-activity dates")
        print(f"   (Adjusted quota: {remaining_guardian_quota} calls remaining)")
        
        articles_by_date = fetch_guardian_for_high_activity_dates(
            articles_by_date,
            high_activity_sorted[:guardian_flesh_limit]
        )
    
    # ========== LAYER 4: FINNHUB SKIN ==========
    print_section("🌐 LAYER 4: FINNHUB - THE SKIN")
    
    if FINNHUB_KEY:
        finnhub_articles = fetch_finnhub_recent()
        for date_str, articles_list in finnhub_articles.items():
            if date_str in articles_by_date:
                articles_by_date[date_str].extend(articles_list)
                api_articles_added["FINNHUB"] += len(articles_list)
    
    # ========== LAYER 5: ALPHA VANTAGE VALIDATION ==========
    print_section("🎯 LAYER 5: ALPHA VANTAGE - VALIDATION")
    
    if ALPHAVANTAGE_KEY and high_activity_sorted:
        alphavantage_articles = fetch_alphavantage_for_top_days(high_activity_sorted)
        for date_str, articles_list in alphavantage_articles.items():
            if date_str in articles_by_date:
                articles_by_date[date_str].extend(articles_list)
                api_articles_added["ALPHAVANTAGE"] += len(articles_list)
    
    return articles_by_date

# ====================================================
# MAIN
# ====================================================

def run():
    """Main execution"""
    
    print_header(f"🧠 {COMPANY.upper()} ({TICKER}) - STRATEGIC NEWS SCRAPER")
    print(f"   Strategy: Layer-by-layer following optimal table")
    print(f"   Date Range: {START_DATE.date()} to {END_DATE.date()}")
    
    connectivity = check_api_connectivity()
    print_quota_status()
    
    # ========== MODE DETECTION ==========
    print_section("🔍 MODE DETECTION")
    
    mode = None
    base_df = None
    
    if os.path.exists(INPUT_DATASET):
        try:
            base_df = pd.read_csv(INPUT_DATASET)
            print(f"   ✅ Input file found: {INPUT_DATASET}")
            print(f"   → Running STRATEGIC BACKFILL MODE")
            mode = "backfill"
        except Exception as e:
            print(f"   ⚠️  Error reading: {str(e)}")
            print(f"   → Falling back to BULK MODE")
            mode = "bulk"
    else:
        print(f"   ℹ️  No input file")
        print(f"   → Running BULK MODE")
        mode = "bulk"
    
    # ========== RUN ==========
    if mode == "bulk":
        articles_by_date = run_bulk_mode()
    else:
        articles_by_date = run_strategic_backfill(base_df)
    
    # ========== SAVE ==========
    print_section("💾 SAVING OUTPUT")
    
    rows = []
    for date_str in sorted(articles_by_date.keys()):
        articles = articles_by_date[date_str]
        
        if isinstance(articles, list):
            articles_text = " | ".join(articles) if articles else ""
        elif isinstance(articles, str):
            articles_text = articles
        else:
            articles_text = ""
        
        rows.append({
            "Date": date_str,
            "news_articles": articles_text
        })
    
    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"   ✅ Saved: {OUTPUT_FILE}")
    print(f"   📊 Rows: {len(final_df):,}")
    
    # ========== REPORT ==========
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_header("✅ STRATEGIC COLLECTION COMPLETE")
    
    print(f"\n📰 ARTICLES BY LAYER:")
    total_articles = 0
    for api_name in ["NYT", "GNEWS", "GUARDIAN", "FINNHUB", "ALPHAVANTAGE"]:
        count = api_articles_added[api_name]
        if count > 0:
            role = {
                "NYT": "SKELETON",
                "GNEWS": "GAP-FILLER",
                "GUARDIAN": "FLESH",
                "FINNHUB": "SKIN",
                "ALPHAVANTAGE": "VALIDATION"
            }[api_name]
            print(f"   {api_name:12s} ({role:12s}): {count:>6,} articles")
            total_articles += count
    
    print(f"   {'─' * 40}")
    print(f"   {'TOTAL':27s}: {total_articles:>6,} articles")
    
    print(f"\n📊 API USAGE:")
    for api, limit in LIMITS.items():
        used = api_usage[api]
        pct = (used / limit * 100) if limit > 0 else 0
        print(f"   {api:12s}: {used:>4,}/{limit:>4,} ({pct:>5.1f}%)")
    
    final_with_news = sum(1 for _, row in final_df.iterrows() if not is_effectively_empty(row["news_articles"]))
    
    print(f"\n📈 COVERAGE:")
    print(f"   Dates with news: {final_with_news:,}/{len(final_df):,} ({final_with_news/len(final_df)*100:.1f}%)")
    
    print(f"\n⏱️  TIME: {int(execution_time//60)}m {int(execution_time%60)}s")
    
    print("\n" + "=" * 70)
    print("  🎉 STRATEGIC COLLECTION COMPLETE!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run()