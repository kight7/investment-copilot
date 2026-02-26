import yfinance as yf
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This is how we access our secret keys without hardcoding them
load_dotenv()


def get_stock_info(ticker: str) -> dict:
    """
    Fetch key financial metrics for a given stock ticker.
    Returns a clean dictionary of fundamentals.
    
    Why a dict? Easy to pass around, easy to convert to a string
    for the AI context later.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # We cherry-pick only what's useful — yfinance returns 100+ fields
        # and most are noise. We want the metrics an analyst actually looks at.
        return {
            "company_name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "beta": info.get("beta", "N/A"),
            "revenue": info.get("totalRevenue", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "analyst_target": info.get("targetMeanPrice", "N/A"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "description": info.get("longBusinessSummary", "N/A")
        }

    except Exception as e:
        # Never let a bad ticker crash the whole app
        # Return a structured error so the UI can handle it gracefully
        return {"error": f"Could not fetch data for '{ticker}': {str(e)}"}


def get_news(ticker: str, company_name: str = "") -> list[dict]:
    """
    Fetch the last 7 news articles related to a stock.
    
    Why both ticker AND company_name?
    Searching for "AAPL" might miss articles that say "Apple".
    We search for both to get better coverage.
    """
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        return [{"error": "NEWS_API_KEY not found in .env file"}]

    # Build the search query — use company name if available, else ticker
    query = company_name if company_name else ticker

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": 7,           # Last 7 articles
        "sortBy": "publishedAt", # Most recent first
        "language": "en"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            return [{"error": f"NewsAPI error: {data.get('message', 'Unknown error')}"}]

        articles = data.get("articles", [])

        # Clean the response — we only need 4 fields, not the full object
        return [
            {
                "title": a.get("title", "No title"),
                "description": a.get("description", "No description"),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", "")
            }
            for a in articles
            if a.get("title")  # Skip articles with no title
        ]

    except requests.exceptions.Timeout:
        return [{"error": "NewsAPI request timed out"}]
    except Exception as e:
        return [{"error": f"Could not fetch news: {str(e)}"}]


def get_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch historical OHLCV price data.
    
    OHLCV = Open, High, Low, Close, Volume
    This is the raw ingredient for Stage 2's entire ML pipeline.
    
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return pd.DataFrame()

        # Keep it clean — drop columns we don't need
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()  # Oldest to newest — important for time-series work

        return df

    except Exception as e:
        print(f"Error fetching price history for {ticker}: {e}")
        return pd.DataFrame()


# ── Quick test ────────────────────────────────────────────────────────────────
# This block only runs when you execute THIS file directly (python data_fetcher.py)
# It does NOT run when other files import from it — that's what if __name__ == "__main__" means
if __name__ == "__main__":
    print("Testing data_fetcher.py...\n")

    print("--- Stock Info ---")
    info = get_stock_info("AAPL")
    for key, val in info.items():
        if key != "description":  # Skip long text in test
            print(f"  {key}: {val}")

    print("\n--- News ---")
    news = get_news("AAPL", "Apple")
    for article in news[:2]:  # Just show first 2
        print(f"  {article.get('title', 'ERROR')}")

    print("\n--- Price History ---")
    df = get_price_history("AAPL", period="1mo")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Latest row:\n{df.tail(1)}")