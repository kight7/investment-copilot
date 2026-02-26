import data_fetcher
import prompts
import ai_client


def research_stock(ticker: str, question: str = None) -> dict:
    """
    Main orchestrator function — runs the full research pipeline.

    Takes a ticker and optional question, returns everything
    the UI needs in one clean dictionary.

    ticker   : stock ticker string e.g. "AAPL"
    question : user's question — defaults to full summary if None
    """

    # Default question if user doesn't type anything
    if not question or question.strip() == "":
        question = (
            "Give me a complete investment research summary. Include: "
            "company overview, valuation analysis, key strengths, "
            "key risks, and overall outlook."
        )

    ticker = ticker.upper().strip()

    # ── Step 1: Fetch stock fundamentals ─────────────────────────────────────
    # This hits Yahoo Finance via yfinance
    print(f"[research] Fetching stock data for {ticker}...")
    stock_data = data_fetcher.get_stock_info(ticker)

    # Check if the ticker was valid — data_fetcher returns {"error": ...} if not
    if "error" in stock_data or stock_data.get("current_price") in (None, "N/A", ""):
        return {
             "success": False,
             "ticker": ticker,
             "error": stock_data.get("error", f"No data found for ticker '{ticker}'. It may be invalid or delisted.")
    }

    # Inject ticker into stock_data so prompts can use it
    stock_data["ticker"] = ticker

    # ── Step 2: Fetch recent news ─────────────────────────────────────────────
    # Use company name for better search results (e.g. "Apple" not "AAPL")
    company_name = stock_data.get("company_name", ticker)
    print(f"[research] Fetching news for {company_name}...")
    news = data_fetcher.get_news(ticker, company_name)

    # ── Step 3: Fetch price history ───────────────────────────────────────────
    # We fetch this now so app.py can display a chart without a second call
    print(f"[research] Fetching price history for {ticker}...")
    price_history = data_fetcher.get_price_history(ticker, period="6mo")

    # ── Step 4: Build the prompt ──────────────────────────────────────────────
    # Combines all data into a structured text block for the AI
    print(f"[research] Building analysis prompt...")
    prompt = prompts.build_research_prompt(stock_data, news, question)

    # ── Step 5: Get AI analysis ───────────────────────────────────────────────
    print(f"[research] Calling AI for analysis...")
    analysis = ai_client.analyze_stock(
        context=prompt,
        question=question
    )

    # ── Step 6: Package everything for the UI ─────────────────────────────────
    # app.py gets ONE dictionary with everything it needs
    # No need for app.py to know how any of this was fetched
    return {
        "success": True,
        "ticker": ticker,
        "company_name": company_name,
        "stock_data": stock_data,
        "news": news,
        "price_history": price_history,
        "analysis": analysis,
        "question": question
    }


def compare_stocks(ticker1: str, ticker2: str, question: str = None) -> dict:
    """
    Runs research on two tickers and returns a side-by-side comparison.
    Used by the Compare tab in app.py.
    """
    if not question or question.strip() == "":
        question = (
            "Compare these two stocks. Which is more attractive right now "
            "and why? Consider valuation, growth, risk, and analyst consensus."
        )

    ticker1 = ticker1.upper().strip()
    ticker2 = ticker2.upper().strip()

    print(f"[research] Comparing {ticker1} vs {ticker2}...")

    # Fetch both stocks
    stock1 = data_fetcher.get_stock_info(ticker1)
    stock2 = data_fetcher.get_stock_info(ticker2)

    if "error" in stock1:
        return {"success": False, "error": f"{ticker1}: {stock1['error']}"}
    if "error" in stock2:
        return {"success": False, "error": f"{ticker2}: {stock2['error']}"}

    stock1["ticker"] = ticker1
    stock2["ticker"] = ticker2

    # Build comparison prompt
    prompt = prompts.build_comparison_prompt(stock1, stock2, question)

    # Get AI comparison
    analysis = ai_client.analyze_stock(context=prompt, question=question)

    return {
        "success": True,
        "ticker1": ticker1,
        "ticker2": ticker2,
        "stock1_data": stock1,
        "stock2_data": stock2,
        "analysis": analysis,
        "question": question
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing research.py...\n")

    print("=" * 50)
    print("TEST 1: Full stock research")
    print("=" * 50)
    result = research_stock("AAPL", "What is the valuation and risk profile?")

    if result["success"]:
        print(f"\nCompany: {result['company_name']}")
        print(f"Price History rows: {len(result['price_history'])}")
        print(f"News articles: {len(result['news'])}")
        print(f"\n--- AI Analysis ---")
        print(result["analysis"])
    else:
        print(f"Error: {result['error']}")

    print("\n" + "=" * 50)
    print("TEST 2: Invalid ticker (should fail gracefully)")
    print("=" * 50)
    bad_result = research_stock("XYZXYZ999")
    print(f"Success: {bad_result['success']}")
    print(f"Error message: {bad_result.get('error', 'N/A')}")