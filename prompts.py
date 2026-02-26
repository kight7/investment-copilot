# prompts.py
# ─────────────────────────────────────────────────────────────────────────────
# All AI prompt templates live here.
# Single source of truth — change prompts in one place, affects whole app.
# ─────────────────────────────────────────────────────────────────────────────


def get_system_prompt() -> str:
    """
    The base personality of our AI analyst.
    This is sent with EVERY request — it defines how the AI behaves.
    
    Why separate from the user prompts?
    System prompt = WHO the AI is (permanent identity)
    User prompt   = WHAT we're asking right now (changes per request)
    """
    return """You are a senior investment analyst with expertise in equity 
research, financial modeling, and market analysis.

Your analysis style:
- Always structured with clear sections
- Balanced: highlight opportunities AND risks equally
- Plain English: explain financial jargon when you use it
- Specific: use the actual numbers provided, don't be vague
- Honest: if data is missing or inconclusive, say so directly
- Never give direct buy/sell advice — frame everything as analysis

Your format:
- Use markdown headers (##) for sections
- Use bullet points for lists of risks or factors
- Bold (**) important numbers and conclusions
- Keep responses focused — no unnecessary padding"""


def build_research_prompt(stock_data: dict, news: list, question: str) -> str:
    """
    Builds the full context prompt for a stock research request.
    
    This is the main prompt used when a user researches a stock.
    It bundles ALL available data into one structured text block
    that the AI can reason over.
    
    stock_data : dict from get_stock_info()
    news       : list of dicts from get_news()
    question   : what the user typed in the app
    """

    # ── Format stock metrics ──────────────────────────────────────────────────
    # We convert the dict into a readable text block
    # The AI performs better with labeled data than raw JSON
    def fmt(value, prefix="", suffix=""):
        """Format a value, return N/A if missing"""
        if value in (None, "N/A", ""):
            return "N/A"
        return f"{prefix}{value}{suffix}"

    def fmt_large(value):
        """Convert large numbers to readable format (4030000000000 → $4.03T)"""
        if value in (None, "N/A", ""):
            return "N/A"
        try:
            value = float(value)
            if value >= 1_000_000_000_000:
                return f"${value/1_000_000_000_000:.2f}T"
            elif value >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            return f"${value:,.0f}"
        except:
            return str(value)

    def fmt_pct(value):
        """Convert decimal to percentage (0.27 → 27.00%)"""
        if value in (None, "N/A", ""):
            return "N/A"
        try:
            return f"{float(value)*100:.2f}%"
        except:
            return str(value)

    metrics_block = f"""
COMPANY: {fmt(stock_data.get('company_name'))} ({fmt(stock_data.get('ticker', ''))})
SECTOR:  {fmt(stock_data.get('sector'))} | INDUSTRY: {fmt(stock_data.get('industry'))}

── PRICE & VALUATION ──────────────────────────────
Current Price:    {fmt(stock_data.get('current_price'), prefix='$')}
52w High:         {fmt(stock_data.get('52w_high'), prefix='$')}
52w Low:          {fmt(stock_data.get('52w_low'), prefix='$')}
Market Cap:       {fmt_large(stock_data.get('market_cap'))}
P/E Ratio:        {fmt(stock_data.get('pe_ratio'))}
Forward P/E:      {fmt(stock_data.get('forward_pe'))}
EPS:              {fmt(stock_data.get('eps'), prefix='$')}

── FUNDAMENTALS ───────────────────────────────────
Revenue:          {fmt_large(stock_data.get('revenue'))}
Profit Margin:    {fmt_pct(stock_data.get('profit_margin'))}
Dividend Yield:   {fmt_pct(stock_data.get('dividend_yield'))}
Beta:             {fmt(stock_data.get('beta'))}

── ANALYST CONSENSUS ──────────────────────────────
Price Target:     {fmt(stock_data.get('analyst_target'), prefix='$')}
Recommendation:   {fmt(stock_data.get('recommendation', '')).upper()}"""

    # ── Format news headlines ─────────────────────────────────────────────────
    if news and not news[0].get("error"):
        news_lines = []
        for i, article in enumerate(news[:5], 1):  # Max 5 articles
            title = article.get("title", "No title")
            date = article.get("published_at", "")[:10]  # Just the date
            news_lines.append(f"  {i}. [{date}] {title}")
        news_block = "\n".join(news_lines)
    else:
        news_block = "  No recent news available"

    # ── Build the full prompt ─────────────────────────────────────────────────
    return f"""Please analyze this stock based on the data below.

═══════════════════════════════════════════
STOCK DATA
═══════════════════════════════════════════
{metrics_block}

── RECENT NEWS (last 5 articles) ──────────
{news_block}

═══════════════════════════════════════════
USER QUESTION
═══════════════════════════════════════════
{question}

Please provide a thorough analysis addressing the question directly,
using the data above as your primary source."""


def build_comparison_prompt(stock1_data: dict, stock2_data: dict, question: str) -> str:
    """
    Builds a side-by-side comparison prompt for two stocks.
    Used when the user wants to compare two tickers.
    
    We'll wire this up in app.py with a 'Compare' tab.
    """
    def fmt(v, prefix="", suffix=""):
        if v in (None, "N/A", ""):
            return "N/A"
        return f"{prefix}{v}{suffix}"

    def fmt_pct(v):
        if v in (None, "N/A", ""):
            return "N/A"
        try:
            return f"{float(v)*100:.2f}%"
        except:
            return str(v)

    name1 = stock1_data.get('company_name', 'Stock 1')
    name2 = stock2_data.get('company_name', 'Stock 2')

    return f"""Compare these two stocks side by side:

═══════════════════════════════════════════
{name1}
═══════════════════════════════════════════
Price:         {fmt(stock1_data.get('current_price'), '$')}
P/E:           {fmt(stock1_data.get('pe_ratio'))}
Forward P/E:   {fmt(stock1_data.get('forward_pe'))}
Market Cap:    {fmt(stock1_data.get('market_cap'))}
Profit Margin: {fmt_pct(stock1_data.get('profit_margin'))}
Beta:          {fmt(stock1_data.get('beta'))}
Analyst Rec:   {fmt(stock1_data.get('recommendation', '')).upper()}
Target Price:  {fmt(stock1_data.get('analyst_target'), '$')}

═══════════════════════════════════════════
{name2}
═══════════════════════════════════════════
Price:         {fmt(stock2_data.get('current_price'), '$')}
P/E:           {fmt(stock2_data.get('pe_ratio'))}
Forward P/E:   {fmt(stock2_data.get('forward_pe'))}
Market Cap:    {fmt(stock2_data.get('market_cap'))}
Profit Margin: {fmt_pct(stock2_data.get('profit_margin'))}
Beta:          {fmt(stock2_data.get('beta'))}
Analyst Rec:   {fmt(stock2_data.get('recommendation', '')).upper()}
Target Price:  {fmt(stock2_data.get('analyst_target'), '$')}

═══════════════════════════════════════════
USER QUESTION
═══════════════════════════════════════════
{question}"""


def build_quant_prompt(ticker: str, signals: dict) -> str:
    """
    Stage 2 prompt — interprets ML model signals in plain English.
    This will be used once we build the quant engine.
    Defined here now so the structure is ready.
    """
    return f"""You are analyzing quantitative signals for {ticker}.

TECHNICAL SIGNALS:
  ML Model Signal:    {signals.get('signal', 'N/A')}
  Confidence:         {signals.get('confidence', 'N/A')}
  RSI (14):           {signals.get('rsi_14', 'N/A')}
  MACD Histogram:     {signals.get('macd_hist', 'N/A')}
  Trend Regime:       {'Bullish (above SMA200)' if signals.get('trend_regime') == 1 else 'Bearish (below SMA200)'}
  Volatility Regime:  {'High volatility' if signals.get('vol_regime') == 1 else 'Low volatility'}

Explain what these signals mean together in plain English.
What is the market telling us about this stock right now?
Keep it to 3-4 clear paragraphs."""


def build_sentiment_prompt(fear_greed_score: int, label: str, components: dict) -> str:
    """
    Stage 3 prompt — interprets the Fear/Greed index in plain English.
    Defined here now, wired up when we build Stage 3.
    """
    return f"""The current market Fear/Greed Index score is {fear_greed_score}/100 — {label}.

COMPONENT BREAKDOWN:
  News Sentiment:  {components.get('news_score', 'N/A')}/100
  VIX Score:       {components.get('vix_score', 'N/A')}/100
  RSI Score:       {components.get('rsi_score', 'N/A')}/100
  Momentum:        {components.get('momentum_score', 'N/A')}/100
  Volume:          {components.get('volume_score', 'N/A')}/100

What does this sentiment regime mean for investors right now?
What historically happens after extreme fear or extreme greed readings?
Give a practical interpretation in 3-4 paragraphs."""


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing prompts.py...\n")

    # Test with fake data
    fake_stock = {
        "company_name": "Apple Inc",
        "ticker": "AAPL",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "current_price": 274.23,
        "market_cap": 4030000000000,
        "pe_ratio": 34.75,
        "forward_pe": 29.49,
        "eps": 7.89,
        "52w_high": 288.62,
        "52w_low": 169.21,
        "dividend_yield": 0.0038,
        "beta": 1.107,
        "revenue": 435617005568,
        "profit_margin": 0.27,
        "analyst_target": 293.07,
        "recommendation": "buy"
    }

    fake_news = [
        {"title": "Apple reports record Q4 earnings", "published_at": "2026-02-20"},
        {"title": "iPhone 17 demand exceeds expectations", "published_at": "2026-02-19"},
    ]

    prompt = build_research_prompt(fake_stock, fake_news, "What is the overall outlook?")

    print("--- Research Prompt Preview ---")
    print(prompt)
    print(f"\n--- Prompt length: {len(prompt)} characters ---")