import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

# Load secrets from Streamlit Cloud if available
try:
    for key in ["GROQ_API_KEY", "NEWS_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass
load_dotenv()

# Groq runs Llama 3.3 70B on their own ultra-fast hardware
# Free tier: 14,400 requests/day — more than enough for this project
# Llama 3.3 70B is Meta's best open-source model, excellent for analysis
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a senior investment analyst with expertise in 
equity research, financial modeling, and market analysis.

When given stock data and asked a question, you:
- Provide clear, structured analysis
- Highlight both opportunities AND risks — never one-sided
- Explain financial metrics in plain English when relevant
- Give a balanced view: bull case, bear case, key risks
- Are direct and specific — avoid vague statements
- Format your response with clear sections when the answer is long
- Never give direct buy/sell financial advice — frame as analysis only

If the data provided is incomplete or unavailable, say so clearly 
rather than making things up."""


def analyze_stock(context: str, question: str, provider: str = "groq") -> str:
    """
    Send stock context + user question to Groq/Llama and return analysis.

    context : formatted stock data string (built by research.py)
    question: what the user wants to know
    provider: kept as parameter for future flexibility
    """
    user_message = f"""Here is the current data for the stock I want to analyze:

{context}

My question: {question}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI analysis failed: {str(e)}"


def test_connection() -> str:
    """Quick sanity check — verifies API key works without burning tokens."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=20,
            messages=[
                {"role": "user", "content": "Reply with exactly: Groq connection successful"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Connection failed: {str(e)}"


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ai_client.py with Groq + Llama 3.3 70B...\n")

    print("--- Connection Test ---")
    print(test_connection())

    print("\n--- Analysis Test ---")
    fake_context = """
    Company: Apple Inc (AAPL)
    Current Price: $274.23
    P/E Ratio: 34.75
    Forward P/E: 29.49
    EPS: $7.89
    52w High: $288.62 | 52w Low: $169.21
    Market Cap: $4.03 Trillion
    Analyst Target: $293.07
    Recommendation: Buy
    Profit Margin: 27%
    Beta: 1.107
    """
    result = analyze_stock(
        fake_context,
        "What does this data tell us about Apple's valuation and risk profile?"
    )
    print(result)