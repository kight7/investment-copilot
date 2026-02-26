"""
app.py — Streamlit Frontend for AI Investment Research Copilot (v2)

Three tabs:
    1. Research  — AI analysis + fundamentals + news
    2. ML Signal — LightGBM quant model prediction
    3. Compare   — side-by-side AI comparison

Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from research import research_stock, compare_stocks
from model import predict_today

import os

# Bridge: load secrets from Streamlit Cloud into environment variables
# Locally, python-dotenv handles this via .env file
try:
    for key in ["GROQ_API_KEY", "NEWS_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Investment Research Copilot",
    page_icon="📈",
    layout="wide",
)


# ─────────────────────────────────────────────
# FORMATTING HELPERS  (module-level)
# ─────────────────────────────────────────────

def fmt_large(n):
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def fmt_val(v, prefix="", suffix="", decimals=2):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return f"{prefix}{v}{suffix}"
    try:
        return f"{prefix}{float(v):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Investment Copilot")
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    AI-powered stock research tool backed by:
    - **Llama 3.3 70B** via Groq
    - **LightGBM** quant model (47 features)
    - **Yahoo Finance** for market data
    - **NewsAPI** for headlines
    """)
    st.markdown("---")
    st.subheader("⌨️ Shortcuts")
    st.markdown("""
    | Key | Action |
    |-----|--------|
    | `Enter` | Submit ticker |
    | `Ctrl+Enter` | Run |
    """)
    st.markdown("---")
    st.caption("Stage 2 of 3 · Quant ML Engine")


# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("AI Investment Research Copilot")
st.markdown("AI analysis + quantitative ML signals — powered by Llama 3.3 70B and LightGBM.")
st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_research, tab_ml, tab_compare = st.tabs([
    "🔍 Research",
    "🤖 ML Signal",
    "⚖️ Compare",
])


# ══════════════════════════════════════════════
# TAB 1 — RESEARCH
# ══════════════════════════════════════════════
with tab_research:

    col_ticker, col_question = st.columns([1, 3])
    with col_ticker:
        ticker_input = st.text_input(
            "Stock Ticker", placeholder="e.g. AAPL", key="research_ticker"
        ).strip().upper()
    with col_question:
        question_input = st.text_area(
            "Custom Question (optional)",
            placeholder="Ask anything about this stock...",
            height=68, key="research_question",
        )

    research_btn = st.button("🔍 Research This Stock", type="primary", key="research_btn")

    if research_btn:
        if not ticker_input:
            st.error("Please enter a stock ticker symbol.")
        else:
            with st.spinner(f"Researching {ticker_input}... ~10 seconds"):
                result = research_stock(ticker_input, question_input or None)
            st.session_state["research_result"] = result

    result = st.session_state.get("research_result")

    if result:
        if not result.get("success"):
            st.error(f"❌ Could not fetch data for **{result.get('ticker', '')}**. Check the ticker and try again.")
        else:
            ticker       = result["ticker"]
            company_name = result.get("company_name", ticker)
            stock_data   = result.get("stock_data", {})
            news         = result.get("news", [])
            price_hist   = result.get("price_history")
            analysis     = result.get("analysis", "")
            question     = result.get("question", "")

            st.subheader(f"{company_name}  ({ticker})")
            if question:
                st.caption(f"Question: *{question}*")
            st.markdown("---")

            # Key Metrics
            st.markdown("#### 📊 Key Metrics")
            price     = stock_data.get("current_price")
            mkt_cap   = stock_data.get("market_cap")
            pe_ratio  = stock_data.get("pe_ratio")
            high_52w  = stock_data.get("52_week_high")
            low_52w   = stock_data.get("52_week_low")
            volume    = stock_data.get("volume")
            div_yield = stock_data.get("dividend_yield")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("💵 Price",      fmt_val(price, prefix="$"))
            m2.metric("🏢 Market Cap", fmt_large(mkt_cap))
            m3.metric("📉 P/E Ratio",  fmt_val(pe_ratio, decimals=1))
            m4.metric("📈 52W High",   fmt_val(high_52w, prefix="$"))
            m5.metric("📉 52W Low",    fmt_val(low_52w, prefix="$"))

            m6, m7, *_ = st.columns(5)
            m6.metric("📦 Volume", f"{volume:,}" if volume else "N/A")
            if div_yield is None:
                dy_str = "N/A"
            elif isinstance(div_yield, str):
                dy_str = div_yield
            else:
                dy_str = f"{div_yield:.2f}%"
            m7.metric("💰 Div. Yield", dy_str)
            st.markdown("---")

            # Price Chart
            if price_hist is not None and not price_hist.empty and "Close" in price_hist.columns:
                st.markdown("#### 📈 6-Month Price History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_hist.index, y=price_hist["Close"],
                    mode="lines", name="Close",
                    line=dict(color="#00C9A7", width=2),
                    fill="tozeroy", fillcolor="rgba(0,201,167,0.08)",
                ))
                fig.update_layout(
                    xaxis_title="Date", yaxis_title="Price (USD)",
                    hovermode="x unified", height=320,
                    margin=dict(l=0, r=0, t=10, b=0),
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")

            # AI Analysis
            st.markdown("#### 🤖 AI Analysis")
            if analysis:
                st.markdown(analysis)
            else:
                st.info("No analysis returned.")
            st.markdown("---")

            # News
            st.markdown("#### 📰 Recent News")
            if news:
                for article in news:
                    title       = article.get("title", "Untitled")
                    description = article.get("description", "No description available.")
                    url         = article.get("url", "#")
                    source      = article.get("source", {})
                    src_name    = source.get("name", "Unknown") if isinstance(source, dict) else str(source)
                    published   = article.get("publishedAt", "")[:10]
                    with st.expander(f"📄 {title}  —  *{src_name}* · {published}"):
                        st.write(description)
                        st.markdown(f"[Read full article →]({url})")
            else:
                st.info("No recent news found.")


# ══════════════════════════════════════════════
# TAB 2 — ML SIGNAL
# ══════════════════════════════════════════════
with tab_ml:

    st.markdown("""
    The ML Signal tab runs a **LightGBM model** trained on 5 years of historical price data
    with 47 technical indicators. It predicts whether the stock will be **higher or lower in 5 trading days**.

    > ⚠️ This is a quantitative signal, not financial advice. CV Accuracy ~57% means the model
    is right slightly more than it's wrong — similar to professional quant signals.
    """)
    st.markdown("---")

    ml_ticker = st.text_input(
        "Stock Ticker", placeholder="e.g. TSLA", key="ml_ticker"
    ).strip().upper()

    ml_btn = st.button("🤖 Run ML Signal", type="primary", key="ml_btn")

    if ml_btn:
        if not ml_ticker:
            st.error("Please enter a ticker symbol.")
        else:
            with st.spinner(f"Training model on 5 years of {ml_ticker} data... ~30 seconds"):
                # Fetch 5 years of data for the model (more than research tab's 6 months)
                raw = yf.download(ml_ticker, period="5y", interval="1d",
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    st.session_state["ml_result"] = {"error": f"No data found for {ml_ticker}."}
                else:
                    ml_result = predict_today(raw)
                    ml_result["ticker"] = ml_ticker
                    st.session_state["ml_result"] = ml_result

    ml_result = st.session_state.get("ml_result")

    if ml_result:
        if "error" in ml_result:
            st.error(f"❌ {ml_result['error']}")
        else:
            ticker_label = ml_result.get("ticker", "")
            signal       = ml_result["signal"]
            confidence   = ml_result["confidence"]
            direction    = ml_result["direction"]
            cv_acc       = ml_result["cv_accuracy"]
            cv_auc       = ml_result["cv_auc"]
            n_rows       = ml_result["n_training_rows"]

            # ── Signal Card ──────────────────────────
            st.markdown(f"### {ticker_label} — ML Prediction")

            signal_colors = {"BUY": "#00C9A7", "SELL": "#FF6B6B", "NEUTRAL": "#FFD93D"}
            signal_icons  = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}
            color         = signal_colors.get(signal, "#888")
            icon          = signal_icons.get(signal, "⚪")

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}11);
                border: 2px solid {color};
                border-radius: 12px;
                padding: 24px 32px;
                margin: 12px 0;
            ">
                <div style="font-size: 48px; font-weight: 900; color: {color};">
                    {icon} {signal}
                </div>
                <div style="font-size: 20px; color: #ccc; margin-top: 8px;">
                    Model predicts price will go <strong style="color:{color};">{direction}</strong>
                    in 5 trading days
                </div>
                <div style="font-size: 32px; font-weight: 700; color: white; margin-top: 12px;">
                    {confidence}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # ── Model Stats ──────────────────────────
            st.markdown("#### 📊 Model Reliability")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("✅ CV Accuracy",     f"{cv_acc}%",
                      help="Walk-forward cross-validation accuracy. Random = 50%.")
            c2.metric("📐 AUC Score",       str(cv_auc),
                      help="Area under ROC curve. 0.5 = random, 1.0 = perfect.")
            c3.metric("📚 Training Rows",   f"{n_rows:,}",
                      help="5 years of daily OHLCV data after feature engineering.")
            c4.metric("🎯 Features",        "47",
                      help="RSI, MACD, Bollinger Bands, Stochastic, OBV, Williams %R, and more.")

            st.markdown("---")

            # ── Today's Indicators ───────────────────
            st.markdown("#### 🔬 Today's Technical Indicators")

            indicators    = ml_result["current_features"]
            ind_col1, ind_col2 = st.columns(2)

            indicator_info = {
                "RSI":             ("14-day Relative Strength Index. >70 overbought, <30 oversold.", 70, 30),
                "Stochastic %K":   ("14-day Stochastic. >80 overbought, <20 oversold.", 80, 20),
                "Williams %R":     ("Williams %R. >-20 overbought, <-80 oversold.", -20, -80),
                "MACD Histogram":  ("MACD histogram. Positive = bullish momentum.", None, None),
                "BB %B":           ("Bollinger Band position. >1 above band, <0 below band.", 1, 0),
                "Volume Ratio":    ("Today's volume vs 20-day average. >1.5 = high conviction.", 1.5, 0.5),
                "5D Return %":     ("5-day price return in %.", None, None),
                "ATR %":           ("Average True Range as % of price. Higher = more volatile.", None, None),
                "Price/SMA200":    ("Price relative to 200-day MA. >1 = above long-term trend.", None, None),
                "Dist 52W High %": ("% below 52-week high. 0 = at the high.", None, None),
            }

            items = list(indicators.items())
            for i, (name, value) in enumerate(items):
                col = ind_col1 if i % 2 == 0 else ind_col2
                info = indicator_info.get(name, (name, None, None))
                col.metric(name, value, help=info[0])

            st.markdown("---")

            # ── Confidence Gauge ─────────────────────
            st.markdown("#### 🎯 Confidence Meter")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "P(UP) %", "font": {"size": 18}},
                gauge={
                    "axis":  {"range": [0, 100], "tickwidth": 1},
                    "bar":   {"color": color},
                    "steps": [
                        {"range": [0,  40],  "color": "rgba(255,107,107,0.2)"},
                        {"range": [40, 60],  "color": "rgba(255,217,61,0.2)"},
                        {"range": [60, 100], "color": "rgba(0,201,167,0.2)"},
                    ],
                    "threshold": {
                        "line":  {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": confidence,
                    },
                },
                number={"suffix": "%", "font": {"size": 36}},
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("---")

            # ── Feature Importance Chart ─────────────
            st.markdown("#### 🏆 Top 10 Most Important Features")
            st.caption("Which indicators drove this prediction the most.")

            fi_data = ml_result["feature_importance"]
            fi_df   = {
                "Feature":    [f["feature"]    for f in fi_data],
                "Importance": [f["importance"] for f in fi_data],
            }

            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature"],
                orientation="h",
                marker_color="#00C9A7",
                marker_line_width=0,
            ))
            fig_fi.update_layout(
                height=360,
                xaxis_title="Importance Score",
                yaxis={"autorange": "reversed"},
                margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown("---")

            # ── CV Fold Results ──────────────────────
            st.markdown("#### 📅 Walk-Forward CV Fold Results")
            st.caption("Each fold trains on the past and tests on the future. No data leakage.")

            folds = ml_result["cv_folds"]
            fold_df_data = {
                "Fold":     [f"Fold {f['fold']}" for f in folds],
                "Accuracy": [round(f["accuracy"] * 100, 1) for f in folds],
                "AUC":      [f["auc"] for f in folds],
                "Test Days":[f["test_n"] for f in folds],
            }

            fig_folds = go.Figure()
            fig_folds.add_trace(go.Bar(
                name="Accuracy %",
                x=fold_df_data["Fold"],
                y=fold_df_data["Accuracy"],
                marker_color="#00C9A7",
                text=[f"{v}%" for v in fold_df_data["Accuracy"]],
                textposition="outside",
            ))
            fig_folds.add_hline(
                y=50, line_dash="dash", line_color="#888",
                annotation_text="Random baseline (50%)",
                annotation_position="bottom right",
            )
            fig_folds.update_layout(
                height=300,
                yaxis={"range": [0, 90], "title": "Accuracy %"},
                margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark",
            )
            st.plotly_chart(fig_folds, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — COMPARE
# ══════════════════════════════════════════════
with tab_compare:

    st.markdown("Compare two stocks side-by-side with an AI-generated analysis.")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        ticker_a = st.text_input(
            "First Ticker", placeholder="e.g. AAPL", key="compare_ticker_a"
        ).strip().upper()
    with col_b:
        ticker_b = st.text_input(
            "Second Ticker", placeholder="e.g. MSFT", key="compare_ticker_b"
        ).strip().upper()

    compare_btn = st.button("⚖️ Compare Stocks", type="primary", key="compare_btn")

    if compare_btn:
        if not ticker_a or not ticker_b:
            st.error("Please enter both ticker symbols.")
        elif ticker_a == ticker_b:
            st.error("Please enter two different tickers.")
        else:
            with st.spinner(f"Comparing {ticker_a} vs {ticker_b}... ~15 seconds"):
                comparison = compare_stocks(ticker_a, ticker_b)
            st.session_state["compare_result"] = comparison

    comparison = st.session_state.get("compare_result")

    if comparison:
        if not comparison.get("success"):
            st.error("❌ Comparison failed. Check both tickers and try again.")
        else:
            name_a = comparison.get("stock_a", {}).get("company_name", ticker_a)
            name_b = comparison.get("stock_b", {}).get("company_name", ticker_b)
            data_a = comparison.get("stock_a", {}).get("stock_data", {})
            data_b = comparison.get("stock_b", {}).get("stock_data", {})

            st.subheader(f"{name_a}  vs  {name_b}")
            st.markdown("---")

            st.markdown("#### 📊 Side-by-Side Metrics")
            col_label, col_left, col_right = st.columns([1.5, 2, 2])
            col_label.markdown("**Metric**")
            col_left.markdown(f"**{name_a}**")
            col_right.markdown(f"**{name_b}**")

            metrics = [
                ("💵 Price",      "current_price", "$",  2),
                ("🏢 Market Cap", "market_cap",    None, 0),
                ("📉 P/E Ratio",  "pe_ratio",      "",   1),
                ("📈 52W High",   "52_week_high",  "$",  2),
                ("📉 52W Low",    "52_week_low",   "$",  2),
            ]

            for label, key, prefix, decs in metrics:
                val_a = data_a.get(key)
                val_b = data_b.get(key)
                if key == "market_cap":
                    str_a = fmt_large(val_a)
                    str_b = fmt_large(val_b)
                else:
                    str_a = fmt_val(val_a, prefix=prefix or "", decimals=decs)
                    str_b = fmt_val(val_b, prefix=prefix or "", decimals=decs)
                col_label.write(label)
                col_left.write(str_a)
                col_right.write(str_b)

            st.markdown("---")
            st.markdown("#### 🤖 AI Comparison Analysis")
            analysis_text = comparison.get("analysis", "")
            st.markdown(analysis_text) if analysis_text else st.info("No analysis returned.")
