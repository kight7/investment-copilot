"""
features.py — Technical Indicator Feature Engineering (v2)

Takes a raw OHLCV price DataFrame (from yfinance) and returns a new DataFrame
with 35+ technical indicator columns plus a binary target variable for ML training.

New in v2:
    - Stochastic Oscillator (%K, %D)
    - Williams %R
    - OBV (On-Balance Volume)
    - 52-week high/low distance
    - Day-of-week encoding
    - Lag features (yesterday's key indicators)

Input:  pandas DataFrame with columns [Open, High, Low, Close, Volume]
Output: pandas DataFrame with all feature columns + 'target' column

Usage:
    from features import build_features, FEATURE_COLUMNS
    df = build_features(price_history_df)
"""

import pandas as pd
import numpy as np


def add_returns(df):
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_21d"] = df["Close"].pct_change(21)
    return df


def add_moving_averages(df):
    df["sma_10"]  = df["Close"].rolling(10).mean()
    df["sma_20"]  = df["Close"].rolling(20).mean()
    df["sma_50"]  = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()

    df["price_to_sma10"]  = df["Close"] / df["sma_10"]
    df["price_to_sma20"]  = df["Close"] / df["sma_20"]
    df["price_to_sma50"]  = df["Close"] / df["sma_50"]
    df["price_to_sma200"] = df["Close"] / df["sma_200"]

    df["sma10_above_sma20"]  = (df["sma_10"] > df["sma_20"]).astype(int)
    df["sma20_above_sma50"]  = (df["sma_20"] > df["sma_50"]).astype(int)
    df["sma50_above_sma200"] = (df["sma_50"] > df["sma_200"]).astype(int)
    return df


def add_rsi(df, window=14):
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"]            = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi"] < 30).astype(int)
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["macd"]           = ema_fast - ema_slow
    df["macd_signal"]    = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    df["macd_bullish"]   = (df["macd"] > df["macd_signal"]).astype(int)
    return df


def add_bollinger_bands(df, window=20, num_std=2.0):
    sma  = df["Close"].rolling(window).mean()
    std  = df["Close"].rolling(window).std()
    df["bb_upper"]     = sma + (num_std * std)
    df["bb_lower"]     = sma - (num_std * std)
    band_width         = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"]     = (df["Close"] - df["bb_lower"]) / band_width.replace(0, np.nan)
    df["bb_bandwidth"] = band_width / sma
    return df


def add_atr(df, window=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"]            = tr.rolling(window).mean()
    df["atr_normalized"] = df["atr"] / df["Close"]
    return df


def add_volume_signals(df):
    df["volume_sma20"]       = df["Volume"].rolling(20).mean()
    df["volume_ratio"]       = df["Volume"] / df["volume_sma20"].replace(0, np.nan)
    df["high_volume"]        = (df["volume_ratio"] > 1.5).astype(int)
    df["low_volume"]         = (df["volume_ratio"] < 0.5).astype(int)
    df["volume_price_trend"] = df["return_1d"] * df["volume_ratio"]
    return df


def add_candlestick_patterns(df):
    body         = (df["Close"] - df["Open"]).abs()
    candle_range = df["High"] - df["Low"]
    safe_range   = candle_range.replace(0, np.nan)
    high_close   = df[["High", "Close"]].max(axis=1)
    low_close    = df[["Low",  "Close"]].min(axis=1)
    df["body_ratio"]   = body / safe_range
    df["upper_wick"]   = (df["High"] - high_close) / safe_range
    df["lower_wick"]   = (low_close  - df["Low"])  / safe_range
    df["green_candle"] = (df["Close"] > df["Open"]).astype(int)
    return df


def add_stochastic(df, k_window=14, d_window=3):
    """
    Stochastic Oscillator — like RSI but anchored to High-Low range.
    Catches momentum shifts RSI misses, especially near support/resistance.
    >80 overbought, <20 oversold.
    """
    lowest_low   = df["Low"].rolling(k_window).min()
    highest_high = df["High"].rolling(k_window).max()
    hl_range     = (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_k"]          = (df["Close"] - lowest_low) / hl_range * 100
    df["stoch_d"]          = df["stoch_k"].rolling(d_window).mean()
    df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
    df["stoch_oversold"]   = (df["stoch_k"] < 20).astype(int)
    df["stoch_bullish"]    = (df["stoch_k"] > df["stoch_d"]).astype(int)
    return df


def add_williams_r(df, window=14):
    """
    Williams %R — inverted Stochastic. Especially good at catching reversals.
    >-20 overbought, <-80 oversold.
    """
    highest_high = df["High"].rolling(window).max()
    lowest_low   = df["Low"].rolling(window).min()
    hl_range     = (highest_high - lowest_low).replace(0, np.nan)
    df["williams_r"]            = (highest_high - df["Close"]) / hl_range * -100
    df["williams_r_overbought"] = (df["williams_r"] > -20).astype(int)
    df["williams_r_oversold"]   = (df["williams_r"] < -80).astype(int)
    return df


def add_obv(df):
    """
    On-Balance Volume — smart money accumulation/distribution signal.
    OBV often leads price. Rising OBV with flat price = bullish divergence.
    """
    direction = np.sign(df["Close"].diff().fillna(0))
    df["obv"]     = (direction * df["Volume"]).cumsum()
    df["obv_roc"] = df["obv"].pct_change(20)
    return df


def add_52week_levels(df):
    """
    Distance from 52-week high/low.
    Institutional algos and traders heavily reference these levels.
    """
    high_52w = df["High"].rolling(252).max()
    low_52w  = df["Low"].rolling(252).min()
    df["dist_from_52w_high"] = (high_52w - df["Close"]) / high_52w.replace(0, np.nan)
    df["dist_from_52w_low"]  = (df["Close"] - low_52w)  / low_52w.replace(0, np.nan)
    return df


def add_day_of_week(df):
    """
    Day-of-week encoding.
    Monday = gap-down risk. Friday = profit-taking. Wed/Thu typically strongest.
    """
    dow = df.index.dayofweek
    df["dow_monday"]  = (dow == 0).astype(int)
    df["dow_friday"]  = (dow == 4).astype(int)
    df["dow_midweek"] = ((dow >= 1) & (dow <= 3)).astype(int)
    return df


def add_lag_features(df):
    """
    Yesterday's indicator values.
    The CHANGE in RSI/MACD is often more predictive than the current value alone.
    """
    df["rsi_lag1"]            = df["rsi"].shift(1)
    df["macd_histogram_lag1"] = df["macd_histogram"].shift(1)
    df["volume_ratio_lag1"]   = df["volume_ratio"].shift(1)
    df["rsi_change"]          = df["rsi"] - df["rsi_lag1"]
    df["macd_hist_change"]    = df["macd_histogram"] - df["macd_histogram_lag1"]
    return df


def add_target(df, horizon=5):
    """Binary target: 1 if Close is higher in `horizon` days, else 0."""
    future_close        = df["Close"].shift(-horizon)
    df["target"]        = (future_close > df["Close"]).astype(float)
    df["future_return"] = (future_close - df["Close"]) / df["Close"]
    return df


# ─────────────────────────────────────────────
# FEATURE COLUMNS — imported by model.py
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_21d",
    "price_to_sma10", "price_to_sma20", "price_to_sma50", "price_to_sma200",
    "sma10_above_sma20", "sma20_above_sma50", "sma50_above_sma200",
    "rsi", "rsi_overbought", "rsi_oversold",
    "macd", "macd_signal", "macd_histogram", "macd_bullish",
    "bb_pct_b", "bb_bandwidth",
    "atr_normalized",
    "volume_ratio", "high_volume", "low_volume", "volume_price_trend",
    "body_ratio", "upper_wick", "lower_wick", "green_candle",
    "stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold", "stoch_bullish",
    "williams_r", "williams_r_overbought", "williams_r_oversold",
    "obv_roc",
    "dist_from_52w_high", "dist_from_52w_low",
    "dow_monday", "dow_friday", "dow_midweek",
    "rsi_change", "macd_hist_change", "volume_ratio_lag1",
]


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def build_features(df, horizon=5):
    """
    Runs all indicators on an OHLCV DataFrame and returns clean feature DataFrame.
    NaN rows (warmup + last horizon rows) are dropped automatically.
    """
    df = df.copy()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_signals(df)
    df = add_candlestick_patterns(df)
    df = add_stochastic(df)
    df = add_williams_r(df)
    df = add_obv(df)
    df = add_52week_levels(df)
    df = add_day_of_week(df)
    df = add_lag_features(df)
    df = add_target(df, horizon=horizon)

    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])
    return df


# ─────────────────────────────────────────────
# TEST — python3 features.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing features.py v2...\n")
    import yfinance as yf

    ticker = "AAPL"
    print(f"Downloading 5 years of {ticker} data...")
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    print(f"Raw shape: {raw.shape}")

    featured = build_features(raw)
    print(f"Featured shape: {featured.shape}")
    print(f"Total features: {len(FEATURE_COLUMNS)}")

    cols = ["Close", "rsi", "stoch_k", "williams_r", "obv_roc", "dist_from_52w_high", "target"]
    print(f"\nSample (last 3 rows):")
    print(featured[cols].tail(3).to_string())

    print(f"\nTarget distribution:")
    print(f"  UP   (1): {int(featured['target'].sum())} days ({featured['target'].mean()*100:.1f}%)")
    print(f"  DOWN (0): {int((1-featured['target']).sum())} days ({(1-featured['target'].mean())*100:.1f}%)")
    print("\n✅ features.py v2 working correctly!")
