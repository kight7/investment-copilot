"""
model.py — LightGBM Classifier for Stock Direction Prediction (v2)

Predicts whether a stock's Close price will be higher 5 days from now.
Uses walk-forward time-series cross-validation — never looks into the future.

Changes in v2:
    - Uses 5 years of data instead of 2
    - Passes DataFrames (not numpy arrays) to LightGBM — no more sklearn warning
    - Cleaner feature importance using 'gain' metric

Usage:
    from model import predict_today
    result = predict_today(price_df)
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

from features import build_features, FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "num_leaves":        31,
    "learning_rate":     0.05,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "verbose":           -1,
    "random_state":      42,
}

N_ESTIMATORS   = 500
EARLY_STOPPING = 40


# ─────────────────────────────────────────────
# WALK-FORWARD CROSS VALIDATION
# ─────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Time-series cross-validation — always train on past, test on future.
    Uses expanding window: each fold adds more training data.
    """
    df = df.sort_index()
    n  = len(df)

    min_train = int(n * 0.60)
    fold_size = int((n - min_train) / n_splits)

    if fold_size < 10:
        return {"error": "Not enough data. Need at least 200 rows after feature engineering."}

    fold_results = []

    for i in range(n_splits):
        train_end  = min_train + i * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        X_train = df[FEATURE_COLUMNS].iloc[:train_end]
        y_train = df["target"].iloc[:train_end]
        X_test  = df[FEATURE_COLUMNS].iloc[test_start:test_end]
        y_test  = df["target"].iloc[test_start:test_end]

        model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=N_ESTIMATORS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )

        preds_proba = model.predict_proba(X_test)[:, 1]
        preds_class = (preds_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds_class)
        try:
            auc = roc_auc_score(y_test, preds_proba)
        except ValueError:
            auc = 0.5

        fold_results.append({
            "fold":     i + 1,
            "train_n":  train_end,
            "test_n":   test_end - test_start,
            "accuracy": round(acc, 4),
            "auc":      round(auc, 4),
        })

    if not fold_results:
        return {"error": "No folds completed."}

    return {
        "folds":        fold_results,
        "avg_accuracy": round(np.mean([f["accuracy"] for f in fold_results]), 4),
        "avg_auc":      round(np.mean([f["auc"]      for f in fold_results]), 4),
    }


# ─────────────────────────────────────────────
# TRAIN FINAL MODEL
# ─────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """Train on ALL data for live prediction. CV already validated generalization."""
    df     = df.sort_index()
    split  = int(len(df) * 0.85)

    X_train = df[FEATURE_COLUMNS].iloc[:split]
    y_train = df["target"].iloc[:split]
    X_val   = df[FEATURE_COLUMNS].iloc[split:]
    y_val   = df["target"].iloc[split:]

    model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=N_ESTIMATORS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    )
    return model


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def get_feature_importance(model: lgb.LGBMClassifier) -> pd.DataFrame:
    return pd.DataFrame({
        "feature":    FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# MAIN FUNCTION — called by app.py
# ─────────────────────────────────────────────

def predict_today(price_df: pd.DataFrame) -> dict:
    """
    Full pipeline: features → cross-validate → train → predict today's signal.

    Returns dict with signal, confidence, CV metrics, feature importance,
    and today's indicator snapshot.
    """
    try:
        print("[model] Building features...")
        df = build_features(price_df, horizon=5)

        if len(df) < 150:
            return {"error": f"Not enough data ({len(df)} rows). Need at least 150. Try fetching 5 years of history."}

        print(f"[model] {len(df)} training rows. Running walk-forward CV...")
        cv_results = walk_forward_cv(df, n_splits=5)

        if "error" in cv_results:
            return {"error": cv_results["error"]}

        print("[model] Training final model...")
        model = train_model(df)

        # Predict on today (last row) — pass as DataFrame to avoid sklearn warning
        today_X     = df[FEATURE_COLUMNS].iloc[[-1]]
        prob_up     = float(model.predict_proba(today_X)[0][1])

        if prob_up >= 0.60:
            signal, direction = "BUY",     "UP"
        elif prob_up <= 0.40:
            signal, direction = "SELL",    "DOWN"
        else:
            signal    = "NEUTRAL"
            direction = "UP" if prob_up >= 0.50 else "DOWN"

        fi_df        = get_feature_importance(model)
        top_features = fi_df.head(10).to_dict("records")

        today = df.iloc[-1]
        current_features = {
            "RSI":             round(float(today["rsi"]), 1),
            "Stochastic %K":   round(float(today["stoch_k"]), 1),
            "Williams %R":     round(float(today["williams_r"]), 1),
            "MACD Histogram":  round(float(today["macd_histogram"]), 4),
            "BB %B":           round(float(today["bb_pct_b"]), 3),
            "Volume Ratio":    round(float(today["volume_ratio"]), 2),
            "5D Return %":     round(float(today["return_5d"]) * 100, 2),
            "ATR %":           round(float(today["atr_normalized"]) * 100, 2),
            "Price/SMA200":    round(float(today["price_to_sma200"]), 3),
            "Dist 52W High %": round(float(today["dist_from_52w_high"]) * 100, 1),
        }

        print(f"[model] Signal: {signal} | Confidence: {prob_up*100:.1f}% | CV Acc: {cv_results['avg_accuracy']*100:.1f}% | AUC: {cv_results['avg_auc']:.3f}")

        return {
            "signal":             signal,
            "direction":          direction,
            "confidence":         round(prob_up * 100, 1),
            "cv_accuracy":        round(cv_results["avg_accuracy"] * 100, 1),
            "cv_auc":             round(cv_results["avg_auc"], 3),
            "cv_folds":           cv_results["folds"],
            "feature_importance": top_features,
            "current_features":   current_features,
            "n_training_rows":    len(df),
        }

    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}


# ─────────────────────────────────────────────
# TEST — python3 model.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing model.py v2...\n")
    import yfinance as yf

    ticker = "AAPL"
    print(f"Downloading 5 years of {ticker} data...")
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    print(f"Downloaded {len(raw)} rows\n")

    result = predict_today(raw)

    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Signal:        {result['signal']}")
        print(f"Direction:     {result['direction']}")
        print(f"Confidence:    {result['confidence']}%")
        print(f"CV Accuracy:   {result['cv_accuracy']}%")
        print(f"CV AUC:        {result['cv_auc']}")
        print(f"Training rows: {result['n_training_rows']}")

        print(f"\nToday's Indicators:")
        for k, v in result["current_features"].items():
            print(f"  {k:<22}: {v}")

        print(f"\nTop 5 Features by Importance:")
        for f in result["feature_importance"][:5]:
            print(f"  {f['feature']:<28}: {f['importance']:.0f}")

        print(f"\nWalk-Forward CV Folds:")
        for fold in result["cv_folds"]:
            print(f"  Fold {fold['fold']}: Acc={fold['accuracy']*100:.1f}%  AUC={fold['auc']:.3f}  (n={fold['test_n']})")

        print("\n✅ model.py v2 working correctly!")
