import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
PARAMS = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000}
FEE = 0.001  # 0.1% cost every time the position changes

FEATURES = [
    "return_1d",
    "return_3d",
    "return_7d",
    "ma7_gap",
    "ma21_gap",
    "volatility_7d",
    "volume_change",
]


def download_prices():
    """Download about 1000 days of Bitcoin daily prices."""
    for url in URLS:
        try:
            response = requests.get(url, params=PARAMS, timeout=20)
            response.raise_for_status()
            df = pd.DataFrame(response.json()).iloc[:, :6]
            df.columns = ["time", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["time"], unit="ms")
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df[["date", "close", "volume"]]
        except Exception as error:
            print("Could not use", url, "-", error)
    raise SystemExit("Could not download price data. Check your internet connection.")


def add_features(df):
    """Turn raw prices into numbers the model can learn from."""
    df = df.copy()
    df["return_1d"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_7d"] = df["close"].pct_change(7)
    df["ma7_gap"] = df["close"] / df["close"].rolling(7).mean() - 1
    df["ma21_gap"] = df["close"] / df["close"].rolling(21).mean() - 1
    df["volatility_7d"] = df["return_1d"].rolling(7).std()
    df["volume_change"] = df["volume"].pct_change()
    # Tomorrow's real return (used only for the backtest, never as a feature)
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1
    # Target: 1 if tomorrow's price is higher than today's, otherwise 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.iloc[:-1]  # the last day has no "tomorrow" yet
    return df.dropna()


def backtest(test, predictions):
    """Simulate trading the model's signals on the unseen test days."""
    result = test[["date", "next_return"]].copy()
    # Predicted up -> long (+1), predicted down -> short (-1)
    result["position"] = [1 if p == 1 else -1 for p in predictions]
    # Fee is charged on every change: opening = 1 unit, flipping = 2 units
    change = result["position"].diff().abs().fillna(1)
    result["strategy_return"] = result["position"] * result["next_return"] - FEE * change
    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()
    result["buy_hold_equity"] = (1 + result["next_return"]).cumprod()
    flips = int((change == 2).sum())
    return result, flips


def max_drawdown(equity):
    """Biggest fall from a previous peak."""
    return (equity / equity.cummax() - 1).min()


def main():
    print("Downloading Bitcoin price history...")
    prices = download_prices()
    print("Downloaded", len(prices), "days of data")

    data = add_features(prices)

    # Split by time: train on the older 80%, test on the newest 20%
    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]
    print("Training days:", len(train), "| Testing days:", len(test))

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train[FEATURES], train["target"])
    predictions = model.predict(test[FEATURES])

    model_accuracy = accuracy_score(test["target"], predictions)
    most_common = train["target"].mode()[0]
    baseline_accuracy = accuracy_score(test["target"], [most_common] * len(test))

    print()
    print("Model accuracy on unseen days:", round(model_accuracy * 100, 1), "%")
    print("Baseline (always guess the most common move):", round(baseline_accuracy * 100, 1), "%")
    print()
    print("Feature importance:")
    for name, score in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(" ", name, round(score, 3))

    # Backtest: simulate long/short trading on the test days
    result, flips = backtest(test, predictions)
    long_days = int((result["position"] == 1).sum())
    short_days = int((result["position"] == -1).sum())
    strategy_total = result["strategy_equity"].iloc[-1] - 1
    buy_hold_total = result["buy_hold_equity"].iloc[-1] - 1

    # Baseline 1: always short
    short_equity = (1 - result["next_return"]).cumprod()
    always_short_total = short_equity.iloc[-1] * (1 - FEE) - 1

    # Baseline 2: random long/short signals, averaged over 200 runs
    rng = np.random.default_rng(42)
    random_totals = []
    for _ in range(200):
        random_predictions = rng.integers(0, 2, len(test))
        random_result, _ = backtest(test, random_predictions)
        random_totals.append(random_result["strategy_equity"].iloc[-1] - 1)
    random_average = sum(random_totals) / len(random_totals)

    print()
    print("Backtest on the", len(test), "test days (fee", FEE * 100, "% per change):")
    print("  Model long/short return:", round(strategy_total * 100, 1), "%")
    print("  Buy and hold return:    ", round(buy_hold_total * 100, 1), "%")
    print("  Always short return:    ", round(always_short_total * 100, 1), "%")
    print("  Random signals (avg of 200 runs):", round(random_average * 100, 1), "%")
    print("  Model max drawdown:     ", round(max_drawdown(result["strategy_equity"]) * 100, 1), "%")
    print("  Buy and hold drawdown:  ", round(max_drawdown(result["buy_hold_equity"]) * 100, 1), "%")
    print("  Long days:", long_days, "| Short days:", short_days, "| Position flips:", flips)

    # Draw and save charts
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(data["date"], data["close"])
    axes[0].axvline(test["date"].iloc[0], color="red", linestyle="--", label="Start of test period")
    axes[0].set_title("Bitcoin closing price (USD)")
    axes[0].legend()
    axes[1].barh(FEATURES, model.feature_importances_)
    axes[1].set_title("What the model relied on most")
    axes[2].plot(result["date"], result["strategy_equity"], label="Model long/short")
    axes[2].plot(result["date"], result["buy_hold_equity"], label="Buy and hold")
    axes[2].set_title("Backtest: growth of 1.0 on unseen test days")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("results.png")
    print()
    print("Chart saved as results.png in this folder.")
    print("Note: this is a learning project, not financial advice.")


if __name__ == "__main__":
    main()