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
    # Target: 1 if tomorrow's price is higher than today's, otherwise 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.iloc[:-1]  # the last day has no "tomorrow" yet
    return df.dropna()


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

    # Draw and save a chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(data["date"], data["close"])
    ax1.axvline(test["date"].iloc[0], color="red", linestyle="--", label="Start of test period")
    ax1.set_title("Bitcoin closing price (USD)")
    ax1.legend()
    ax2.barh(FEATURES, model.feature_importances_)
    ax2.set_title("What the model relied on most")
    plt.tight_layout()
    plt.savefig("results.png")
    print()
    print("Chart saved as results.png in this folder.")
    print("Note: this is a learning project, not financial advice.")


if __name__ == "__main__":
    main()