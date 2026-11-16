# Bitcoin Price Direction Predictor

A beginner machine learning project in Python that tries to predict whether Bitcoin's price will go up or down the next day, then backtests a long/short strategy on those predictions.

![Results](results.png)

## What it does

1. Downloads about 1,000 days of daily Bitcoin prices from the public Binance API
2. Builds features: 1, 3 and 7-day returns, 7 and 21-day moving average gaps, 7-day volatility and volume change
3. Trains a Random Forest classifier (scikit-learn) on the older 80% of the data
4. Tests it on the newest 20% (196 days it has never seen)
5. Backtests long/short trading on those days (long when it predicts up, short when it predicts down, with a 0.1% fee on every position change)
6. Compares the strategy with buy and hold, always short and random signals, then saves charts

## Results (test period: 196 unseen days)

| | Result |
|---|---|
| Model accuracy | 52.6% (baseline 48.0%) |
| Model long/short return, after 0.1% fees | +6.8% |
| Always short | +13.2% |
| Random long/short signals (average of 200 runs) | -19.2% |
| Buy and hold | -18.6% |

The model beat buy and hold and random signals, but not the simple always-short baseline, because Bitcoin fell during the test window. A single 196-day window is a small sample, so I treat this as promising but not proven. The model changed position 87 times, so fees had a large effect.

## Limitations

- One test window only, so results may not hold in other periods
- No funding costs for shorts and no slippage
- Simple price-based features only

## Next steps

- Walk-forward testing across multiple time windows
- Fewer position changes to reduce fee costs

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, requests

## How to Run

```
pip install pandas numpy scikit-learn requests matplotlib
python predict.py
```

Results change slightly between runs because the price data updates daily.

## Notes

This is a learning project, not financial advice or a trading tool.
