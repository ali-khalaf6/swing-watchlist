# Swing Watchlist

Daily swing-trade watchlist for S&P 500 stocks and major ETFs, published with GitHub Pages.

Two long-only, buy-the-dip strategies run after every US close:

| | Strategy | Reward : risk | Stop | Entry |
|---|---|---|---|---|
| **C** | Market-dip stretch: stock above its 200-day SMA closes at least 1.5 ATR below its 5-day SMA, while SPY is above its 200-day SMA with RSI(2) under 20 | 2 : 1 | 0.5 × ATR | limit at close − 0.5 × ATR |
| **A** | Market-dip RSI(2): stock above its 200-day SMA with RSI(2) under 5, while SPY RSI(2) is under 20 | 1.5 : 1 | 0.75 × ATR | limit at close − 0.5 × ATR |

Each trade is held for at most 10 trading days. Up to 5 picks per strategy per day.

* `scanner/scan.py` downloads prices, backtests both strategies, publishes the next session's picks and grades the live demo.
* `.github/workflows/scan.yml` runs the scanner Mon–Fri at 21:35 UTC and commits `data/`.
* `index.html` is the website, which reads `data/latest.json`, `data/demo.json` and `data/backtest.json`.

To run it by hand, go to **Actions → Daily scan → Run workflow**.

This is a mechanical screen, not financial advice.
