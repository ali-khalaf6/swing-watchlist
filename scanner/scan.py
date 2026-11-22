"""Swing Watchlist scanner.

Runs once per US trading day after the close (GitHub Actions). It:
  1. downloads ~10 years of daily prices for the S&P 500 + 16 major ETFs,
  2. backtests both strategies on that history (data/backtest.json),
  3. publishes the next session's picks (data/latest.json, data/history/<date>.json),
  4. grades every pick published so far for the live demo (data/demo.json).

Strategies (long only, liquid large caps, max hold 10 trading days):
  A  "Market-dip RSI(2)"    stock above its 200-day SMA, RSI(2) < 5, market dipping (SPY RSI(2) < 20).
                            Reward:risk 1.5 : 1, stop 0.75 x ATR(14).
  C  "Market-dip stretch"   stock above its 200-day SMA, close >= 1.5 x ATR(14) below its 5-day SMA,
                            SPY above its 200-day SMA and SPY RSI(2) < 20.
                            Reward:risk 2 : 1, stop 0.5 x ATR(14).
Both enter with a next-day limit buy at (signal close - 0.5 x ATR). Not filled that day = cancelled.

Grading is deliberately conservative (daily bars cannot show intraday order):
  * stop and target touched on the same day -> stop
  * a target touched on the fill day is not counted when the fill happened intraday
  * a gap through the stop exits at the open
  * 0.05% slippage added to every entry
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HIST = DATA / "history"

SPY = "SPY"
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 20e6
WARMUP = 200
HOLD = 10
TOP_N = 5
SLIP = 0.0005
MAX_RISK_PCT = 0.15
BACKTEST_FROM = "2017-01-01"
SPLIT_DATE = "2023-01-01"   # designed on data before this, validated on data after

STRATEGIES = {
    "C": {
        "name": "Market-dip stretch",
        "rr": 2.0,
        "stop_atr": 0.5,
        "entry_atr": 0.5,
        "summary": "Stock in a long-term uptrend that closed at least 1.5 ATR below its 5-day average, "
                   "on a day the S&P 500 is above its 200-day average but dipping (SPY RSI(2) < 20).",
    },
    "A": {
        "name": "Market-dip RSI(2)",
        "rr": 1.5,
        "stop_atr": 0.75,
        "entry_atr": 0.5,
        "summary": "Stock in a long-term uptrend with RSI(2) under 5 (deeply oversold for 2 days), "
                   "on a day the whole market is dipping (SPY RSI(2) < 20).",
    },
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- indicators
def sma(a, n):
    r = [None] * len(a)
    s = 0.0
    for i, v in enumerate(a):
        s += v
        if i >= n:
            s -= a[i - n]
        if i >= n - 1:
            r[i] = s / n
    return r


def atr(h, l, c, n=14):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(c))]
    r = [None] * len(c)
    v = None
    for i in range(len(tr)):
        if i == n - 1:
            v = sum(tr[:n]) / n
            r[i] = v
        elif i >= n:
            v = (v * (n - 1) + tr[i]) / n
            r[i] = v
    return r


def rsi(c, n):
    r = [None] * len(c)
    g = lo = 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        up, dn = max(d, 0.0), max(-d, 0.0)
        if i <= n:
            g += up
            lo += dn
            if i == n:
                g /= n
                lo /= n
                r[i] = 100.0 if lo == 0 else 100 - 100 / (1 + g / lo)
        else:
            g = (g * (n - 1) + up) / n
            lo = (lo * (n - 1) + dn) / n
            r[i] = 100.0 if lo == 0 else 100 - 100 / (1 + g / lo)
    return r


def indicators(d):
    c = d["c"]
    return {
        "sma200": sma(c, 200),
        "sma5": sma(c, 5),
        "atr": atr(d["h"], d["l"], c, 14),
        "rsi2": rsi(c, 2),
        "av20": sma(d["v"], 20),
    }


# ---------------------------------------------------------------- data
def load_universe():
    rows = list(csv.DictReader(open(DATA / "universe.csv")))
    return [r["symbol"] for r in rows], {r["symbol"]: r for r in rows}


def download(symbols):
    import yfinance as yf

    out = {}
    for k in range(0, len(symbols), 100):
        batch = symbols[k:k + 100]
        for attempt in range(3):
            try:
                df = yf.download(batch, period="10y", interval="1d", auto_adjust=False, actions=False,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # noqa: BLE001
                log("download retry", attempt, e)
                time.sleep(5)
        else:
            continue
        for s in batch:
            try:
                sub = df[s] if len(batch) > 1 else df
                sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
            except Exception:  # noqa: BLE001
                continue
            if len(sub) < WARMUP + 20:
                continue
            out[s] = {
                "t": [ix.strftime("%Y-%m-%d") for ix in sub.index],
                "o": [float(x) for x in sub["Open"]],
                "h": [float(x) for x in sub["High"]],
                "l": [float(x) for x in sub["Low"]],
                "c": [float(x) for x in sub["Close"]],
                "v": [0.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in sub["Volume"]],
            }
        log(f"downloaded {min(k + 100, len(symbols))}/{len(symbols)}  ok={len(out)}")
    return out


# ---------------------------------------------------------------- signals
def market_regime(D, I):
    d, x = D[SPY], I[SPY]
    reg = {}
    for i, t in enumerate(d["t"]):
        reg[t] = {
            "up200": x["sma200"][i] is not None and d["c"][i] > x["sma200"][i],
            "rsi2": x["rsi2"][i],
            "close": d["c"][i],
            "sma200": x["sma200"][i],
        }
    return reg


def regime_ok(key, r):
    if not r or r["rsi2"] is None:
        return False
    if key == "A":
        return r["rsi2"] < 20
    if key == "C":
        return r["up200"] and r["rsi2"] < 20
    return False


def signal(key, i, d, x):
    """Return a ranking score (higher = better) or None."""
    s200 = x["sma200"][i]
    if s200 is None or d["c"][i] <= s200:
        return None
    if key == "A":
        r = x["rsi2"][i]
        return -r if (r is not None and r < 5) else None
    if key == "C":
        if x["sma5"][i] is None or not x["atr"][i]:
            return None
        z = (x["sma5"][i] - d["c"][i]) / x["atr"][i]
        return z if z > 1.5 else None
    return None


def liquid(i, d, x):
    return d["c"][i] >= MIN_PRICE and x["av20"][i] is not None and x["av20"][i] * d["c"][i] >= MIN_DOLLAR_VOL


def candidates(key, D, I, reg, date_from, date_to):
    out = {}
    for s, d in D.items():
        x = I[s]
        for i in range(WARMUP, len(d["t"]) - 1):
            t = d["t"][i]
            if t < date_from or t > date_to:
                continue
            if not regime_ok(key, reg.get(t)):
                continue
            if not liquid(i, d, x):
                continue
            sc = signal(key, i, d, x)
            if sc is None:
                continue
            out.setdefault(t, []).append((sc, s, i))
    return out


# ---------------------------------------------------------------- trade simulation
def simulate(key, s, i, D, I, plan=None):
    """Simulate one trade signalled on bar i. Returns a dict (status in
    PENDING / NOT_FILLED / OPEN / TARGET / STOP / TIME)."""
    st_cfg = STRATEGIES[key]
    d, x = D[s], I[s]
    a = x["atr"][i]
    limit = plan["entry"] if plan else d["c"][i] - st_cfg["entry_atr"] * a
    risk_abs = plan["risk"] if plan else st_cfg["stop_atr"] * a
    j = i + 1
    if j >= len(d["t"]):
        return {"status": "PENDING"}
    if d["l"][j] > limit:
        return {"status": "NOT_FILLED", "entry_date": d["t"][j]}
    intraday = d["o"][j] > limit
    e = min(d["o"][j], limit) * (1 + SLIP)
    risk = risk_abs
    if not risk > 0 or risk / e > MAX_RISK_PCT:
        return {"status": "SKIPPED"}
    stop = e - risk
    target = e + st_cfg["rr"] * risk
    res, ex, k = "TIME", None, j
    last = min(len(d["t"]), j + HOLD)
    for k in range(j, last):
        hit_stop = d["l"][k] <= stop
        hit_tgt = d["h"][k] >= target and not (k == j and intraday)
        if hit_stop:
            res, ex = "STOP", min(stop, d["o"][k])
            break
        if hit_tgt:
            res, ex = "TARGET", target
            break
    if res == "TIME":
        k = min(len(d["t"]) - 1, j + HOLD - 1)
        ex = d["c"][k]
        if j + HOLD - 1 >= len(d["t"]):
            res = "OPEN"
    return {
        "status": res, "entry_date": d["t"][j], "fill": e, "stop": stop, "target": target,
        "exit_date": d["t"][k], "exit": ex, "R": (ex - e) / risk, "ret": ex / e - 1,
        "days_held": k - j + 1,
    }


def backtest(key, D, I, reg, date_from, date_to):
    cands = candidates(key, D, I, reg, date_from, date_to)
    busy = {}
    trades = []
    for t in sorted(cands):
        taken = 0
        for sc, s, i in sorted(cands[t], key=lambda z: (-z[0], z[1])):
            if taken >= TOP_N:
                break
            if busy.get(s) and busy[s] >= t:
                continue
            r = simulate(key, s, i, D, I)
            if r["status"] in ("NOT_FILLED", "SKIPPED", "PENDING"):
                continue
            r.update({"symbol": s, "signal_date": t})
            trades.append(r)
            busy[s] = r["exit_date"]
            taken += 1
    return trades


def stats(trades):
    c = [t for t in trades if t["status"] != "OPEN"]
    n = len(c)
    if not n:
        return {"n": 0}
    w = sum(t["status"] == "TARGET" for t in c)
    lo = sum(t["status"] == "STOP" for t in c)
    tm = sum(t["status"] == "TIME" for t in c)
    gw = sum(t["R"] for t in c if t["R"] > 0)
    gl = -sum(t["R"] for t in c if t["R"] < 0)
    return {
        "n": n, "wins": w, "losses": lo, "time_exits": tm,
        "win_rate": w / n, "loss_rate": lo / n, "time_rate": tm / n,
        "avg_R": sum(t["R"] for t in c) / n,
        "avg_ret": sum(t["ret"] for t in c) / n,
        "profit_factor": (gw / gl) if gl else None,
        "avg_days": sum(t["days_held"] for t in c) / n,
    }


# ---------------------------------------------------------------- live picks + demo
def todays_picks(key, D, I, reg, signal_date, meta, active):
    cfg = STRATEGIES[key]
    if not regime_ok(key, reg.get(signal_date)):
        return []
    rows = []
    for s, d in D.items():
        if not d["t"] or d["t"][-1] != signal_date:
            continue
        i = len(d["t"]) - 1
        x = I[s]
        if i < WARMUP or not liquid(i, d, x):
            continue
        sc = signal(key, i, d, x)
        if sc is None or s in active:
            continue
        rows.append((sc, s, i))
    picks = []
    for sc, s, i in sorted(rows, key=lambda z: (-z[0], z[1]))[:TOP_N]:
        d, x = D[s], I[s]
        a = x["atr"][i]
        entry = d["c"][i] - cfg["entry_atr"] * a
        risk = cfg["stop_atr"] * a
        if risk / entry > MAX_RISK_PCT:
            continue
        picks.append({
            "symbol": s,
            "name": meta.get(s, {}).get("name", s),
            "sector": meta.get(s, {}).get("sector", ""),
            "close": round(d["c"][i], 2),
            "entry": round(entry, 2),
            "stop": round(entry - risk, 2),
            "target": round(entry + cfg["rr"] * risk, 2),
            "risk": round(risk, 4),
            "risk_pct": round(risk / entry * 100, 2),
            "reward_pct": round(cfg["rr"] * risk / entry * 100, 2),
            "rsi2": round(x["rsi2"][i], 1),
            "below_sma5_atr": round((x["sma5"][i] - d["c"][i]) / a, 2),
            "score": round(sc, 3),
        })
    return picks


def grade_demo(demo, D, I):
    for p in demo["picks"]:
        s = p["symbol"]
        d = D.get(s)
        if not d or p["signal_date"] not in d["t"]:
            continue
        i = d["t"].index(p["signal_date"])
        r = simulate(p["strategy"], s, i, D, I, plan={"entry": p["entry"], "risk": p["risk"]})
        p["result"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        p["last_price"] = round(d["c"][-1], 2)
    return demo


# ---------------------------------------------------------------- main
def main():
    DATA.mkdir(exist_ok=True)
    HIST.mkdir(exist_ok=True)
    symbols, meta = load_universe()
    D = download(symbols)
    if SPY not in D:
        log("SPY missing; aborting")
        sys.exit(1)
    I = {s: indicators(d) for s, d in D.items()}
    reg = market_regime(D, I)
    signal_date = D[SPY]["t"][-1]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # demo history
    demo_path = DATA / "demo.json"
    demo = json.loads(demo_path.read_text()) if demo_path.exists() else {"started": signal_date, "picks": []}
    demo = grade_demo(demo, D, I)
    active = {}
    for p in demo["picks"]:
        st = (p.get("result") or {}).get("status")
        if st in (None, "PENDING", "OPEN"):
            active.setdefault(p["strategy"], set()).add(p["symbol"])

    # backtests
    bt = {"generated_at": now, "from": BACKTEST_FROM, "to": signal_date, "split": SPLIT_DATE,
          "universe_size": len(D), "strategies": {}}
    for key, cfg in STRATEGIES.items():
        tr = backtest(key, D, I, reg, BACKTEST_FROM, signal_date)
        years = {}
        for t in tr:
            years.setdefault(t["signal_date"][:4], []).append(t)
        bt["strategies"][key] = {
            **{k: v for k, v in cfg.items()},
            "all": stats(tr),
            "design": stats([t for t in tr if t["signal_date"] < SPLIT_DATE]),
            "validation": stats([t for t in tr if t["signal_date"] >= SPLIT_DATE]),
            "by_year": {y: stats(v) for y, v in sorted(years.items())},
            "recent": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in tr[-40:]][::-1],
        }
        a = bt["strategies"][key]["all"]
        log(f"backtest {key}: n={a.get('n')} win={a.get('win_rate', 0):.3f} pf={a.get('profit_factor')}")
    (DATA / "backtest.json").write_text(json.dumps(bt, indent=1))

    # today's picks
    r = reg[signal_date]
    latest = {
        "generated_at": now,
        "signal_date": signal_date,
        "market": {"spy_close": round(r["close"], 2), "spy_sma200": round(r["sma200"], 2) if r["sma200"] else None,
                   "spy_rsi2": round(r["rsi2"], 1) if r["rsi2"] is not None else None, "spy_up200": r["up200"]},
        "strategies": {},
    }
    for key, cfg in STRATEGIES.items():
        picks = todays_picks(key, D, I, reg, signal_date, meta, active.get(key, set()))
        latest["strategies"][key] = {"name": cfg["name"], "rr": cfg["rr"], "market_ok": regime_ok(key, r),
                                     "picks": picks}
        known = {(p["strategy"], p["symbol"], p["signal_date"]) for p in demo["picks"]}
        for p in picks:
            if (key, p["symbol"], signal_date) not in known:
                demo["picks"].append({"strategy": key, "signal_date": signal_date, **p, "result": {"status": "PENDING"}})
    (DATA / "latest.json").write_text(json.dumps(latest, indent=1))
    (HIST / f"{signal_date}.json").write_text(json.dumps(latest, indent=1))
    demo["updated"] = now
    demo_path.write_text(json.dumps(demo, indent=1))
    log("picks:", {k: [p["symbol"] for p in v["picks"]] for k, v in latest["strategies"].items()})


if __name__ == "__main__":
    main()
