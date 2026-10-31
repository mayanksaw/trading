#!/usr/bin/env python3
"""
Trading Plan Assistant - works for stocks, indices, crypto, forex, commodities.

It reads real price data, finds trend / support / resistance, and builds a TRADE PLAN:
entry, stop-loss, targets, and how many units to buy based on YOUR capital and risk %.
It can also backtest the same rules on past data, and (optionally) explain the plan
in simple words using Groq.

This is a planning tool, NOT financial advice. No tool can predict the market.

Install:   pip install yfinance pandas numpy requests
Examples:  python trading_assistant.py                      (chat mode)
           python trading_assistant.py reliance --tf 1d --capital 100000 --risk 1
           python trading_assistant.py banknifty --tf 15m --capital 50000 --backtest
           python trading_assistant.py btc --tf 1h --capital 1000 --ai
Symbols:   nifty, banknifty, sensex, btc, eth, eurusd, gold, silver, crude, or any
           Yahoo symbol (AAPL, TCS.NS, ^GSPC). Plain names like "reliance" get .NS added.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"   # change if Groq renames it

ALIASES = {
    "nifty": "^NSEI", "banknifty": "^NSEBANK", "sensex": "^BSESN",
    "btc": "BTC-USD", "bitcoin": "BTC-USD", "eth": "ETH-USD", "ethereum": "ETH-USD",
    "sol": "SOL-USD", "eurusd": "EURUSD=X", "gbpusd": "GBPUSD=X", "usdjpy": "USDJPY=X",
    "usdinr": "USDINR=X", "gold": "GC=F", "silver": "SI=F", "crude": "CL=F",
    "sp500": "^GSPC", "nasdaq": "^IXIC",
}

# timeframe -> (yfinance interval, history period)
TIMEFRAMES = {
    "5m": ("5m", "30d"), "15m": ("15m", "60d"), "30m": ("30m", "60d"),
    "1h": ("60m", "6mo"), "1d": ("1d", "2y"), "1wk": ("1wk", "5y"),
}


# ------------------------------------------------------------------ data
def resolve_symbol(s):
    s = s.strip()
    low = s.lower()
    if low in ALIASES:
        return ALIASES[low]
    if any(c in s for c in ".^=-"):
        return s.upper()
    return s.upper() + ".NS"          # assume Indian stock


def asset_type(sym):
    if sym.endswith("-USD"):
        return "crypto"
    if sym.endswith("=X"):
        return "forex"
    if sym.startswith("^"):
        return "index"
    if sym.endswith("=F"):
        return "commodity"
    return "stock"


def get_data(sym, tf):
    import yfinance as yf
    interval, period = TIMEFRAMES[tf]
    df = yf.download(sym, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ indicators
def add_indicators(df):
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def pivots(df, i, lookback=120, w=5):
    """Swing highs / lows using only data up to bar i."""
    start = max(0, i - lookback)
    hi = df["high"].values[start:i + 1]
    lo = df["low"].values[start:i + 1]
    highs, lows = [], []
    for k in range(w, len(hi) - w):
        if hi[k] == hi[k - w:k + w + 1].max():
            highs.append(hi[k])
        if lo[k] == lo[k - w:k + w + 1].min():
            lows.append(lo[k])
    return sorted(set(lows)), sorted(set(highs))


# ------------------------------------------------------------------ the plan
def make_plan(df, i, min_rr=1.5):
    """Build a trade idea using only data up to bar i. Returns a dict."""
    r = df.iloc[i]
    price, atr = float(r["close"]), float(r["atr"])
    if not np.isfinite(atr) or atr <= 0 or i < 60:
        return {"direction": "WAIT", "reasons": ["Not enough data yet"]}

    score, why = 0, []
    if price > r["ema50"]:
        score += 1; why.append("price above EMA50 (uptrend bias)")
    else:
        score -= 1; why.append("price below EMA50 (downtrend bias)")
    if r["ema20"] > r["ema50"]:
        score += 1; why.append("EMA20 above EMA50")
    else:
        score -= 1; why.append("EMA20 below EMA50")
    if r["macd_hist"] > 0:
        score += 1; why.append("MACD momentum positive")
    else:
        score -= 1; why.append("MACD momentum negative")
    rsi = float(r["rsi"])
    if rsi > 55:
        score += 1; why.append("RSI %.0f (strength)" % rsi)
    elif rsi < 45:
        score -= 1; why.append("RSI %.0f (weakness)" % rsi)
    else:
        why.append("RSI %.0f (neutral)" % rsi)

    direction = "LONG" if score >= 3 else "SHORT" if score <= -3 else "WAIT"
    if direction == "LONG" and rsi > 75:
        direction = "WAIT"; why.append("but RSI is overbought - chasing is risky")
    if direction == "SHORT" and rsi < 25:
        direction = "WAIT"; why.append("but RSI is oversold - chasing is risky")

    lows, highs = pivots(df, i)
    supports = [x for x in lows if x < price][-2:]
    resists = [x for x in highs if x > price][:2]
    plan = {"direction": direction, "score": score, "reasons": why, "price": price,
            "atr": atr, "rsi": rsi, "supports": supports, "resists": resists}
    if direction == "WAIT":
        return plan

    # entry: market if price is near EMA20, otherwise wait for a pullback to EMA20
    ema20 = float(r["ema20"])
    if abs(price - ema20) <= atr:
        entry, mode = price, "near current price"
    else:
        entry, mode = ema20, "limit order at EMA20 pullback (price is stretched)"

    sign = 1 if direction == "LONG" else -1
    recent = df.iloc[max(0, i - 9):i + 1]
    if direction == "LONG":
        stop = float(recent["low"].min()) - 0.5 * atr
    else:
        stop = float(recent["high"].max()) + 0.5 * atr
    risk = abs(entry - stop)
    if not (0.5 * atr <= risk <= 3 * atr):          # stop too tight / too wide -> use ATR stop
        risk = 1.5 * atr
        stop = entry - sign * risk

    t1, t2 = entry + sign * 1.5 * risk, entry + sign * 2.5 * risk
    plan.update({"entry": entry, "entry_mode": mode, "stop": stop, "risk": risk,
                 "t1": t1, "t2": t2})
    # if a strong level blocks target 1, reward:risk may be poor
    blocker = [x for x in (resists if direction == "LONG" else supports)]
    if blocker:
        nearest = blocker[0] if direction == "LONG" else blocker[-1]
        rr_to_level = abs(nearest - entry) / risk
        plan["rr_to_level"] = rr_to_level
        if rr_to_level < min_rr:
            plan["direction"] = "WAIT"
            plan["reasons"].append("nearest level is only %.1fR away - poor reward:risk" % rr_to_level)
    return plan


def position_size(plan, capital, risk_pct, kind, leverage=1.0):
    risk_amount = capital * risk_pct / 100.0
    qty = risk_amount / plan["risk"]
    max_qty = capital * leverage / plan["entry"]
    capped = qty > max_qty
    qty = min(qty, max_qty)
    qty = math.floor(qty) if kind in ("stock",) else round(qty, 4)
    value = qty * plan["entry"]
    actual_risk = qty * plan["risk"]
    return {"risk_amount": risk_amount, "qty": qty, "value": value,
            "actual_risk": actual_risk, "capped": capped}


# ------------------------------------------------------------------ backtest
def backtest(df, hold=20, rr=1.5):
    results = []
    i = 60
    while i < len(df) - hold - 1:
        p = make_plan(df, i)
        if p["direction"] == "WAIT":
            i += 1
            continue
        sign = 1 if p["direction"] == "LONG" else -1
        entry = float(df["open"].iloc[i + 1])
        risk = p["risk"]
        stop, target = entry - sign * risk, entry + sign * rr * risk
        out = None
        for j in range(i + 1, i + 1 + hold):
            lo, hi = float(df["low"].iloc[j]), float(df["high"].iloc[j])
            hit_stop = lo <= stop if sign == 1 else hi >= stop
            hit_tgt = hi >= target if sign == 1 else lo <= target
            if hit_stop:                 # stop checked first = pessimistic
                out = (-1.0, j); break
            if hit_tgt:
                out = (rr, j); break
        if out is None:
            j = i + hold
            out = (sign * (float(df["close"].iloc[j]) - entry) / risk, j)
        results.append(out[0])
        i = out[1] + 1
    return results


def backtest_report(results, rr=1.5):
    if not results:
        return "Backtest: no trades were generated on this data."
    r = np.array(results)
    wins = (r > 0).mean() * 100
    eq = r.cumsum()
    dd = (np.maximum.accumulate(eq) - eq).max()
    return ("BACKTEST (same rules on past data, target %.1fR, no fees/slippage)\n"
            "  trades: %d | win rate: %.0f%% | average: %+.2fR | total: %+.1fR | worst drawdown: %.1fR\n"
            "  (1R = the amount you risk per trade. Positive total is needed before trusting the rules.)"
            % (rr, len(r), wins, r.mean(), r.sum(), dd))


# ------------------------------------------------------------------ output
def fmt(v, price):
    return ("%.5f" % v) if price < 10 else ("%.2f" % v)


def build_text(sym, tf, kind, plan, capital, risk_pct, leverage):
    p = plan["price"] if "price" in plan else 0
    lines = ["=" * 60,
             "%s  [%s, %s]   price: %s" % (sym, kind, tf, fmt(p, p)),
             "=" * 60,
             "Signal: %s   (score %+d of 4)" % (plan["direction"], plan.get("score", 0)),
             "Why: " + "; ".join(plan["reasons"])]
    if plan.get("supports"):
        lines.append("Support levels:    " + ", ".join(fmt(x, p) for x in plan["supports"]))
    if plan.get("resists"):
        lines.append("Resistance levels: " + ", ".join(fmt(x, p) for x in plan["resists"]))
    if plan["direction"] == "WAIT":
        lines.append("\nNo clean setup right now. Waiting is also a position - do not force a trade.")
        return "\n".join(lines)

    ps = position_size(plan, capital, risk_pct, kind, leverage)
    lines += ["",
              "TRADE PLAN (%s)" % plan["direction"],
              "  Entry:      %s  - %s" % (fmt(plan["entry"], p), plan["entry_mode"]),
              "  Stop-loss:  %s  (idea is wrong if price reaches this)" % fmt(plan["stop"], p),
              "  Target 1:   %s  (1.5R)" % fmt(plan["t1"], p),
              "  Target 2:   %s  (2.5R)" % fmt(plan["t2"], p),
              "",
              "CAPITAL / SIZE   (capital %.0f, risk %.1f%% per trade)" % (capital, risk_pct),
              "  Money you risk on this trade: %.2f" % ps["risk_amount"],
              "  Quantity: %s   | position value: %.2f" % (ps["qty"], ps["value"]),
              "  Actual loss if stop-loss hits: %.2f" % ps["actual_risk"]]
    if ps["capped"]:
        lines.append("  NOTE: quantity was limited by your capital (leverage %.1fx). "
                     "Real risk is lower than %.1f%%." % (leverage, risk_pct))
    if ps["qty"] <= 0:
        lines.append("  NOTE: capital too small for even 1 unit at this stop distance.")
    lines += ["",
              "Rules: move stop to entry after Target 1, never widen the stop, skip the trade",
              "if the entry is missed. Not financial advice."]
    return "\n".join(lines)


def ai_explain(text):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("\n(Set GROQ_API_KEY to get a simple-language AI explanation.)")
        return
    try:
        import requests
        resp = requests.post(
            GROQ_URL, timeout=40,
            headers={"Authorization": "Bearer " + key},
            json={"model": GROQ_MODEL, "temperature": 0.2, "messages": [
                {"role": "system", "content":
                    "Explain this trading plan in simple English with a few easy Hindi words. "
                    "Use ONLY the numbers given, never invent levels, never promise profit, "
                    "and remind the user about risk. Keep it under 150 words."},
                {"role": "user", "content": text}]})
        print("\nAI explanation:\n" + resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print("\nAI explanation failed:", e)


# ------------------------------------------------------------------ main
def analyse(symbol, tf, capital, risk_pct, leverage, do_backtest, do_ai):
    sym = resolve_symbol(symbol)
    kind = asset_type(sym)
    df = get_data(sym, tf)
    if df is None or len(df) < 80:
        print("Could not get enough data for %s (%s). Check the symbol / internet." % (sym, tf))
        return
    df = add_indicators(df)
    plan = make_plan(df, len(df) - 1)
    text = build_text(sym, tf, kind, plan, capital, risk_pct, leverage)
    print("\n" + text)
    if kind in ("crypto", "forex", "commodity") or sym in ("^GSPC", "^IXIC"):
        print("\nNote: prices for this symbol are in USD - enter your capital in USD too.")
    if kind == "index":
        print("\nNote: an index is not directly tradable. Use futures/options/ETF; "
              "convert the quantity using the contract lot size.")
    if do_backtest:
        print("\n" + backtest_report(backtest(df)))
    if do_ai:
        ai_explain(text)


def chat():
    print("Trading Plan Assistant (type q to quit)")
    while True:
        s = input("\nSymbol (e.g. reliance, nifty, btc, eurusd, gold): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        tf = input("Timeframe (5m/15m/30m/1h/1d/1wk) [15m]: ").strip() or "15m"
        if tf not in TIMEFRAMES:
            print("Unknown timeframe."); continue
        try:
            capital = float(input("Your trading capital: "))
            risk_pct = float(input("Risk per trade in %% [1]: ") or 1)
        except ValueError:
            print("Please enter numbers."); continue
        bt = input("Backtest the rules too? (y/n) [y]: ").strip().lower() != "n"
        ai = input("AI explanation? (y/n) [n]: ").strip().lower() == "y"
        analyse(s, tf, capital, risk_pct, 1.0, bt, ai)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trading plan assistant")
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("--tf", default="15m", choices=list(TIMEFRAMES))
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--risk", type=float, default=1.0, help="percent of capital risked per trade")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--ai", action="store_true")
    a = ap.parse_args()
    if not a.symbol:
        chat()
    else:
        analyse(a.symbol, a.tf, a.capital, a.risk, a.leverage, a.backtest, a.ai)
