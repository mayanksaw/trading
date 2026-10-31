"""Paper trader (FAKE money) for the Trading Plan Assistant.
 
It runs in the background on the server. Every few seconds/minutes it checks your symbols
(crypto, Indian stocks, indices, forex, commodities - anything Yahoo has), takes the same
BUY / SELL signals as the Live monitor, and "trades" them with virtual money:
 
  - opens a trade at the entry, with stop-loss and 2 targets
  - at Target 1 it books half and moves the stop to entry (the app's own rule)
  - closes everything at Target 2 or at the stop
  - keeps a history, win rate and total result in R (1R = money risked per trade)
 
Safety rules: max 3 open trades, 1 trade per symbol, stops opening new trades for the
day after -3R, 0.05% cost per side (fees + slippage) is charged on every trade.
 
Open the page at  /paper . No real money and no broker is touched.
Needs ONE gunicorn worker (see render.yaml): gunicorn app:app --workers 1 --threads 4
"""
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
 
from flask import jsonify, request
 
STATE_FILE = os.getenv("PAPER_STATE_FILE", "paper_state.json")
FEE_PCT = 0.05            # cost per side, percent of trade value (fees + slippage)
MAX_OPEN = 3              # open + waiting orders at the same time
DAILY_LOSS_R = 3.0        # stop opening new trades for the day after this loss (in R)
PENDING_MAX_AGE = 3600    # seconds a limit order may wait before it is cancelled
MAX_SYMBOLS = 8
 
_lock = threading.RLock()
_stop = threading.Event()
_thread = None
_live = None
_recent = {}              # symbol -> last prices (to detect a closed market)
STATE = None
 
 
# ------------------------------------------------------------------ state
def _fresh():
    return {"enabled": False, "config": {}, "open": [], "pending": [], "closed": [],
            "events": [], "heartbeat": 0.0, "day": "", "day_r": 0.0,
            "paused_today": False, "next_id": 1, "start_capital": 0.0}
 
 
def _load():
    try:
        with open(STATE_FILE) as fh:
            data = json.load(fh)
        s = _fresh()
        s.update(data)
        return s
    except Exception:
        return _fresh()
 
 
def _save():
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(STATE, fh)
        os.replace(tmp, STATE_FILE)
    except Exception:
        pass
 
 
def _f(v):
    return ("%.5f" % v) if abs(v) < 10 else ("%.2f" % v)
 
 
def _event(text):
    ev = STATE["events"]
    if ev and ev[0]["text"] == text:
        return
    ev.insert(0, {"t": datetime.now(timezone.utc).strftime("%d %b %H:%M UTC"), "text": text})
    del ev[60:]
 
 
def _roll_day():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if STATE["day"] != today:
        STATE["day"] = today
        STATE["day_r"] = 0.0
        STATE["paused_today"] = False
 
 
# ------------------------------------------------------------------ trade logic
def _pnl(side, entry, exit_, qty):
    gross = (exit_ - entry) * qty if side == "LONG" else (entry - exit_) * qty
    fees = (entry + exit_) * qty * FEE_PCT / 100.0
    return gross - fees
 
 
def _rm(t):
    """Money value of 1R for this trade, in the user's own capital currency."""
    rm = t.get("risk_money")
    if rm:
        return rm
    cfg = STATE["config"]
    return cfg.get("capital", 0) * cfg.get("risk", 0) / 100.0
 
 
def _now_r(t):
    """Open result in R (booked part + what is still open)."""
    risk = t["risk_amount"] or 1.0
    return (t["pnl"] + _pnl(t["side"], t["entry"], t["last"], t["qty_left"])) / risk
 
 
def _book(t, qty, price):
    t["pnl"] += _pnl(t["side"], t["entry"], price, qty)
    t["qty_left"] = round(t["qty_left"] - qty, 6)
 
 
def _finish(t, reason):
    if t in STATE["open"]:
        STATE["open"].remove(t)
    r = t["pnl"] / t["risk_amount"] if t["risk_amount"] else 0.0
    rec = dict(t)
    rec.update(reason=reason, r=round(r, 3), money=round(r * _rm(t), 2), closed=time.time())
    STATE["closed"].append(rec)
    del STATE["closed"][:-500]
    STATE["day_r"] += r
    _event("CLOSE %s %s: %s, result %+.2fR" % (t["side"], t["symbol"], reason, r))
    if STATE["day_r"] <= -DAILY_LOSS_R and not STATE["paused_today"]:
        STATE["paused_today"] = True
        _event("Daily loss limit (-%.0fR) reached - no new trades until tomorrow (UTC)." % DAILY_LOSS_R)
 
 
def _new_trade(d, side, entry):
    qty = d["qty"]
    cfg = STATE["config"]
    t = {"id": STATE["next_id"], "risk_money": cfg.get("capital", 0) * cfg.get("risk", 0) / 100.0, "symbol": d["symbol"], "tf": d["tf"], "side": side,
         "entry": entry, "stop": d["stop"], "stop0": d["stop"], "t1": d["t1"], "t2": d["t2"],
         "qty": qty, "qty_left": qty, "pnl": 0.0, "be": False,
         "risk_amount": abs(entry - d["stop"]) * qty, "last": entry,
         "placed": time.time(), "opened": None}
    STATE["next_id"] += 1
    return t
 
 
def _manage_open(t, price):
    sign = 1 if t["side"] == "LONG" else -1
    t["last"] = price
    if sign * (price - t["stop"]) <= 0:                       # stop-loss (or break-even stop)
        _book(t, t["qty_left"], price)
        _finish(t, "break-even stop" if t["be"] else "stop-loss")
        return
    if not t["be"] and sign * (price - t["t1"]) >= 0:         # Target 1
        q = t["qty_left"]
        half = q // 2 if float(q).is_integer() else round(q / 2, 4)
        if half <= 0 or half >= q:
            _book(t, q, t["t1"])
            _finish(t, "target 1")
            return
        _book(t, half, t["t1"])
        t["be"] = True
        t["stop"] = t["entry"]
        _event("%s %s: Target 1 hit, half booked, stop moved to entry" % (t["side"], t["symbol"]))
    if t["be"] and sign * (price - t["t2"]) >= 0:             # Target 2
        _book(t, t["qty_left"], t["t2"])
        _finish(t, "target 2")
 
 
def _manage_pending(t, price):
    sign = 1 if t["side"] == "LONG" else -1
    if time.time() - t["placed"] > PENDING_MAX_AGE:
        STATE["pending"].remove(t)
        _event("Cancelled %s %s limit order (waited too long)" % (t["side"], t["symbol"]))
    elif sign * (price - t["stop"]) <= 0 or sign * (price - t["t1"]) >= 0:
        STATE["pending"].remove(t)
        _event("Cancelled %s %s limit order (idea no longer valid)" % (t["side"], t["symbol"]))
    elif sign * (price - t["entry"]) <= 0:                    # limit order filled
        STATE["pending"].remove(t)
        t["opened"] = time.time()
        STATE["open"].append(t)
        _event("FILLED %s %s at %s | stop %s | T1 %s" % (t["side"], t["symbol"], _f(t["entry"]),
                                                          _f(t["stop"]), _f(t["t1"])))
 
 
def _handle(d):
    """Apply one fresh signal to the paper account."""
    sym, price = d["symbol"], d["price"]
    hist = _recent.setdefault(sym, deque(maxlen=4))
    hist.append(price)
    stale = len(hist) == 4 and len(set(hist)) == 1            # price frozen = market closed
    for t in [x for x in STATE["pending"] if x["symbol"] == sym]:
        _manage_pending(t, price)
    for t in [x for x in STATE["open"] if x["symbol"] == sym]:
        _manage_open(t, price)
    if stale or STATE["paused_today"]:
        return
    decision = d.get("decision", "WAIT")
    if decision == "WAIT" or not d.get("qty") or d["qty"] <= 0:
        return
    if any(x["symbol"] == sym for x in STATE["open"] + STATE["pending"]):
        return
    if len(STATE["open"]) + len(STATE["pending"]) >= MAX_OPEN:
        return
    side = "LONG" if decision.startswith("BUY") else "SHORT"
    sign = 1 if side == "LONG" else -1
    if decision.endswith("NOW"):
        if not (sign * (price - d["stop"]) > 0 and sign * (d["t1"] - price) > 0):
            return
        t = _new_trade(d, side, price)
        t["opened"] = time.time()
        STATE["open"].append(t)
        _event("OPEN %s %s at %s | stop %s | T1 %s | qty %s" % (
            side, sym, _f(price), _f(t["stop"]), _f(t["t1"]), t["qty"]))
    else:                                                      # LIMIT order: wait for the entry price
        t = _new_trade(d, side, d["entry"])
        STATE["pending"].append(t)
        _event("ORDER %s %s limit at %s | stop %s | T1 %s | qty %s" % (
            side, sym, _f(t["entry"]), _f(t["stop"]), _f(t["t1"]), t["qty"]))
 
 
# ------------------------------------------------------------------ background worker
def _worker():
    while not _stop.is_set():
        with _lock:
            if not STATE["enabled"]:
                return
            cfg = dict(STATE["config"])
        for sym in cfg.get("symbols", []):
            if _stop.is_set():
                return
            d, err = None, ""
            try:
                d = _live(sym, cfg["tf"], cfg["capital"], cfg["risk"], cfg["leverage"])
            except Exception as e:
                err = str(e)[:100]
            with _lock:
                if not STATE["enabled"]:
                    return
                _roll_day()
                if d is None:
                    _event("%s: could not get data (%s)" % (sym.upper(), err))
                else:
                    _handle(d)
                STATE["heartbeat"] = time.time()
                _save()
        _stop.wait(cfg.get("interval", 60))
 
 
def _start_thread():
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_worker, daemon=True)
        _thread.start()
 
 
# ------------------------------------------------------------------ views
def _stats():
    rs = [c["r"] for c in STATE["closed"]]
    eq = peak = dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    n = len(rs)
    wins = sum(1 for r in rs if r > 0)
    start = STATE.get("start_capital") or STATE["config"].get("capital", 0)
    realized = sum(c.get("money", 0.0) for c in STATE["closed"])
    floating = sum(_now_r(t) * _rm(t) for t in STATE["open"])
    day_money = STATE["day_r"] * (STATE["config"].get("capital", 0) * STATE["config"].get("risk", 0) / 100.0)
    return {"start": round(start, 2), "balance": round(start + realized + floating, 2),
            "profit": round(realized + floating, 2), "realized": round(realized, 2),
            "floating": round(floating, 2), "day_money": round(day_money, 2), "trades": n, "wins": wins, "win_rate": round(100.0 * wins / n, 1) if n else 0.0,
            "total_r": round(sum(rs), 2), "avg_r": round(sum(rs) / n, 3) if n else 0.0,
            "max_dd_r": round(dd, 2), "day_r": round(STATE["day_r"], 2),
            "paused": STATE["paused_today"]}
 
 
def _view(t):
    now_r = round(_now_r(t), 2) if t.get("opened") else None
    return {"id": t["id"], "symbol": t["symbol"], "tf": t["tf"], "side": t["side"],
            "entry": t["entry"], "stop": t["stop"], "t1": t["t1"], "t2": t["t2"],
            "qty": t["qty"], "last": t["last"], "be": t["be"], "now_r": now_r,
            "reason": t.get("reason"), "r": t.get("r"),
            "money": t.get("money") if t.get("reason") else (round(now_r * _rm(t), 2) if now_r is not None else None),
            "risk_money": round(_rm(t), 2)}
 
 
# ------------------------------------------------------------------ routes
def register(app, authorized, live_signal, parse_money, ta):
    global _live, STATE
    _live = live_signal
    STATE = _load()
 
    def guard(d):
        return authorized(d.get("password"))
 
    @app.get("/paper")
    def paper_page():
        return PAGE
 
    @app.post("/api/paper/start")
    def paper_start():
        d = request.get_json(silent=True) or {}
        if not guard(d):
            return jsonify(error="Wrong password"), 401
        syms = list(dict.fromkeys(x.strip() for x in str(d.get("symbols", "")).split(",") if x.strip()))
        if not syms or len(syms) > MAX_SYMBOLS or any(len(x) > 20 for x in syms):
            return jsonify(error="Enter 1 to %d symbols, separated by commas." % MAX_SYMBOLS), 400
        tf = d.get("tf", "15m")
        if tf not in ta.TIMEFRAMES:
            return jsonify(error="Unknown timeframe."), 400
        try:
            cap, risk, lev = parse_money(d.get("capital", 100000), d.get("risk", 1), d.get("leverage", 1))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        try:
            interval = max(30, min(int(d.get("interval", 60)), 900))
        except (TypeError, ValueError):
            interval = 60
        with _lock:
            STATE["config"] = {"symbols": syms, "tf": tf, "capital": cap, "risk": risk,
                               "leverage": lev, "interval": interval}
            STATE["enabled"] = True
            if not STATE.get("start_capital"):
                STATE["start_capital"] = cap
            _event("Paper trading started: %s on %s, check every %ds" % (", ".join(syms), tf, interval))
            _save()
        _start_thread()
        return jsonify(ok=True)
 
    @app.post("/api/paper/stop")
    def paper_stop():
        d = request.get_json(silent=True) or {}
        if not guard(d):
            return jsonify(error="Wrong password"), 401
        with _lock:
            STATE["enabled"] = False
            _event("Paper trading stopped (open trades stay open until you start again)")
            _save()
        _stop.set()
        return jsonify(ok=True)
 
    @app.post("/api/paper/reset")
    def paper_reset():
        global STATE
        d = request.get_json(silent=True) or {}
        if not guard(d):
            return jsonify(error="Wrong password"), 401
        _stop.set()
        with _lock:
            STATE = _fresh()
            _recent.clear()
            _save()
        return jsonify(ok=True)
 
    @app.post("/api/paper/status")
    def paper_status():
        d = request.get_json(silent=True) or {}
        if not guard(d):
            return jsonify(error="Wrong password"), 401
        with _lock:
            alive = _thread is not None and _thread.is_alive()
            age = time.time() - STATE["heartbeat"] if STATE["heartbeat"] else None
            return jsonify(
                running=bool(STATE["enabled"] and alive), config=STATE["config"], age=age,
                stats=_stats(), open=[_view(t) for t in STATE["open"]],
                pending=[_view(t) for t in STATE["pending"]],
                closed=[_view(t) for t in STATE["closed"][-20:]][::-1],
                events=STATE["events"][:25])
 
    if STATE["enabled"]:                 # resume after a restart / redeploy
        _start_thread()
 
 
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper Trader</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#0f1419;color:#e6e6e6}
  h1{font-size:1.4rem;margin:.2em 0} h2{font-size:1.05rem;margin:0 0 8px}
  a{color:#58a6ff} .sub,.note{color:#9aa4af} .note{font-size:.8rem}
  section{background:#18202a;padding:14px;border-radius:10px;margin-top:14px}
  label{display:grid;gap:4px;font-size:.9rem;color:#b8c2cc}
  input,select{padding:10px;border-radius:6px;border:1px solid #2c3a4a;background:#0f1419;color:#e6e6e6;font-size:1rem}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
  .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px}
  .btns{display:flex;gap:10px;margin-top:12px}
  button{padding:11px 16px;border:0;border-radius:6px;background:#2f81f7;color:#fff;font-size:1rem;cursor:pointer;flex:1}
  button.red{background:#a33a3a} button.grey{background:#4a5563}
  .err{color:#ff8a8a;min-height:1.2em}
  .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
  .box{background:#0f1419;border:1px solid #2c3a4a;border-radius:8px;padding:8px;text-align:center}
  .box b{display:block;font-size:1.1rem} .box span{font-size:.75rem;color:#9aa4af}
  .t{background:#0f1419;border:1px solid #2c3a4a;border-radius:8px;padding:10px;margin-top:8px;font-size:.85rem;line-height:1.5}
  .pos{color:#4cc27a} .neg{color:#ff8a8a}
  .tag{padding:2px 8px;border-radius:10px;font-weight:700;font-size:.75rem;color:#fff;margin-right:6px}
  .tag.LONG{background:#1f7a3d} .tag.SHORT{background:#a33a3a}
  #ev div{font-size:.8rem;color:#b8c2cc;margin-top:4px}
</style></head><body>
<h1>Paper Trader <span class="note">fake money, real signals</span></h1>
<p class="sub">Trades the app's BUY/SELL signals automatically with virtual money, 24 hours, on crypto, Indian stocks, forex and commodities. <a href="/">Back to planner</a></p>
<section>
  <label>Symbols (comma separated, max 8)
    <input id="syms" value="btc, eth, eurusd, gold, nifty, reliance"></label>
  <div class="row3">
    <label>Timeframe<select id="tf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>1d</option><option>1wk</option></select></label>
    <label>Check every<select id="intv"><option value="30">30 sec</option><option value="60" selected>1 min</option><option value="120">2 min</option><option value="300">5 min</option></select></label>
    <label>Leverage<input id="lev" type="number" min="1" max="50" value="1"></label>
  </div>
  <div class="row">
    <label>Paper capital<input id="cap" type="number" min="1" value="100000"></label>
    <label>Risk per trade (%)<input id="risk" type="number" step="any" min="0.1" max="10" value="1"></label>
  </div>
  <label style="margin-top:10px">Password (leave empty if you did not set APP_PASSWORD)<input id="pw" type="password"></label>
  <div class="btns"><button id="start">Start</button><button id="stop" class="grey">Stop</button><button id="reset" class="red">Reset</button></div>
  <p class="err" id="msg"></p>
  <p class="note" id="status"></p>
</section>
<section><h2>Result so far <span class="note">money in your capital currency, and in R</span></h2><div class="grid" id="stats"></div>
  <p class="note">Money = result in R x the amount you risk per trade (capital x risk %), so BTC (USD), Nifty (INR) and forex all add up in one currency. Fees and slippage (0.05% per side) are included.</p></section>
<section><h2>Open trades</h2><div id="open"></div><h2 style="margin-top:12px">Waiting limit orders</h2><div id="pend"></div></section>
<section><h2>Closed trades (last 20)</h2><div id="closed"></div></section>
<section><h2>Log</h2><div id="ev"></div></section>
<p class="note">Paper results are not a promise of real results. Real markets have gaps, delays and bigger slippage. Not financial advice.</p>
<script>
const $ = id => document.getElementById(id);
function el(tag, cls, text){const e=document.createElement(tag); if(cls)e.className=cls; if(text!=null)e.textContent=text; return e;}
const f = v => v==null ? '-' : (Math.abs(v)<10 ? v.toFixed(5) : v.toFixed(2));
const sg = v => (v>0?'+':'')+v.toFixed(2)+'R';
const mn = v => (v==null ? '-' : (v>0?'+':v<0?'-':'')+Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:2}));
const am = v => Number(v).toLocaleString(undefined,{maximumFractionDigits:2});
try{ $('pw').value = sessionStorage.getItem('ppw')||''; }catch(e){}
function msg(t){ $('msg').textContent = t||''; }
async function api(path, body){
  const pw = $('pw').value; try{ sessionStorage.setItem('ppw', pw); }catch(e){}
  const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(Object.assign({password:pw}, body||{}))});
  const d = await r.json(); if(!r.ok) throw new Error(d.error||'Error'); return d;
}
$('start').onclick = async()=>{ msg(); try{ await api('/api/paper/start',{symbols:$('syms').value,tf:$('tf').value,
  capital:$('cap').value,risk:$('risk').value,leverage:$('lev').value,interval:$('intv').value}); refresh(); }catch(e){ msg(e.message);} };
$('stop').onclick = async()=>{ msg(); try{ await api('/api/paper/stop'); refresh(); }catch(e){ msg(e.message);} };
$('reset').onclick = async()=>{ if(!confirm('Delete all paper trades and history?')) return;
  msg(); try{ await api('/api/paper/reset'); refresh(); }catch(e){ msg(e.message);} };
function tradeBox(t, closed){
  const b = el('div','t'), h = el('div');
  h.appendChild(el('span','tag '+t.side,t.side)); h.appendChild(el('b','',t.symbol+' | '+t.tf));
  b.appendChild(h);
  b.appendChild(el('div','note','Amount risked on this trade: '+am(t.risk_money)));
  b.appendChild(el('div','','Entry '+f(t.entry)+' | Stop '+f(t.stop)+' | T1 '+f(t.t1)+' | T2 '+f(t.t2)+' | Qty '+t.qty));
  if(closed){ const c = el('div', t.r>0?'pos':'neg', t.reason+': '+mn(t.money)+' money ('+sg(t.r)+')'); b.appendChild(c); }
  else if(t.now_r!=null){ const c = el('div', t.now_r>=0?'pos':'neg', 'Price '+f(t.last)+' | now '+mn(t.money)+' money ('+sg(t.now_r)+')'+(t.be?' | stop at entry':'')); b.appendChild(c); }
  else b.appendChild(el('div','note','Waiting for price to reach '+f(t.entry)));
  return b;
}
function fill(id, list, closed, empty){ const c=$(id); c.textContent='';
  if(!list.length){ c.appendChild(el('p','note',empty)); return; }
  list.forEach(t=>c.appendChild(tradeBox(t, closed))); }
function render(d){
  const s = d.stats, c = d.config||{};
  $('status').textContent = (d.running?'RUNNING':'STOPPED')+(c.symbols?' | '+c.symbols.join(', ')+' | '+c.tf:'')
    +(d.age!=null?' | last check '+Math.round(d.age)+'s ago':'');
  const g = $('stats'); g.textContent='';
  [['Starting money',am(s.start)],['Money now',am(s.balance)],['Profit / Loss',mn(s.profit)],
   ['Today',mn(s.day_money)+(s.paused?' (paused)':'')],['Still open (floating)',mn(s.floating)],['Closed result',mn(s.realized)],
   ['Trades',s.trades],['Win rate',s.win_rate+'%'],['Total',sg(s.total_r)],['Average',sg(s.avg_r)],
   ['Worst drawdown',s.max_dd_r+'R']].forEach(x=>{
    const b=el('div','box'); b.appendChild(el('b','',String(x[1]))); b.appendChild(el('span','',x[0])); g.appendChild(b); });
  fill('open', d.open, false, 'No open trades.');
  fill('pend', d.pending, false, 'No waiting orders.');
  fill('closed', d.closed, true, 'No closed trades yet.');
  const e=$('ev'); e.textContent=''; d.events.forEach(x=>e.appendChild(el('div','',x.t+'  '+x.text)));
}
async function refresh(){ try{ render(await api('/api/paper/status')); }catch(e){ msg(e.message);} }
refresh(); setInterval(refresh, 5000);
</script></body></html>"""
 
