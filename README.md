# Trading Plan Assistant

Web app for stocks, indices, crypto, forex and commodities. Two ways to use it:

1. **Chart screenshot** - upload or paste (Ctrl+V) a screenshot. An AI vision model (Groq) reads
   the chart, then the app checks the levels and works out position size from your capital and risk %.
2. **Symbol** - live price data (Yahoo Finance): trend, support/resistance, entry, stop-loss,
   targets, position size, optional backtest.

3. **Live monitor** - list up to 6 symbols (e.g. `nifty, btc, eth`), pick a timeframe and how
   often to check. The page keeps re-checking the market and shows BUY NOW / BUY LIMIT / SELL NOW /
   SELL LIMIT / WAIT for each one, with the entry zone, stop-loss, targets and quantity for your
   capital. It plays a sound and shows a notification when a new signal appears. It works while the
   page stays open, and signals use the last completed candle so they do not flicker.
4. **AI chat** - ask questions under the plan ("why WAIT?", "explain simply"), or tell it
   "analyse nifty 15m" and it fetches a live plan itself. Replies in English, Hindi or Hinglish.
   It uses the capital and risk you typed in the form, and is limited to 20 messages per minute
   per visitor.

Planning tool only. Not financial advice. Prices read from an image can be wrong - always compare
with your real chart.

## Run locally

    pip install -r requirements.txt
    set GROQ_API_KEY=your_key        (Windows; use export on Mac/Linux)
    python app.py
    # open http://localhost:5000

Command-line version (symbol mode): `python trading_assistant.py`

## Deploy on Render

1. Push this folder to a GitHub repository.
2. On render.com: New > Blueprint (uses `render.yaml`) or New > Web Service, then pick the repo.
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
3. Add environment variables in the Render dashboard (never put keys in the code):
   - `GROQ_API_KEY` - required for screenshot reading and the AI chat
   - `GROQ_CHAT_MODEL` - default `qwen/qwen3.8-27b`; if a model is not available on your Groq account the app falls back to the vision model
   - `APP_PASSWORD` - protects your site so strangers cannot use your Groq key
   - `GROQ_VISION_MODEL` - default `qwen/qwen3.8-27b`; change it if Groq renames the model
   - `PYTHON_VERSION` - `3.12.7`

## API (symbol mode)

    curl -X POST https://YOUR-APP.onrender.com/api/plan \
      -H "Content-Type: application/json" -H "X-Password: YOUR_PASSWORD" \
      -d '{"symbol":"btc","tf":"1h","capital":1000,"risk":1,"backtest":true}'
