# NSE Swing Trade Screener + Telegram/WhatsApp Alerts

A rules-based daily screener for NSE stocks that sends candidate setups to
Telegram and/or WhatsApp. **This is a screening tool, not financial advice** —
it flags technical setups for you to review, it does not place trades.

## What it does
1. Pulls a list of NSE stocks (live NSE list, falls back to a static ~100-stock
   list of liquid names if NSE blocks the request — common from cloud IPs).
2. For each stock, fetches ~9 months of daily price/volume via `yfinance`.
3. Scores each stock 0-5 on: trend (price vs rising SMA50), momentum (RSI zone
   or oversold bounce), MACD bullish crossover, volume spike, and proximity to
   a 20-day high.
4. Stocks scoring >= `MIN_SCORE` (default 3) get a suggested stop-loss
   (20-day low) and target (1:2 risk-reward) attached.
5. Sends the results to Telegram and/or WhatsApp.

Edit the scoring rules in `screener.py` freely — swap in your own indicators
or thresholds.

## Local setup
```bash
git clone <your-repo-url>
cd swing_trader
pip install -r requirements.txt
cp .env.example .env   # then fill in your tokens
python main.py
```

## Getting a Telegram bot token (free, 2 minutes)
1. Open Telegram, message **@BotFather**, send `/newbot`, follow the prompts.
2. Copy the token it gives you into `TELEGRAM_BOT_TOKEN`.
3. Send your new bot any message (e.g. "hi") so it can message you back.
4. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and
   copy the `"chat":{"id": ...}` number into `TELEGRAM_CHAT_ID`.

## Getting WhatsApp alerts (optional, via Twilio)
1. Create a free Twilio account, open the WhatsApp Sandbox in the console.
2. Join the sandbox from your phone (send the given code to the given number).
3. Put your Account SID / Auth Token / sandbox numbers into `.env` and set
   `SEND_WHATSAPP=true`.
4. For *production* WhatsApp (not sandbox), you need Meta's WhatsApp Business
   API approval via Twilio — more paperwork, sandbox is fine for personal use.

## Where to deploy (fully automated, no laptop needed)

**Easiest & free: GitHub Actions** — included at
`.github/workflows/daily_screener.yml`. It already runs Mon–Fri at 3:45 PM IST.
Steps:
1. Push this folder to a new GitHub repo (can be private).
2. Repo → Settings → Secrets and variables → Actions → add `TELEGRAM_BOT_TOKEN`
   and `TELEGRAM_CHAT_ID` (and Twilio ones if using WhatsApp).
3. Done — GitHub runs it on schedule for free (within their generous free
   tier for scheduled jobs), no server to maintain or pay for.

**Other options, if you want more control or always-on behavior:**
- **Oracle Cloud Free Tier VM** — a genuinely free-forever small VM; run
  `scheduler.py` on it with `nohup` or as a `systemd` service for an
  always-on process, or just add `main.py` to `cron`.
- **Railway / Render** — deploy as a "cron job" or scheduled task service;
  free tiers are limited but easy to set up from a GitHub repo.
- **PythonAnywhere** — has a built-in scheduled-task feature; free tier
  restricts the exact time slightly but works fine for daily jobs.
- **AWS Lambda + EventBridge** — serverless, pay-per-invocation (this
  workload would cost pennies/month); needs a Lambda packaging step since
  `yfinance`/`ta` aren't in the default runtime.

For a single daily run, GitHub Actions or a free Oracle VM + cron are the
simplest — no cost, no maintenance.

## Disclaimer
Technical screens can and will produce false signals. Markets carry risk of
loss. This tool does not constitute investment advice; verify independently
and size positions according to your own risk tolerance.
