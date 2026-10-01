# Going live: full deployment guide

## 1. Get your API keys and database (10 min)

- **football-data.org**: register free at https://www.football-data.org/client/register — key arrives by email instantly.
- **the-odds-api.com** (optional, only needed for value bets): free tier gives 500 requests/month.
- **Supabase** (needed for the Track Record tab): create a free project at https://supabase.com, then go to Project Settings > Database and copy the connection string (the "URI" one, not the individual host/port fields). This is what actually stores your prediction history — without it, the app falls back to a local SQLite file that Render wipes on every restart, and your track record would reset constantly.

## 2. Put the code on GitHub

Render deploys from a GitHub repo. Create a new repo, push everything in this
`football-predictor` folder to it (README, all `.py` files, `requirements.txt`,
`frontend/`).

## 3. Deploy the backend on Render (free tier)

1. Go to https://render.com, sign up, click **New > Web Service**.
2. Connect your GitHub repo.
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add:
   - `FOOTBALL_DATA_API_KEY` = your key from step 1
   - `ODDS_API_KEY` = your key from step 1 (optional)
   - `DATABASE_URL` = your Supabase connection string from step 1
   - `BLEND_WEIGHT` = optional, 0-1, defaults to 0.5 (how much the displayed prediction trusts your model vs. the bookmaker's odds — 1.0 ignores the market entirely)
5. Deploy. Render gives you a URL like `https://matchday-predictor.onrender.com`.
6. Visit `https://your-url.onrender.com/health` — you should see `{"status": "ok"}`
   once the first data refresh finishes (can take a minute or two on first boot).

**Free-tier note:** Render's free web services sleep after 15 minutes of no
traffic and take ~30-60 seconds to wake on the next request. Fine for personal
use; for something you show other people regularly, a $7/month Render
instance removes the sleep.

## 4. Deploy the frontend on Netlify (free)

1. Go to https://app.netlify.com/drop and drag in the `frontend/` folder — that's it, no account setup needed for a quick test (an account is needed to keep it live long-term and get a stable URL).
2. Netlify gives you a URL like `https://matchday-yourname.netlify.app`.
3. Open it, paste your Render backend URL (from step 3) into the "API base URL" box, and it fetches live predictions.

## 5. Optional: error tracking with Sentry

1. Create a free project at https://sentry.io (choose "FastAPI" as the platform).
2. Copy the DSN it gives you.
3. Add `SENTRY_DSN` as an environment variable on Render with that value.

Without this set, the app runs exactly as before — Sentry is entirely opt-in.
With it set, any unhandled error anywhere in the app gets reported to your
Sentry dashboard with a full stack trace, instead of just disappearing into
Render's logs where you'd only find it by chance.

## 6. Optional: uptime monitoring

1. Create a free account at https://uptimerobot.com.
2. Add a new HTTP(s) monitor pointing at `https://your-app.onrender.com/health`.
3. Set the check interval to 5 minutes and add your email as an alert contact.

This means you find out within minutes if the backend goes down, rather than
waiting for a friend to mention the app looks empty.

## 7. Optional: push notifications for strong value bets

1. Generate your own VAPID keypair (don't reuse any keys from earlier
   conversations about this project — generate fresh ones for your actual
   deployment):
   ```bash
   pip install py-vapid
   python -c "
   from py_vapid import Vapid02; import base64
   v = Vapid02(); v.generate_keys()
   priv = v.private_key.private_numbers().private_value.to_bytes(32,'big')
   print('VAPID_PRIVATE_KEY=', base64.urlsafe_b64encode(priv).decode().rstrip('='))
   pn = v.public_key.public_numbers()
   pub = b'\x04' + pn.x.to_bytes(32,'big') + pn.y.to_bytes(32,'big')
   print('VAPID_PUBLIC_KEY=', base64.urlsafe_b64encode(pub).decode().rstrip('='))
   "
   ```
2. Add both printed values as environment variables on Render:
   `VAPID_PRIVATE_KEY` and `VAPID_PUBLIC_KEY`.
3. Add `VAPID_CLAIMS_EMAIL` = `mailto:your-email@example.com` (required by
   the Web Push spec — it's how push services can contact you if your server
   is misbehaving, not shown to users).
4. Redeploy. Open the frontend and tap "Enable value-bet notifications" —
   each friend needs to do this once on their own device/browser.

A notification fires automatically the first time a fixture's value bet
crosses an 8% edge (configurable via `NOTIFY_EDGE_THRESHOLD`) — not every
refresh cycle, just the first time that specific bet is detected.

## 8. Optional: manually adjusting for injuries

No free API gives reliable real-time injury data, so this is a manual
override you set yourself when you know something the model doesn't:

```bash
curl -X POST "https://your-app.onrender.com/admin/team-adjustment" \
  -H "X-Admin-Secret: your-secret" \
  -d "league=Premier League&team=Arsenal&goal_multiplier=0.8&note=Saka out injured"
```

Set `ADMIN_SECRET` as an environment variable on Render first — without it
set, this endpoint rejects every request, which is the safe default. Clear
an adjustment the same way with a DELETE request to the same URL.

## 9. Confirm the full pipeline

- Backend `/health` returns ok → data fetching and model fitting works
- Backend `/predictions/Premier League` returns real JSON with fixtures → model is serving predictions
- Frontend loads and shows fixtures with real team names and dates → full pipeline is live

## What to do if something's wrong

- **Backend `/health` never shows leagues**: check Render's logs tab — usually a bad or missing API key.
- **"Need at least ~20 played matches" error**: happens right after a season starts, before enough matches have been played. Wait for more fixtures, or lower the threshold in `poisson_model.py`'s `fit()` method.
- **Frontend shows a CORS or network error**: double-check the API URL has no trailing slash and starts with `https://`.
