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

## 5. Confirm the full pipeline

- Backend `/health` returns ok → data fetching and model fitting works
- Backend `/predictions/Premier League` returns real JSON with fixtures → model is serving predictions
- Frontend loads and shows fixtures with real team names and dates → full pipeline is live

## What to do if something's wrong

- **Backend `/health` never shows leagues**: check Render's logs tab — usually a bad or missing API key.
- **"Need at least ~20 played matches" error**: happens right after a season starts, before enough matches have been played. Wait for more fixtures, or lower the threshold in `poisson_model.py`'s `fit()` method.
- **Frontend shows a CORS or network error**: double-check the API URL has no trailing slash and starts with `https://`.
