# Matchday Predictor — backend

A working prediction engine for the top 5 European leagues: live data ingestion,
a proper statistical model (not a toy formula), and value-bet detection against
real bookmaker odds.

## Setup

```bash
pip install -r requirements.txt
export FOOTBALL_DATA_API_KEY=your_key   # free tier: football-data.org
export ODDS_API_KEY=your_key            # optional, for value bets: the-odds-api.com
python main.py
```

Both providers have free tiers generous enough to develop and test against.
Rate limits will matter at scale — see "Scaling this up" below.

## Files

| File | What it does |
|---|---|
| `data_fetcher.py` | Pulls historical results and upcoming fixtures from football-data.org |
| `poisson_model.py` | Dixon-Coles model — fits attack/defense ratings per team from real results |
| `value_bet.py` | Pulls bookmaker odds, strips the margin out, flags mispriced markets |
| `main.py` | Runs the full pipeline end to end for all five leagues |

## The model: why Dixon-Coles instead of a rating gap

The demo app used a made-up formula (rating difference → win probability). This
version fits a real statistical model to actual results:

- Every team gets an **attack** rating and a **defense** rating, estimated by
  maximum likelihood from every match they've played (weighted so recent form
  counts more than results from a year ago — a team fitted from stale data is
  a team you'll misprice).
- Expected goals for a fixture come from combining the home side's attack with
  the away side's defense (and vice versa), plus a fitted home-advantage term.
- A correlation adjustment (the "Dixon-Coles" part specifically) fixes a known
  quirk where plain independent Poisson slightly underrates low-scoring
  results like 0-0 and 1-1.
- From expected goals you get a full scoreline probability grid, which is where
  the 1X2 odds, most-likely-scoreline, and over/under markets all come from.

This is the same family of model used by professional trading desks as a
baseline before layering in injuries, referee tendencies, or player-level xG —
each of those is a natural next iteration once the core model is running.

## The value-bet layer — where the money actually is

Predictions alone don't make money; **mispriced predictions relative to the
market** do. `value_bet.py`:

1. Takes bookmaker decimal odds and strips out the bookmaker's margin
   ("de-vigging") to get their true implied probability.
2. Compares that to the model's probability. If the model thinks a result is
   meaningfully more likely than the market does, that's a value bet.
3. Sizes the bet using fractional (half) Kelly — full Kelly assumes your
   model is perfectly calibrated, which no model is, so half-Kelly trades a
   bit of theoretical growth for much lower risk of a bad run wiping you out.
4. Only flags edges above a threshold (default 4 percentage points), since a
   1-2% edge is inside the model's own noise and won't survive bookmaker
   margin and odds movement in practice.

**Important:** an edge here means the model disagrees with the market, not
that a bet is safe. Treat this as a research/screening tool, backtest it
properly against historical odds before risking real money, and never bet
more than you can lose. Where you are, gambling on sports may also require a
license if you're offering this to others rather than using it yourself.

## Turning this into a product

Three viable paths, in order of how defensible they are:

**1. Subscription tips/insights app (lowest moat, fastest to launch).**
Package the predictions and value-bet flags into a clean app or Telegram bot.
$10-20/month. Weakness: tipster apps are a crowded, trust-poor market, and
customers churn hard the moment a losing streak hits — which will happen even
with a genuinely good model, because variance is real. Retention lives or
dies on transparent long-run tracking, not this week's picks.

**2. B2B model-as-a-service (stronger moat).** License predictions via API to
fantasy sports platforms, betting content sites, or smaller sportsbooks that
can't build this in-house. Priced per league per month, or per API call.
Selling infrastructure to other businesses is a more durable position than
selling picks to individual bettors.

**3. Your own trading operation (highest ceiling, highest effort).** Use the
model yourself against multiple bookmakers, staking systematically with strict
bankroll rules, and treat it as a quant trading problem rather than a product.
This is where professional betting syndicates actually operate — no
subscribers, no churn, just the edge itself. It's also the most demanding:
bookmakers limit or ban accounts that consistently win, so this doesn't scale
the way a normal business does, and results need a large sample before you
can trust them over model risk and bad luck.

**Realistic starting point:** run this on yourself first, exactly like the
running coach app — track every prediction against what actually happened for
a full season before trusting the edge, let alone monetizing it. A model that
looks good on paper and a model that beats closing odds over 500+ bets are
very different claims.

## Scaling this up

- football-data.org's free tier is 10 requests/minute — fine for personal use,
  too slow for many users hitting it live. API-Football (via RapidAPI) has
  higher-throughput paid tiers built for this.
- Cache fixture and result data in a real database (Postgres) rather than
  refetching on every request — results don't change after the final whistle,
  so there's no reason to re-pull them.
- Odds move constantly; if you go the trading route, you need near-real-time
  odds polling, not a daily pull, since value bets vanish within minutes as
  bookmakers correct the market.
