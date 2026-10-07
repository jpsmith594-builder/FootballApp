-- Run this once in Supabase's SQL editor (Project > SQL Editor > New query)
-- AFTER you've run refresh_cache.py at least once locally (so the
-- cached_fixtures / refresh_meta / push_subscriptions tables already exist
-- — db.py's metadata.create_all() creates them automatically the first time
-- it connects, same as it always has for the predictions table).
--
-- This locks every table down to the minimum the frontend actually needs:
-- read-only on the prediction data, write-only (insert, no read) on push
-- subscriptions, and nothing at all on anything else. Your DATABASE_URL
-- connection (used by engine.py / refresh_cache.py) uses the `postgres`
-- role, which has BYPASSRLS — none of this affects it. It only affects the
-- `anon` role, which is what the public anon key in frontend/index.html
-- authenticates as.

-- Public, read-only: the current fixture/prediction cache.
alter table cached_fixtures enable row level security;
create policy "cached_fixtures_public_read" on cached_fixtures
  for select using (true);

-- Public, read-only: "last updated" metadata per league.
alter table refresh_meta enable row level security;
create policy "refresh_meta_public_read" on refresh_meta
  for select using (true);

-- Public, read-only: resolved predictions, for the Track Record tab.
-- (Deliberately NOT exposing unresolved/future predictions here — those
-- live in the same table but the frontend's own query already filters to
-- resolved = true; RLS just backstops that at the database level too.)
alter table predictions enable row level security;
create policy "predictions_public_read_resolved_only" on predictions
  for select using (resolved = true);

-- Public, INSERT-ONLY: a visitor can register their own browser for push
-- notifications, but can never read back the list of subscriptions (which
-- would otherwise leak every subscriber's push endpoint — a quasi-identifier
-- you don't want publicly listable) or delete someone else's.
alter table push_subscriptions enable row level security;
create policy "push_subscriptions_public_insert" on push_subscriptions
  for insert with check (true);

-- team_adjustments: deliberately NOT given any policy below, so with RLS
-- enabled and no policies, the anon/authenticated API roles get zero
-- access — admin-only, as before, still reachable only via the
-- /admin/team-adjustment endpoint if you keep api_server.py deployed, or
-- by editing the table directly in the Supabase dashboard.
alter table team_adjustments enable row level security;
