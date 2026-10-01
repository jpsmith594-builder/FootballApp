"""
push.py
Real browser push notifications via the Web Push standard — no app store,
no Firebase account, no third-party push service account needed. Works
through VAPID (Voluntary Application Server Identification): your server
signs each push with a private key, and the browser verifies it against the
matching public key the frontend registered with.

Generate your own keypair (don't reuse the sample one committed anywhere
public) with:
    python -c "
    from py_vapid import Vapid02; import base64
    v = Vapid02(); v.generate_keys()
    priv = v.private_key.private_numbers().private_value.to_bytes(32,'big')
    print('VAPID_PRIVATE_KEY=', base64.urlsafe_b64encode(priv).decode().rstrip('='))
    pn = v.public_key.public_numbers()
    pub = b'\x04' + pn.x.to_bytes(32,'big') + pn.y.to_bytes(32,'big')
    print('VAPID_PUBLIC_KEY=', base64.urlsafe_b64encode(pub).decode().rstrip('='))
    "
"""

import os
import json
import logging
from pywebpush import webpush, WebPushException
import db

log = logging.getLogger("push")

VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "")
VAPID_CLAIMS_EMAIL = os.environ.get("VAPID_CLAIMS_EMAIL", "mailto:admin@example.com")

# Only fire a push for a genuinely strong value bet — the same 4% threshold
# used for *flagging* a bet in value_bet.py would mean a notification most
# refresh cycles, which trains people to ignore them. Notifications should
# be rarer and more meaningful than the in-app value-bet list.
NOTIFY_EDGE_THRESHOLD = float(os.environ.get("NOTIFY_EDGE_THRESHOLD", "0.08"))


def send_notification_to_all(title: str, body: str, url: str = None):
    if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        log.warning("Push not configured (missing VAPID keys) — skipping notification")
        return
    subs = db.get_push_subscriptions()
    if not subs:
        return
    payload = json.dumps({"title": title, "body": body, "url": url or "/"})
    for sub in subs:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
                },
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_CLAIMS_EMAIL},
            )
        except WebPushException as e:
            status = e.response.status_code if e.response is not None else None
            log.warning(f"Push failed for a subscriber (status {status}): {e}")
            if status in (404, 410):
                # The browser has permanently unsubscribed or the
                # subscription expired — stop retrying it every cycle.
                db.remove_push_subscription(sub["endpoint"])


def maybe_notify_value_bet(league: str, home: str, away: str, value_bet: dict):
    """Called only when a prediction was *newly* recorded (see
    db.record_prediction's return value) — this is what stops the same
    value bet from re-triggering a notification on every 6-hour refresh."""
    if not value_bet or value_bet["edge"] < NOTIFY_EDGE_THRESHOLD:
        return
    send_notification_to_all(
        title=f"Value bet: {league}",
        body=f"{home} vs {away} — {value_bet['market']} @ {value_bet['odds']} "
             f"(edge +{value_bet['edge']*100:.0f}%)",
        url="/",
    )
