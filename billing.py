"""
billing.py
Stripe subscription billing — a single "Pro" plan, upgrading a user's tier
on successful checkout and downgrading on cancellation.

This deliberately does NOT touch Stripe at import time or module load:
every function checks whether Stripe is configured first and raises a
clear, catchable error if not, rather than crashing the whole app on
startup when keys haven't been set yet (e.g. during initial deploy, or in
tests). Set these environment variables to activate billing:

    STRIPE_SECRET_KEY       — from the Stripe dashboard (Developers > API keys)
    STRIPE_PRICE_ID         — the recurring Price object for your Pro plan
    STRIPE_WEBHOOK_SECRET   — from the webhook endpoint you create (see
                              MONETIZATION.md for the full walkthrough)
    STRIPE_SUCCESS_URL      — where Stripe sends the user after paying
    STRIPE_CANCEL_URL       — where Stripe sends the user if they back out
"""

import os
import logging

log = logging.getLogger("billing")

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SUCCESS_URL = os.environ.get("STRIPE_SUCCESS_URL", "https://example.com/success")
STRIPE_CANCEL_URL = os.environ.get("STRIPE_CANCEL_URL", "https://example.com/cancel")


class BillingNotConfigured(Exception):
    pass


def _require_stripe():
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise BillingNotConfigured(
            "Billing isn't set up yet — STRIPE_SECRET_KEY and STRIPE_PRICE_ID "
            "must both be set. See MONETIZATION.md for the setup steps."
        )
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    return stripe


def create_checkout_session(email: str) -> str:
    """Returns the Stripe-hosted Checkout URL to redirect the user to. The
    email is attached as both the Checkout customer email AND passed through
    client_reference_id, so the webhook handler can match the completed
    payment back to a user without needing its own session state."""
    stripe = _require_stripe()
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        customer_email=email,
        client_reference_id=email,
        success_url=STRIPE_SUCCESS_URL,
        cancel_url=STRIPE_CANCEL_URL,
    )
    return session.url


def create_billing_portal_session(stripe_customer_id: str) -> str:
    """Lets an existing Pro subscriber manage or cancel their own
    subscription without you handling it manually — Stripe's hosted portal
    covers payment method updates, invoices, and cancellation."""
    stripe = _require_stripe()
    session = stripe.billing_portal.Session.create(
        customer=stripe_customer_id, return_url=STRIPE_SUCCESS_URL,
    )
    return session.url


def verify_and_parse_webhook(payload: bytes, signature_header: str):
    """Verifies the webhook actually came from Stripe (not a forged request
    hitting your public endpoint) using the signing secret. Raises on a bad
    signature — the caller should turn that into a 400, never process an
    unverified payload."""
    stripe = _require_stripe()
    if not STRIPE_WEBHOOK_SECRET:
        raise BillingNotConfigured("STRIPE_WEBHOOK_SECRET is not set")
    return stripe.Webhook.construct_event(payload, signature_header, STRIPE_WEBHOOK_SECRET)


def handle_webhook_event(event) -> dict:
    """
    Pure-ish dispatcher: given a verified Stripe event, returns what should
    change in our own database, without actually touching db.py itself —
    this is what makes it testable without a real database or a real Stripe
    account (see tests/test_billing.py). The caller (api_server.py) applies
    the returned instruction via db.set_user_tier.
    """
    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        email = data.get("client_reference_id") or data.get("customer_email")
        return {
            "action": "upgrade", "email": email,
            "stripe_customer_id": data.get("customer"),
            "stripe_subscription_id": data.get("subscription"),
        }

    if event_type in ("customer.subscription.deleted", "customer.subscription.updated"):
        # For "updated", only treat it as a downgrade if Stripe says the
        # subscription is no longer active (past_due, canceled, unpaid,
        # etc.) — an ordinary renewal also fires "updated" and must NOT
        # downgrade an otherwise-healthy subscriber.
        status = data.get("status")
        if event_type == "customer.subscription.deleted" or status not in ("active", "trialing"):
            return {"action": "downgrade", "stripe_customer_id": data.get("customer")}
        return {"action": "none"}

    return {"action": "none"}
