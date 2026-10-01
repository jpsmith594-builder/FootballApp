# Matchday Android

This branch prepares the Matchday football predictions frontend for Capacitor Android.

- Prediction backend remains on Render.
- Supabase remains the database/authentication layer.
- Android is a native shell around the packaged web frontend.
- Premium entitlement and Google Play Billing will be added after authentication.

## Local setup

1. Install Node.js LTS and Android Studio.
2. Run `npm install`.
3. Run `npm run sync`.
4. Run `npx cap add android` once if the Android platform has not yet been created.
5. Run `npx cap sync`.
6. Run `npx cap open android`.

The production API URL remains configured in the frontend code for this transition; the user-facing API URL control will be removed in the next backend/authentication phase.
