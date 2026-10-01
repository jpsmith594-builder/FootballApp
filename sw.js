// Minimal service worker — this exists mainly to satisfy the PWA
// installability requirement (a manifest + a registered service worker).
// It only caches the app shell itself (this page, icons), never the live
// predictions from your API — those always need to be fetched fresh.
const CACHE_NAME = "matchday-shell-v1";
const SHELL_FILES = ["./index.html", "./manifest.json", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

// Handles an actual push event arriving from the server (via push.py +
// VAPID), distinct from the cache-related events above.
self.addEventListener("push", (event) => {
  let data = { title: "Matchday", body: "New update" };
  try { data = event.data ? event.data.json() : data; } catch (e) { /* non-JSON payload, use default */ }
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: "icon-192.png",
      data: { url: data.url || "./index.html" },
    })
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(clients.openWindow(event.notification.data.url || "./index.html"));
});

self.addEventListener("fetch", (event) => {
  // Never intercept API calls to your backend — only serve cached shell
  // files, and only when the network is genuinely unavailable.
  if (event.request.url.includes("/predictions/") || event.request.url.includes("/health")) {
    return;
  }
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
