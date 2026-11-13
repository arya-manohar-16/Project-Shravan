// A tiny service worker to allow the browser to recognize the site as an installable PWA.
self.addEventListener('install', (e) => {
  self.skipWaiting();
});

self.addEventListener('fetch', (e) => {
  e.respondWith(fetch(e.request));
});