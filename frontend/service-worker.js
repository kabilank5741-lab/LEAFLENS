const CACHE_NAME = 'leaflens-shell-v3';
const CORE = ['/', '/index.html', '/style.css', '/app.js', '/manifest.json', '/icon-192.png', '/icon-512.png'];
self.addEventListener('install', event => { event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(CORE)).then(() => self.skipWaiting())); });
self.addEventListener('activate', event => { event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', event => {
  const req = event.request; const url = new URL(req.url);
  if (req.method !== 'GET') return;
  if (url.origin === self.location.origin) {
    event.respondWith(caches.match(req).then(cached => cached || fetch(req).then(res => { if (res.ok && (url.pathname.startsWith('/model/') || CORE.includes(url.pathname))) { const clone = res.clone(); caches.open(CACHE_NAME).then(c => c.put(req, clone)); } return res; }).catch(() => caches.match('/index.html'))));
  } else if (url.hostname === 'cdn.jsdelivr.net' && url.pathname.includes('@tensorflow/tfjs')) {
    event.respondWith(caches.match(req).then(cached => cached || fetch(req).then(res => { const clone=res.clone(); caches.open(CACHE_NAME).then(c=>c.put(req,clone)); return res; })));
  }
});
