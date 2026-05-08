import { registerRoute } from 'workbox-routing';
import { CacheFirst } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { precacheAndRoute } from 'workbox-precaching';
import { RangeRequestsPlugin } from 'workbox-range-requests';

// Precache static assets
precacheAndRoute(self.__WB_MANIFEST);

/**
 * Strategi CacheFirst untuk file model AI (.bin dan .json).
 * Penting untuk mendukung ketersediaan offline (Advanced grade).
 * Ditambahkan RangeRequestsPlugin untuk menangani file besar (Partial Content).
 */
registerRoute(
  ({ url }) => url.pathname.endsWith('.bin') || url.pathname.endsWith('.json'),
  new CacheFirst({
    cacheName: 'ai-models-cache',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30 Days
      }),
      new RangeRequestsPlugin(),
    ],
  })
);

// Fallback for general navigation
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match('/index.html');
      })
    );
  }
});
