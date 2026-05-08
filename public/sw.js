import { registerRoute } from 'workbox-routing';
import { CacheFirst } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { precacheAndRoute } from 'workbox-precaching';
import { RangeRequestsPlugin } from 'workbox-range-requests';

// Precache static assets
precacheAndRoute(self.__WB_MANIFEST);

/**
 * Strategi CacheFirst untuk file model AI (.bin dan .json).
 * Ditambahkan validasi Content-Type untuk mencegah caching index.html saat 404.
 */
registerRoute(
  ({ url }) => url.pathname.endsWith('.bin') || url.pathname.endsWith('.json'),
  new CacheFirst({
    cacheName: 'ai-models-cache',
    plugins: [
      {
        /**
         * Verifikasi response sebelum disimpan di cache.
         * Jika response adalah HTML (hasil redirect 404 Netlify), jangan di-cache.
         */
        cacheWillUpdate: async ({ response }) => {
          if (response) {
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('text/html')) {
              console.warn('SW: Mencegah caching HTML untuk file model AI:', response.url);
              return null;
            }
          }
          return response;
        },
      },
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
