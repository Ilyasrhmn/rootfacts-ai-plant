import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';

// Precache static assets (Vite build assets)
precacheAndRoute(self.__WB_MANIFEST);

const CACHE_NAME = 'ai-models-storage';

/**
 * Custom Handler untuk Model AI (Transformers.js & TensorFlow.js)
 * Menggunakan Manual Cache API untuk menghindari limitasi Workbox plugins
 * pada file besar (.bin, .wasm, .json).
 */
async function modelCacheHandler({ request }) {
  const cache = await caches.open(CACHE_NAME);

  // 1. Cek di cache dulu
  const cachedResponse = await cache.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }

  // 2. Jika tidak ada, fetch dari network
  try {
    const networkResponse = await fetch(request);

    /**
     * CRITICAL: Hanya simpan response lengkap (200 OK).
     * Transformers.js sering menggunakan Range Requests (206 Partial Content).
     * Cache API .put() akan ERROR jika mencoba menyimpan response 206.
     */
    if (
      networkResponse.status === 200 &&
      (networkResponse.type === 'basic' || networkResponse.type === 'cors')
    ) {
      // Pastikan bukan HTML (Netlify 404 redirect)
      const contentType = networkResponse.headers.get('content-type');
      if (!contentType || !contentType.includes('text/html')) {
        await cache.put(request, networkResponse.clone());
      }
    }

    return networkResponse;
  } catch (error) {
    console.error('SW: Model fetch failed:', error);
    throw error;
  }
}

// Register route untuk berkas model AI: bobot TensorFlow.js (.bin), metadata/config
// (.json), runtime ONNX (.wasm), dan bobot model Transformers.js (.onnx). Dengan begitu
// deteksi maupun generasi fun fact tetap berfungsi dalam mode luring setelah sekali online.
registerRoute(
  ({ url }) =>
    url.pathname.endsWith('.bin') ||
    url.pathname.endsWith('.json') ||
    url.pathname.endsWith('.wasm') ||
    url.pathname.endsWith('.onnx'),
  modelCacheHandler
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
