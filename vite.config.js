import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'public',
      filename: 'sw.js',
      registerType: 'autoUpdate',
      manifest: {
        name: 'Root Fact App',
        short_name: 'RootFacts',
        description: 'AI Vegetable Detector & Fun Fact Generator',
        theme_color: '#10b981',
        icons: [
          {
            src: 'icons/icon-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'icons/icon-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      },
      // Untuk strategi 'injectManifest', opsi globPatterns HARUS berada di dalam objek
      // injectManifest (bukan di objek `workbox`, yang hanya berlaku untuk strategi
      // 'generateSW'). Sebelumnya salah tempat sehingga tidak pernah benar-benar
      // diterapkan -- itulah sebabnya beberapa berkas (mis. favicon.ico) tidak ikut
      // ter-precache walau sudah tercantum di pattern.
      injectManifest: {
        maximumFileSizeToCacheInBytes: 50 * 1024 * 1024, // 50MB
        globPatterns: ['**/*.{js,css,html,ico,png,svg,wasm,json}']
      }
    })
  ],
  server: {
    port: 3001,
    host: true
  }
});
