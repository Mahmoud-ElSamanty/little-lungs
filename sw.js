/**
 * sw.js - Service Worker for Offline Execution
 * 
 * Pre-caches web assets, ONNX runtime binaries, and models so that the
 * Huawei MatePad 11.5 tablet can operate completely offline in classrooms.
 */

const CACHE_NAME = "little-lungs-v1";

const ASSETS_TO_CACHE = [
  "./",
  "index.html",
  "style.css",
  "config.json",
  "manifest.json",
  "bus.jpg",
  "js/yolo.js",
  "js/app.js",
  "libs/ort.min.js",
  "libs/ort-wasm-simd.wasm",
  "libs/ort-wasm.wasm",
  "libs/ort-wasm-simd-threaded.wasm",
  "models/yolov8n.onnx"
];

// Install: pre-cache assets
self.addEventListener("install", (event) => {
  console.log("[Service Worker] Install event: caching assets...");
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      // Add assets individually so a missing optional asset (e.g. model before export) does not abort everything
      for (const asset of ASSETS_TO_CACHE) {
        try {
          await cache.add(asset);
        } catch (err) {
          console.warn(`[Service Worker] Warning: Could not pre-cache ${asset}:`, err.message);
        }
      }
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener("activate", (event) => {
  console.log("[Service Worker] Activate event: clearing stale caches...");
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch: Cache-first strategy for instant offline responsiveness
self.addEventListener("fetch", (event) => {
  // Only handle GET requests
  if (event.request.method !== "GET") return;

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        // Cache valid fetched items dynamically
        if (networkResponse && networkResponse.status === 200) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      }).catch(() => {
        // Fallback if offline and not in cache
        if (event.request.headers.get("accept") && event.request.headers.get("accept").includes("text/html")) {
          return caches.match("index.html");
        }
      });
    })
  );
});
