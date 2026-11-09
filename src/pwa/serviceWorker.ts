/**
 * Offline support for the built web app: the source of the service worker (`sw.js`) that the
 * build writes next to index.html (see `offlineServiceWorker` in vite.config.ts).
 *
 * - install:  pre-caches every file of the build (HTML, scripts, styles, lazily loaded chunks), so
 *             after one online visit the whole app works without a network connection
 * - activate: removes the caches of earlier builds and takes control of open pages
 * - fetch:    page navigations go to the network first (to pick up new deployments) and fall back
 *             to the cached app; other same-origin GET requests are served from the cache first
 *             (build files have content hashes in their names). Requests to other origins and
 *             non-GET requests are not intercepted — the app makes none.
 *
 * The worker is plain JavaScript generated from this template; it contains no third-party code.
 */

export const SERVICE_WORKER_FILE = 'sw.js';
export const CACHE_PREFIX = 'neuro-lingua-offline-';

export interface ServiceWorkerOptions {
  /** Build identifier; a new value makes browsers install the new worker and drop old caches. */
  version: string;
  /** Files to pre-cache, relative to the worker's location (e.g. "index.html", "assets/x.js"). */
  precache: string[];
  /** Page served for navigations when offline (default "index.html"). */
  fallback?: string;
}

/** Files of a build that the worker should pre-cache (source maps and the worker itself excluded). */
export function selectPrecacheFiles(fileNames: string[]): string[] {
  const files = fileNames.filter(
    (name) => !name.endsWith('.map') && name !== SERVICE_WORKER_FILE && !name.startsWith('.')
  );
  if (!files.includes('index.html')) files.push('index.html');
  return [...new Set(files)].sort();
}

/**
 * Short identifier of a build: FNV-1a over the given entries (file names with their contents), in
 * sorted order so that it does not depend on the order in which files were produced.
 */
export function buildVersion(entries: string[]): string {
  let h = 0x811c9dc5;
  for (const entry of [...entries].sort()) {
    for (let i = 0; i < entry.length; i++) {
      h ^= entry.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    h ^= 0x0a;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export function buildServiceWorkerSource(options: ServiceWorkerOptions): string {
  const fallback = options.fallback ?? 'index.html';
  const precache = options.precache.map((file) => `./${file.replace(/^\.?\//, '')}`);
  return `/* Neuro-Lingua offline service worker (generated at build time). */
const CACHE_PREFIX = ${JSON.stringify(CACHE_PREFIX)};
const CACHE = CACHE_PREFIX + ${JSON.stringify(options.version)};
const PRECACHE = ${JSON.stringify(precache)};
const FALLBACK = ${JSON.stringify(`./${fallback.replace(/^\.?\//, '')}`)};

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

// A cached response that went through a redirect cannot answer a navigation: copy it.
function servable(response) {
  if (!response.redirected) return response;
  return new Response(response.body, {
    headers: response.headers,
    status: response.status,
    statusText: response.statusText
  });
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = (await cache.match(request)) || (await cache.match(FALLBACK));
    if (cached) return servable(cached);
    throw error;
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok && response.type === 'basic') await cache.put(request, response.clone());
  return response;
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  if (new URL(request.url).origin !== self.location.origin) return;
  event.respondWith(request.mode === 'navigate' ? networkFirst(request) : cacheFirst(request));
});
`;
}
