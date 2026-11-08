import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import {
  SERVICE_WORKER_FILE,
  buildServiceWorkerSource,
  buildVersion,
  selectPrecacheFiles
} from './src/pwa/serviceWorker';

const DEFAULT_BASE = './';

/** Files under a directory, as forward-slash paths relative to it. */
function listFiles(root: string, dir = root): string[] {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return [];
  }
  return entries.flatMap((entry) => {
    const full = path.join(dir, entry);
    return statSync(full).isDirectory()
      ? listFiles(root, full)
      : [path.relative(root, full).split(path.sep).join('/')];
  });
}

/**
 * Write the offline service worker (src/pwa/serviceWorker.ts) next to index.html, pre-caching
 * every file of this build (bundle and public directory).
 */
function offlineServiceWorker(): Plugin {
  let publicDir = '';
  return {
    name: 'neuro-lingua-offline-service-worker',
    apply: 'build',
    enforce: 'post',
    configResolved(config) {
      publicDir = config.publicDir;
    },
    writeBundle(options, bundle) {
      const publicFiles = publicDir ? listFiles(publicDir) : [];
      const precache = selectPrecacheFiles([...Object.keys(bundle), ...publicFiles]);
      // The version changes whenever any built or public file changes.
      const version = buildVersion([
        ...Object.values(bundle).map(
          (item) =>
            `${item.fileName}:${
              item.type === 'chunk' ? item.code : Buffer.from(item.source).toString('base64')
            }`
        ),
        ...publicFiles.map(
          (file) => `${file}:${readFileSync(path.join(publicDir, file)).toString('base64')}`
        )
      ]);
      const source = buildServiceWorkerSource({ version, precache });
      writeFileSync(path.join(options.dir ?? 'dist', SERVICE_WORKER_FILE), source);
    }
  };
}

export default defineConfig(() => ({
  plugins: [react(), offlineServiceWorker()],
  // Use a relative base so assets load correctly on static hosts and nested paths
  base: process.env.VITE_BASE_PATH || DEFAULT_BASE,
  resolve: {
    alias: {
      // Provide a browser-friendly polyfill for Node's events module used by tsne-js
      events: 'events'
    }
  },
  optimizeDeps: {
    // Ensure the events polyfill is prebundled for dev/prod parity
    include: ['events']
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: undefined
      }
    }
  },
  server: {
    open: true
  },
  test: {
    environment: 'jsdom',
    setupFiles: './tests/setup.ts',
    globals: true
  }
}));
