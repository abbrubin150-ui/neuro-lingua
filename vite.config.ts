import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const DEFAULT_BASE = './';

export default defineConfig(() => ({
  plugins: [react()],
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
