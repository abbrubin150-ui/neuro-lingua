import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(() => ({
  plugins: [react()],
  base: '/neuro-lingua/',
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
