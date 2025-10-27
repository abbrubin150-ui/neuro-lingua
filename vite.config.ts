import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: process.env.NODE_ENV === 'production' ? '/neuro-lingua/' : '/',
  server: {
    open: true
  },
  test: {
    environment: 'jsdom',
    setupFiles: './tests/setup.ts',
    globals: true
  }
});
