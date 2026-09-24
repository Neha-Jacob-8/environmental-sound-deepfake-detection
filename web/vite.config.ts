import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Published under /<repo>/ on GitHub Pages; "./" keeps asset URLs relative so
  // the same build also works when opened from any other path.
  base: './',
  resolve: {alias: {'@': path.resolve(__dirname, '.')}},
  server: {
    // Same-origin in the browser, so no CORS and no base URL to configure.
    proxy: {'/api': {target: 'http://127.0.0.1:8000', changeOrigin: true}},
  },
});
