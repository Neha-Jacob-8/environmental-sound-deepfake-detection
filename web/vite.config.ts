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
});
