import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Dev: Vite serves the SPA and proxies the API to FastAPI (`make dev`). Built, FastAPI
// serves the SPA from the same origin (`make viewer`), so the proxy does not exist there.
const API = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8083';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    proxy: {
      '/api': { target: API, changeOrigin: false },
    },
  },
});
