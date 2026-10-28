import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: `npm run dev` on :5173 proxies /api to the FastAPI service on :8000.
// Prod: `npm run build` writes dist/, which FastAPI serves at / (one server for the demo).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
