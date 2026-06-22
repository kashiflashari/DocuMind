import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API base is configurable via VITE_API_URL (defaults to the local backend).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
