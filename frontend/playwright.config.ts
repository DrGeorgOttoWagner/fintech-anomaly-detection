import { defineConfig } from '@playwright/test'
import os from 'node:os'
import path from 'node:path'

// The E2E suite starts its own FTAD backend on a separate port with an isolated,
// throw-away data directory, so it never touches the user's FTAD/var state.
const PORT = Number(process.env.FTAD_E2E_PORT ?? 8791)
const DATA_DIR = process.env.FTAD_E2E_DATA_DIR ?? path.join(os.tmpdir(), `ftad-e2e-${Date.now()}`)
const EXPORT_DIR = process.env.FTAD_E2E_EXPORT_DIR ?? path.join(DATA_DIR, 'export')

export default defineConfig({
  testDir: './e2e',
  timeout: 30 * 60 * 1000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    channel: process.env.FTAD_E2E_BROWSER_CHANNEL ?? 'chrome',
    viewport: { width: 1440, height: 900 },
    colorScheme: 'dark',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `mkdir -p "${EXPORT_DIR}" && cd ../backend && .venv/bin/python -m ftad.app`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: { FTAD_PORT: String(PORT), FTAD_DATA_DIR: DATA_DIR, FTAD_EXPORT_DIR: EXPORT_DIR },
  },
})
