/**
 * Browser acceptance workflow (mirrors the mandatory acceptance checklist).
 *
 * Dataset: set FTAD_E2E_DATASET to a local CSV/Parquet file (e.g. the Kaggle creditcard.csv)
 * to run the workflow on real data. Without it, a clearly labelled synthetic fixture is used.
 * Set FTAD_E2E_SCREENSHOTS=<dir> to save full-page screenshots of each page.
 */
import { expect, test, type Page } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { loginEventsFixture, transactionsFixture } from './fixtures'

const DATASET = process.env.FTAD_E2E_DATASET ? path.resolve(process.env.FTAD_E2E_DATASET) : transactionsFixture()
const SHOTS = process.env.FTAD_E2E_SCREENSHOTS ? path.resolve(process.env.FTAD_E2E_SCREENSHOTS) : null
const fileName = path.basename(DATASET)

async function shot(page: Page, name: string) {
  if (!SHOTS) return
  fs.mkdirSync(SHOTS, { recursive: true })
  await page.waitForTimeout(600)
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: false })
}

async function uploadViaBrowse(page: Page, file: string) {
  const chooserPromise = page.waitForEvent('filechooser')
  await page.getByTestId('browse-files').first().click()
  const chooser = await chooserPromise
  await chooser.setFiles(file)
  // The new dataset becomes the active session (Overview then navigates on to schema mapping).
  await expect(page.getByTestId('active-dataset')).toContainText(path.basename(file), { timeout: 10 * 60_000 })
}

test.describe.configure({ mode: 'serial' })

test('acceptance workflow', async ({ page, request }) => {
  const consoleErrors: string[] = []
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()))

  // 1-2. Launch app, confirm empty dataset state
  await page.goto('/')
  await expect(page.getByText('Upload Your Dataset')).toBeVisible()
  await expect(page.getByText('No dataset loaded')).toBeVisible()
  expect(await (await request.get('/api/datasets')).json()).toEqual([])
  await shot(page, '01_empty_state')

  // 3-5. Browse Files → native file chooser → upload + parsing
  await uploadViaBrowse(page, DATASET)
  await expect(page).toHaveURL(/\/schema$/)
  await expect(page.getByTestId('active-dataset')).toContainText(fileName)

  // 6. Configure schema (accept/verify suggestions, then confirm)
  await expect(page.getByTestId('positive-labels')).toBeVisible()
  if (fileName.toLowerCase().includes('creditcard') || fileName.includes('transactions')) {
    await expect(page.getByTestId('role-Class')).toHaveValue('target')
    await expect(page.getByTestId('role-Amount')).toHaveValue('numeric')
    await expect(page.getByTestId('role-Time')).toHaveValue('time')
    await expect(page.getByTestId('role-V1')).toHaveValue('pca')
  }
  await shot(page, '02_schema_mapping')
  await page.getByTestId('confirm-schema').click()
  await expect(page.getByText('Schema confirmed').first()).toBeVisible()

  // 7. Descriptive statistics (computed values, not placeholders)
  await page.goto('/explore')
  const stats = page.getByTestId('overview-stats')
  await expect(stats).toContainText('Records', { timeout: 120_000 })
  const ds = (await (await request.get('/api/session')).json()).active_dataset
  await expect(stats).toContainText(ds.n_rows.toLocaleString('en-US'))
  await shot(page, '03_descriptive')

  // 8. Class imbalance
  await page.getByRole('tab', { name: 'Class imbalance' }).click()
  await expect(page.getByText('Why accuracy misleads here')).toBeVisible()
  await expect(page.getByText('Majority baseline accuracy')).toBeVisible()
  await shot(page, '04_class_imbalance')

  // 9. EDA: amount, time, features, findings
  await page.getByRole('tab', { name: /^Amount/ }).click()
  await expect(page.getByText(/Distribution of Amount by class/)).toBeVisible({ timeout: 120_000 })
  await shot(page, '05_amount')
  await page.getByRole('tab', { name: /^Time/ }).click()
  await expect(page.getByText('Events over time')).toBeVisible({ timeout: 120_000 })
  await expect(page.getByText(/not a calendar timestamp/).first()).toBeVisible()
  await shot(page, '06_time')
  await page.getByRole('tab', { name: 'Features' }).click()
  await expect(page.getByText('Correlation heatmap')).toBeVisible()
  await expect(page.getByText('Feature relevance indicators')).toBeVisible()
  await expect(page.locator('td', { hasText: /^V\d+/ }).first()).toBeVisible({ timeout: 180_000 })
  await shot(page, '07_features')
  await page.getByRole('tab', { name: 'Findings' }).click()
  await expect(page.getByTestId('findings')).toContainText('accuracy', { timeout: 120_000 })

  // 10. Split sliders: linked and always 100%
  await page.goto('/studio')
  await page.getByTestId('slider-train').fill('80')
  const pct = async (k: string) => Number((await page.getByTestId(`pct-${k}`).innerText()).replace('%', ''))
  expect(await pct('train')).toBe(80)
  expect((await pct('train')) + (await pct('validation')) + (await pct('test'))).toBe(100)
  await page.getByRole('button', { name: '70/15/15' }).click()
  expect(await pct('train')).toBe(70)
  await expect(page.getByText('Prevalence').first()).toBeVisible()
  await shot(page, '08_experiment_studio')

  // 11. Quick Benchmark (asynchronous; API stays responsive)
  await page.getByTestId('mode-quick').click()
  await page.getByTestId('run-experiment').click()
  await expect(page.getByTestId('run-status')).toBeVisible()
  const t0 = Date.now()
  const health = await request.get('/api/health')
  expect(health.ok()).toBeTruthy()
  expect(Date.now() - t0).toBeLessThan(3000)
  await expect(page.getByRole('button', { name: 'Open leaderboard' })).toBeVisible({ timeout: 25 * 60_000 })

  // 12-13. Leaderboard with actual metrics, confusion matrices
  await page.getByRole('button', { name: 'Open leaderboard' }).click()
  await expect(page.getByText('Selection basis')).toBeVisible()
  for (const name of ['Dummy (majority class)', 'Logistic Regression', 'Random Forest', 'XGBoost', 'Isolation Forest']) {
    await expect(page.locator('td', { hasText: name }).first()).toBeVisible()
  }
  await shot(page, '09_leaderboard')
  await page.getByText('evaluate →').first().click()
  await expect(page.getByText(/Confusion matrix/)).toBeVisible()
  await shot(page, '10_model_evaluation')

  // 14. Threshold changes update metrics
  await page.getByRole('tab', { name: 'Threshold studio' }).click()
  const metrics = page.getByTestId('threshold-metrics')
  const before = await metrics.innerText()
  await page.getByTestId('threshold-slider').fill('0')
  await expect(metrics).not.toHaveText(before)
  await shot(page, '11_threshold_studio')

  // Anomaly explorer
  await page.goto('/anomalies')
  await page.getByTestId('filter-FN').click()
  await expect(page.getByRole('button', { name: 'Export CSV' })).toBeVisible()
  await shot(page, '12_anomaly_explorer')

  // 15. Research report export
  await page.goto('/reports')
  const expSelect = page.getByLabel('Experiment to include')
  const opts = await expSelect.locator('option').allTextContents()
  if (opts.length > 1) await expSelect.selectOption({ index: 1 })
  await page.getByTestId('generate-report').click()
  await expect(page.getByText('Bundle contents')).toBeVisible({ timeout: 300_000 })
  await expect(page.getByText('report.md')).toBeVisible()
  await page.getByTestId('export-report').click()
  await expect(page.getByText(/files copied to/)).toBeVisible({ timeout: 120_000 })
  await shot(page, '13_reports')

  // 16-18. Different schema → new isolated session, no stale data
  const firstSchema = (await (await request.get('/api/session')).json()).schema.hash
  await page.goto('/datasets')
  await uploadViaBrowse(page, loginEventsFixture())
  await expect(page.getByTestId('active-dataset')).toContainText('ftad_fixture_login_events.csv')
  const sess = await (await request.get('/api/session')).json()
  expect(sess.schema).toBeNull()
  await page.goto('/explore')
  await expect(page.getByText('Confirm the schema mapping first')).toBeVisible()
  await page.goto('/benchmark')
  await expect(page.getByText('Confirm the schema mapping first')).toBeVisible()
  await page.goto('/schema')
  await expect(page.getByTestId('role-is_anomaly')).toHaveValue('target')
  await page.getByTestId('confirm-schema').click()
  await expect(page.getByText('Schema confirmed').first()).toBeVisible()
  await page.goto('/explore')
  await expect(page.getByTestId('overview-stats')).toContainText('1,500')
  await expect(page.getByTestId('overview-stats')).not.toContainText(ds.n_rows.toLocaleString('en-US'))
  const newSchema = (await (await request.get('/api/session')).json()).schema.hash
  expect(newSchema).not.toBe(firstSchema)

  // 19. No fake metrics: the new dataset has no runs, and the leaderboard says so
  await page.goto('/benchmark')
  await expect(page.getByText('No completed runs yet')).toBeVisible()

  // 20. Stable: no uncaught console errors
  expect(consoleErrors.filter((e) => !e.includes('favicon'))).toEqual([])
})
