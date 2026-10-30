// SYNTHETIC SOFTWARE-TESTING FIXTURES — generated for UI tests only. They are not empirical
// data and must never be reported as research results.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

function rng(seed: number) {
  let s = seed >>> 0
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32)
}
function normal(r: () => number) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r())
}

/** Card-transaction-shaped fixture: Time, V1..V8, Amount, Class. */
export function transactionsFixture(n = 6000, prevalence = 0.03): string {
  const r = rng(7)
  const rows = ['Time,V1,V2,V3,V4,V5,V6,V7,V8,Amount,Class']
  let t = 0
  for (let i = 0; i < n; i++) {
    t += Math.floor(r() * 30)
    const y = r() < prevalence ? 1 : 0
    const v = Array.from({ length: 8 }, (_, j) => (normal(r) - (j < 3 ? 1.6 * y : 0)).toFixed(5))
    const amount = (Math.exp(3 + 1.1 * normal(r)) * (1 + 0.4 * y)).toFixed(2)
    rows.push([t, ...v, amount, y].join(','))
  }
  const file = path.join(os.tmpdir(), `ftad_fixture_transactions_${n}.csv`)
  fs.writeFileSync(file, rows.join('\n'))
  return file
}

/** A differently shaped fixture (other names, categoricals, timestamps, entity IDs). */
export function loginEventsFixture(n = 1500): string {
  const r = rng(11)
  const rows = ['event_id,customer_id,event_time,channel,country,login_attempts,session_value,risk_score_a,is_anomaly']
  const start = Date.UTC(2024, 0, 1)
  for (let i = 0; i < n; i++) {
    const y = r() < 0.08
    const ts = new Date(start + i * 7 * 60_000).toISOString().replace('T', ' ').slice(0, 19)
    rows.push([
      `E${String(i).padStart(6, '0')}`, Math.floor(r() * 200), ts, ['web', 'app', 'branch'][Math.floor(r() * 3)],
      ['DE', 'FR', 'NL', 'PL'][Math.floor(r() * 4)], Math.floor(r() * (y ? 6 : 2)), (r() * 120).toFixed(2), (normal(r) + (y ? 1.2 : 0)).toFixed(4), y ? 'yes' : 'no',
    ].join(','))
  }
  const file = path.join(os.tmpdir(), 'ftad_fixture_login_events.csv')
  fs.writeFileSync(file, rows.join('\n'))
  return file
}
