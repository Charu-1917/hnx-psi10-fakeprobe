// Single source of truth: demo_results/*.json (repo root). Copies them into src/demo/ for the offline fallback.
// src/demo/ is generated (gitignored) - never edit it by hand.
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const src = join(here, '..', '..', 'demo_results')
const dst = join(here, '..', 'src', 'demo')
if (!existsSync(src)) {
  console.warn('sync-demo: demo_results/ not found, skipping (bundled offline fallback will be empty)')
  process.exit(0)
}
rmSync(dst, { recursive: true, force: true })
mkdirSync(dst, { recursive: true })
for (const f of readdirSync(src).filter((n) => n.endsWith('.json'))) cpSync(join(src, f), join(dst, f))
console.log('sync-demo: copied', readdirSync(dst).join(', '))
