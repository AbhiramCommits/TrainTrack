import { chromium } from 'playwright'
import { existsSync } from 'node:fs'
import path from 'node:path'

const BASE = process.env.SCREENSHOT_BASE_URL || 'http://127.0.0.1:5275'
const OUT = path.resolve(process.cwd(), '..', 'docs', 'screenshots')
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

const browser = await chromium.launch(
  existsSync(CHROME) ? { executablePath: CHROME } : {},
)
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
await page.goto(BASE, { waitUntil: 'networkidle' })
await page.waitForSelector('.runs-table tbody tr', { timeout: 20000 })
await page.waitForTimeout(1500)
await page.screenshot({ path: path.join(OUT, 'kubernetes-runs.png'), fullPage: true })
console.log('saved kubernetes-runs.png')
await page.locator('.runs-table tbody tr').first().click()
await page.waitForSelector('.chart-panel', { timeout: 20000 })
await page.waitForTimeout(2500)
await page.screenshot({ path: path.join(OUT, 'kubernetes-run-detail.png'), fullPage: true })
console.log('saved kubernetes-run-detail.png')
await browser.close()
process.exit(0)
