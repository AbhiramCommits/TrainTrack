import { chromium } from 'playwright'
import { existsSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import path from 'node:path'

const BASE = process.env.SCREENSHOT_BASE_URL || 'http://127.0.0.1:5273'
const OUT = path.resolve(process.cwd(), '..', 'docs', 'screenshots')
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

async function main() {
  await mkdir(OUT, { recursive: true })
  const browser = await chromium.launch(
    existsSync(CHROME) ? { executablePath: CHROME } : {},
  )
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.on('console', (message) => {
    if (message.type() === 'error') console.log('[browser]', message.text())
  })

  await page.goto(BASE, { waitUntil: 'networkidle' })
  await page.waitForSelector('.runs-table tbody tr', { timeout: 20000 })
  await page.waitForTimeout(1200)
  await page.screenshot({ path: path.join(OUT, 'runs-dashboard.png'), fullPage: true })
  console.log('saved runs-dashboard.png')

  await page.locator('.leaderboard a').first().click()
  await page.waitForSelector('.chart-panel', { timeout: 20000 })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: path.join(OUT, 'run-detail.png'), fullPage: true })
  console.log('saved run-detail.png')

  await page.locator('.logs-panel').scrollIntoViewIfNeeded()
  await page.waitForTimeout(1000)
  await page.locator('.logs-panel').screenshot({ path: path.join(OUT, 'live-logs.png') })
  console.log('saved live-logs.png')

  await page.goto(`${BASE}/compare`, { waitUntil: 'networkidle' })
  await page.waitForSelector('.run-option')
  const options = page.locator('.run-option input')
  const count = Math.min(await options.count(), 3)
  for (let index = 0; index < count; index += 1) {
    await options.nth(index).check()
  }
  await page.waitForTimeout(2500)
  await page.screenshot({ path: path.join(OUT, 'compare.png'), fullPage: true })
  console.log('saved compare.png')

  await page.goto(`${BASE}/submit`, { waitUntil: 'networkidle' })
  await page.waitForSelector('form.form-panel')
  await page.screenshot({ path: path.join(OUT, 'submit-run.png'), fullPage: true })
  console.log('saved submit-run.png')

  await page.goto(`${BASE}/queue`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: path.join(OUT, 'queue-health.png'), fullPage: true })
  console.log('saved queue-health.png')

  await browser.close()
  process.exit(0)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
