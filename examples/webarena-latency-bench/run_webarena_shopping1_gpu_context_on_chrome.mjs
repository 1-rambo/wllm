import { chromium } from 'playwright';
const browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
const page = await browser.newPage();
page.on('console', msg => {
  const text = msg.text();
  if (text.includes('offloaded') || text.includes('using device') || text.includes('[WebArena]')) console.log('[browser]', text);
});
page.on('pageerror', err => console.log('[pageerror]', err.message));
page.setDefaultTimeout(0);
await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle' });
await page.getByRole('button', { name: 'Load Dataset' }).click();
await page.waitForFunction(() => document.body.innerText.includes('Total loaded tasks: 231'));
await page.locator('.site-list .checkbox-row').evaluateAll((labels) => {
  for (const label of labels) {
    const text = (label.textContent || '').trim();
    const input = label.querySelector('input');
    if (!input) continue;
    const shouldBeChecked = text === 'shopping';
    if (input.checked !== shouldBeChecked) input.click();
  }
});
await page.waitForFunction(() => document.body.innerText.includes('Filtered tasks: 81'));
const numberInputs = page.locator('input[type="number"]');
await numberInputs.nth(0).fill('1');
await numberInputs.nth(1).fill('8');
const pageCtxInput = page.locator('label.checkbox-row', { hasText: 'Use preloaded page context in shared prefix' }).locator('input');
if (!(await pageCtxInput.isChecked())) await pageCtxInput.click();
await page.getByRole('button', { name: 'Run Exp1' }).click();
await page.waitForFunction(() => document.body.innerText.includes('[WebArena] TTFT flat/tree='), { timeout: 3600000 });
const summary = await page.locator('.summary-list').innerText();
console.log(summary);
await browser.close();
