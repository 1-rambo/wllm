import { chromium } from 'playwright';

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', msg => console.log('[browser]', msg.type(), msg.text()));
page.on('pageerror', err => console.log('[pageerror]', err.message));
page.setDefaultTimeout(120000);
await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle' });
await page.getByRole('button', { name: 'Load Dataset' }).click();
await page.waitForFunction(() => document.body.innerText.includes('Total loaded tasks: 231'));

async function setSiteChecked(site, desired) {
  const checkbox = page.locator('.site-list .checkbox-row').evaluateAll((labels, { site, desired }) => {
    const target = labels.find(label => label.textContent?.trim() === site);
    if (!target) return false;
    const input = target.querySelector('input');
    if (!input) return false;
    if (input.checked !== desired) input.click();
    return input.checked;
  }, { site, desired });
  return checkbox;
}

await setSiteChecked('shopping_admin', false);
await setSiteChecked('shopping', false);
await setSiteChecked('gitlab', false);
await setSiteChecked('reddit', true);
await page.waitForFunction(() => document.body.innerText.includes('Filtered tasks: 11'));

const numberInputs = page.locator('input[type="number"]');
await numberInputs.nth(0).fill('11');
await numberInputs.nth(1).fill('48');

const pageCtxInput = page.locator('label.checkbox-row', { hasText: 'Use preloaded page context in shared prefix' }).locator('input');
if (!(await pageCtxInput.isChecked())) {
  await pageCtxInput.check();
}

await page.getByRole('button', { name: 'Run Exp1' }).click();
await page.waitForFunction(() => document.body.innerText.includes('[WebArena] TTFT flat/tree='), { timeout: 1800000 });
const summary = await page.locator('.summary-list').innerText();
const table = await page.locator('.metrics-table').innerText();
const logs = await page.locator('.log-view').innerText();
console.log('===SUMMARY===\n' + summary);
console.log('===TABLE===\n' + table);
console.log('===LOGS===\n' + logs);
await browser.close();
