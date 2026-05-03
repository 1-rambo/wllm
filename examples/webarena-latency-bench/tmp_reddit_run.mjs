import { chromium } from 'playwright';

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', msg => console.log('[browser]', msg.type(), msg.text()));
page.on('pageerror', err => console.log('[pageerror]', err.message));
await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle' });
console.log('title=', await page.title());
console.log('content chars=', (await page.content()).length);
await browser.close();
