import { chromium } from 'playwright';
const browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
const page = await browser.newPage();
await page.goto('http://localhost:7780/admin', { waitUntil: 'networkidle' });
await page.locator('input[name="login[username]"]').fill('codexadmin');
await page.locator('input[name="login[password]"]').fill('Codex123!');
await page.getByRole('button', { name: /sign in/i }).click();
await page.waitForTimeout(5000);
console.log('url=', page.url());
console.log('title=', await page.title());
const body = await page.locator('body').innerText();
for (const needle of ['Dashboard','Stores','Reports','Welcome, please sign in','Invalid login or password.','The account sign-in was incorrect']) {
  console.log(needle, body.includes(needle));
}
await browser.screenshot({ path: '/Users/rambo/Desktop/wllama-webgpu/examples/webarena-latency-bench/admin-login-check.png', fullPage: true });
await browser.close();
