import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

function getArg(name, fallback = undefined) {
  const idx = process.argv.indexOf(name);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

function requireArg(name) {
  const value = getArg(name);
  if (!value) {
    throw new Error(`Missing required argument: ${name}`);
  }
  return value;
}

function normalizeWhitespace(text) {
  return String(text ?? '').replace(/\s+/g, ' ').trim();
}

function renderUrl(template, taskSites, configEnvs) {
  let rendered = template;
  for (const site of taskSites) {
    const env = configEnvs[site];
    if (!env || !env.urls?.length) continue;
    const placeholder = `__${site.toUpperCase()}__`;
    if (rendered.startsWith(placeholder)) {
      rendered = rendered.replace(placeholder, env.urls[env.active_url_idx ?? 0] || env.urls[0]);
      break;
    }
  }
  return rendered;
}

async function extractPageInfo(page, url, maxLinks) {
  await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
  return await page.evaluate((maxLinksInPage) => {
    const normalize = (text) => String(text ?? '').replace(/\s+/g, ' ').trim();
    const title = document.title ? normalize(document.title) : '';
    const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
      .slice(0, 12)
      .map((node) => normalize(node.textContent))
      .filter(Boolean);
    const bodyText = normalize(document.body?.innerText ?? '');
    const origin = window.location.origin;
    const links = [];
    const seen = new Set();
    for (const anchor of Array.from(document.querySelectorAll('a[href]'))) {
      const href = anchor.getAttribute('href') || '';
      if (!href || href.startsWith('#') || href.startsWith('javascript:') || href.startsWith('mailto:')) continue;
      try {
        const abs = new URL(href, window.location.href);
        if (abs.origin !== origin) continue;
        abs.hash = '';
        abs.search = '';
        const normalized = abs.toString().replace(/\/$/, '');
        if (seen.has(normalized)) continue;
        seen.add(normalized);
        links.push(normalized);
        if (links.length >= maxLinksInPage) break;
      } catch {
        // ignore malformed URLs
      }
    }
    return {
      finalUrl: window.location.href,
      title,
      headings,
      bodyText,
      links,
      html: document.documentElement.outerHTML,
    };
  }, maxLinks);
}

async function loginShoppingAdmin(page, loginUrl, username, password) {
  await page.goto(loginUrl, { waitUntil: 'networkidle', timeout: 120000 });
  if (await page.locator('input[name="login[username]"]').count()) {
    await page.locator('input[name="login[username]"]').fill(username);
    await page.locator('input[name="login[password]"]').fill(password);
    await page.getByRole('button', { name: /sign in/i }).click();
    await page.waitForTimeout(3000);
  }
}

async function main() {
  const datasetPath = requireArg('--dataset');
  const configPath = requireArg('--config');
  const outputPath = requireArg('--output');
  const maxChars = Number(getArg('--max-chars', '12000'));
  const crawlMaxPages = Number(getArg('--crawl-max-pages', '4'));
  const crawlMaxLinksPerPage = Number(getArg('--crawl-max-links-per-page', '8'));
  const snapshotDir = getArg('--snapshot-dir', '');
  const executablePath = getArg('--chrome-path', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');

  const tasks = JSON.parse(await fs.readFile(datasetPath, 'utf-8'));
  const rawConfig = JSON.parse(await fs.readFile(configPath, 'utf-8'));
  const configEnvs = rawConfig.environments || {};

  const grouped = new Map();
  for (const task of tasks) {
    const renderedStartUrls = (task.startUrls || []).map((u) => renderUrl(u, task.sites || [task.site], configEnvs));
    const key = renderedStartUrls.join(' | ');
    const list = grouped.get(key) || [];
    list.push({ ...task, renderedStartUrls, pageContextKey: key });
    grouped.set(key, list);
  }

  const browser = await chromium.launch({
    executablePath,
    headless: true,
  });

  const taskContexts = {};
  for (const [key, groupedTasks] of grouped.entries()) {
    const first = groupedTasks[0];
    const site = first.site;
    const env = configEnvs[site] || {};
    const creds = env.credentials || {};
    const page = await browser.newPage();

    if (site === 'shopping_admin' && creds.username && creds.password) {
      await loginShoppingAdmin(page, env.urls?.[env.active_url_idx ?? 0] || first.renderedStartUrls[0], creds.username, creds.password);
    }

    const queue = [...first.renderedStartUrls];
    const visited = new Set();
    const blocks = [];

    while (queue.length && visited.size < crawlMaxPages) {
      const url = queue.shift();
      const normalized = String(url).replace(/\/$/, '');
      if (visited.has(normalized)) continue;
      visited.add(normalized);
      try {
        const info = await extractPageInfo(page, url, crawlMaxLinksPerPage);
        const parts = [`URL: ${info.finalUrl}`];
        if (info.title) parts.push(`Page Title: ${info.title}`);
        if (info.headings.length) parts.push(`Headings: ${info.headings.join(' | ')}`);
        if (info.bodyText) parts.push(`Visible Text: ${info.bodyText}`);
        blocks.push(parts.join('\n'));

        if (snapshotDir) {
          const siteDir = path.join(snapshotDir, site);
          await fs.mkdir(siteDir, { recursive: true });
          const slug = new URL(info.finalUrl).pathname.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'root';
          await fs.writeFile(path.join(siteDir, `${slug}.html`), info.html, 'utf-8');
        }

        for (const link of info.links) {
          const linkNorm = String(link).replace(/\/$/, '');
          if (!visited.has(linkNorm) && !queue.includes(link)) {
            queue.push(link);
          }
        }
      } catch (err) {
        blocks.push(`URL: ${url}\nFetch Error: ${err instanceof Error ? err.message : String(err)}`);
      }
      if (blocks.join('\n\n').length >= maxChars) break;
    }

    const merged = blocks.join('\n\n').slice(0, maxChars);
    for (const task of groupedTasks) {
      taskContexts[task.id] = {
        renderedStartUrls: task.renderedStartUrls,
        pageContextKey: task.pageContextKey,
        pageContext: merged,
      };
    }

    await page.close();
  }

  await browser.close();
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(taskContexts, null, 2), 'utf-8');
  console.log(`Wrote ${Object.keys(taskContexts).length} task page contexts to ${outputPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
