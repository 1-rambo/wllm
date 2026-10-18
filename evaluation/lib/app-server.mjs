import { createReadStream } from "node:fs";
import { access, readFile, stat } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { assetsRoot, distRoot, implRoot, modelsRoot } from "./paths.mjs";

const MIME = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".svg", "image/svg+xml"],
  [".wasm", "application/wasm"],
  [".gguf", "application/octet-stream"],
  [".map", "application/json; charset=utf-8"],
  [".woff2", "font/woff2"],
  [".txt", "text/plain; charset=utf-8"]
]);

export async function createEvaluationServer(options = {}) {
  const roots = await resolveRoots();
  const server = http.createServer((request, response) => {
    void handleRequest(request, response, roots).catch((error) => {
      response.statusCode = 500;
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ ok: false, error: String(error?.stack ?? error) }));
    });
  });
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 0;
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, resolve);
  });
  const address = server.address();
  return {
    server,
    host,
    port: address.port,
    url: `http://${host}:${address.port}`,
    roots,
    close: () =>
      new Promise((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
        server.closeAllConnections?.();
      })
  };
}

async function handleRequest(request, response, roots) {
  const url = new URL(request.url ?? "/", "http://127.0.0.1");
  if (request.method !== "GET" && request.method !== "HEAD") {
    response.statusCode = 405;
    response.end("method not allowed");
    return;
  }
  setSecurityHeaders(response);
  if (url.pathname === "/health") {
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ ok: true, roots }));
    return;
  }
  const mapping = mapRequest(url.pathname, roots);
  if (mapping === null) {
    response.statusCode = 404;
    response.end("not found");
    return;
  }
  const file = await safeFile(mapping.root, mapping.relative);
  if (file === null) {
    response.statusCode = 404;
    response.end("not found");
    return;
  }
  const info = await stat(file);
  if (!info.isFile()) {
    response.statusCode = 404;
    response.end("not found");
    return;
  }
  const extension = path.extname(file).toLowerCase();
  response.setHeader("content-type", MIME.get(extension) ?? "application/octet-stream");
  response.setHeader("cache-control", extension === ".gguf" ? "public, max-age=31536000, immutable" : "no-store");
  response.setHeader("accept-ranges", "bytes");
  if (extension === ".html") {
    response.setHeader("content-type", "text/html; charset=utf-8");
    const html = await injectEvaluationShim(await readFile(file, "utf8"), url.pathname);
    response.setHeader("content-length", Buffer.byteLength(html));
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    response.end(html);
    return;
  }
  const range = request.headers.range;
  if (typeof range === "string" && range.startsWith("bytes=")) {
    const [startText, endText] = range.slice(6).split("-", 2);
    const start = startText === "" ? Math.max(0, info.size - Number(endText)) : Number(startText);
    const end = endText === "" ? info.size - 1 : Math.min(info.size - 1, Number(endText));
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start > end) {
      response.statusCode = 416;
      response.setHeader("content-range", `bytes */${info.size}`);
      response.end();
      return;
    }
    response.statusCode = 206;
    response.setHeader("content-range", `bytes ${start}-${end}/${info.size}`);
    response.setHeader("content-length", String(end - start + 1));
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    createReadStream(file, { start, end }).pipe(response);
    return;
  }
  response.setHeader("content-length", String(info.size));
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  createReadStream(file).pipe(response);
}

async function resolveRoots() {
  const prepared = await readPreparedManifest();
  const artifexRoot = prepared?.artifexDist ?? path.join(implRoot, "vendor", "artifex-webgame", "dist");
  const tavernRoot = prepared?.tavernRoot ?? path.join(implRoot, "vendor", "holtwick-tavern");
  await Promise.all([
    access(path.join(distRoot, "engines", "wllm.html")),
    access(path.join(distRoot, "engines", "wllama.html")),
    access(path.join(artifexRoot, "game.html")),
    access(path.join(tavernRoot, "index.html"))
  ]);
  return { distRoot, modelsRoot, assetsRoot, artifexRoot, tavernRoot };
}

async function readPreparedManifest() {
  try {
    const text = await readFile(path.join(implRoot, "artifacts", "evaluation", "prepared-apps.json"), "utf8");
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function mapRequest(pathname, roots) {
  if (pathname === "/render/pressure.html") {
    return { root: roots.assetsRoot, relative: "pressure.html" };
  }
  if (pathname === "/evaluation/app-adapter.js") {
    return { root: roots.assetsRoot, relative: "app-adapter.js" };
  }
  if (pathname === "/evaluation/webllm-stub.js") {
    return { root: roots.assetsRoot, relative: "webllm-stub.js" };
  }
  if (pathname === "/evaluation/phaser-3.80.1.min.js") {
    return { root: roots.assetsRoot, relative: "phaser-3.80.1.min.js" };
  }
  if (pathname.startsWith("/engines/")) {
    return { root: roots.distRoot, relative: `engines/${pathname.slice("/engines/".length)}` };
  }
  if (pathname.startsWith("/assets/")) {
    return { root: roots.distRoot, relative: `assets/${pathname.slice("/assets/".length)}` };
  }
  if (pathname.startsWith("/models/")) {
    return { root: roots.modelsRoot, relative: pathname.slice("/models/".length) };
  }
  if (pathname.startsWith("/apps/artifex/")) {
    return { root: roots.artifexRoot, relative: pathname.slice("/apps/artifex/".length) };
  }
  if (pathname.startsWith("/apps/tavern/")) {
    return { root: roots.tavernRoot, relative: pathname.slice("/apps/tavern/".length) };
  }
  if (pathname === "/") {
    return { root: roots.assetsRoot, relative: "index.html" };
  }
  return null;
}

async function safeFile(root, relative) {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, relative);
  if (resolved !== resolvedRoot && !resolved.startsWith(`${resolvedRoot}${path.sep}`)) {
    return null;
  }
  try {
    await access(resolved);
    return resolved;
  } catch {
    return null;
  }
}

async function injectEvaluationShim(html, pathname) {
  let output = html;
  const isApplicationPage =
    pathname.startsWith("/apps/artifex/") || pathname.startsWith("/apps/tavern/");
  if (!isApplicationPage) return output;
  if (pathname.startsWith("/apps/tavern/")) {
    output = output.replace(
      /https:\/\/esm\.run\/@mlc-ai\/web-llm@[^"']+/g,
      "/evaluation/webllm-stub.js"
    );
    output = output.replace(
      /https:\/\/cdn\.jsdelivr\.net\/npm\/phaser@3\.80\.1\/dist\/phaser\.min\.js/g,
      "/evaluation/phaser-3.80.1.min.js"
    );
  }
  const tag = '    <script type="module" src="/evaluation/app-adapter.js"></script>\n';
  if (!output.includes("/evaluation/app-adapter.js")) {
    const bodyEnd = output.lastIndexOf("</body>");
    if (bodyEnd >= 0) {
      output = `${output.slice(0, bodyEnd)}${tag}${output.slice(bodyEnd)}`;
    } else {
      output = `${tag}${output}`;
    }
  }
  return output;
}

function setSecurityHeaders(response) {
  response.setHeader("cross-origin-opener-policy", "same-origin");
  response.setHeader("cross-origin-embedder-policy", "credentialless");
  response.setHeader("cross-origin-resource-policy", "cross-origin");
  response.setHeader("x-content-type-options", "nosniff");
}
