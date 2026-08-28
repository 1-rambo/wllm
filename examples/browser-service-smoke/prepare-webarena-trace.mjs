import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);

const required = (name) => {
  if (!args[name]) throw new Error(`Missing --${name}=...`);
  return args[name];
};

const input = path.resolve(required('input'));
const output = path.resolve(required('output'));
const contextsFile = args.contexts ? path.resolve(args.contexts) : null;
const taskCount = Math.max(1, Number(args['task-count'] ?? 80));
const clientCount = Math.max(2, Number(args['client-count'] ?? 4));
const switchIntervalMs = Math.max(1, Number(args['switch-interval-ms'] ?? 750));
const preloadIntervalMs = Math.max(
  0,
  Number(args['preload-interval-ms'] ?? 100)
);
const resumeDelayMs = Math.max(0, Number(args['resume-delay-ms'] ?? 5000));
const outputTokens = Math.max(1, Number(args['output-tokens'] ?? 32));
const maxContextChars = Math.max(0, Number(args['max-context-chars'] ?? 6000));
const sites = new Set(
  String(args.sites ?? 'shopping_admin,shopping,gitlab,reddit')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
);
const inputBuffer = await readFile(input);
const inputSha256 = createHash('sha256').update(inputBuffer).digest('hex');
const raw = JSON.parse(inputBuffer.toString('utf8'));
const contextsBuffer = contextsFile ? await readFile(contextsFile) : null;
const contexts = contextsBuffer
  ? JSON.parse(contextsBuffer.toString('utf8'))
  : {};
const contextsSha256 = contextsBuffer
  ? createHash('sha256').update(contextsBuffer).digest('hex')
  : null;

const isReadOnlyTask = (task) => {
  if (!Array.isArray(task.sites) || task.sites.length !== 1) return false;
  if (!sites.has(task.sites[0])) return false;
  const evalTypes = task.eval?.eval_types ?? [];
  if (!evalTypes.includes('string_match')) return false;
  const intent = String(task.intent ?? '').toLowerCase();
  return ![
    ' add ',
    ' buy ',
    ' change ',
    ' comment ',
    ' create ',
    ' delete ',
    ' edit ',
    ' follow ',
    ' like ',
    ' post ',
    ' purchase ',
    ' remove ',
    ' send ',
    ' upload ',
    ' vote ',
  ].some((verb) => ` ${intent} `.includes(verb));
};

const candidates = raw
  .filter(isReadOnlyTask)
  .sort(
    (left, right) =>
      String(left.sites[0]).localeCompare(String(right.sites[0])) ||
      Number(left.task_id) - Number(right.task_id)
  );
const buckets = new Map();
for (const task of candidates) {
  const site = task.sites[0];
  const bucket = buckets.get(site) ?? [];
  bucket.push(task);
  buckets.set(site, bucket);
}
const orderedSites = [...sites].filter((site) => buckets.has(site));
const selected = [];
let cursor = 0;
while (selected.length < taskCount && orderedSites.length > 0) {
  const site = orderedSites[cursor % orderedSites.length];
  const bucket = buckets.get(site);
  if (bucket.length > 0) selected.push(bucket.shift());
  if (bucket.length === 0) {
    orderedSites.splice(cursor % orderedSites.length, 1);
  } else {
    cursor += 1;
  }
}
if (selected.length === 0) {
  throw new Error('No eligible single-site string-match tasks were found');
}

const clients = Array.from({ length: clientCount }, (_, index) => ({
  id: `tab-${index}`,
}));
const siteToClient = Object.fromEntries(
  [...sites].map((site, index) => [site, index % clientCount])
);
const events = clients.map((_, client) => ({
  atMs: 0,
  action: 'lifecycle',
  client,
  state: 'hidden',
}));
const selectedRecords = [];
for (let index = 0; index < selected.length; index++) {
  const task = selected[index];
  const site = task.sites[0];
  const client = siteToClient[site];
  const preloadAtMs = index * preloadIntervalMs;
  const contextRecord =
    contexts[String(task.task_id)] ??
    contexts[`webarena-${task.task_id}`] ??
    contexts[task.id] ??
    {};
  const pageContext = String(contextRecord.pageContext ?? '').slice(
    0,
    maxContextChars
  );
  const contextPrompt = [
    `WebArena task ${task.task_id}.`,
    `Site: ${site}.`,
    `Start URL: ${task.start_url ?? ''}.`,
    pageContext ? `Observed page context:\n${pageContext}` : '',
    'Retain this page context for a later foreground request.',
  ]
    .filter(Boolean)
    .join('\n');
  events.push({
    atMs: preloadAtMs,
    action: 'request',
    client,
    session: `task-${task.task_id}`,
    parent: 'root',
    prompt: contextPrompt,
    nPredict: 0,
    phase: 'background-preload',
    sourceTaskId: task.task_id,
    site,
  });
  selectedRecords.push({ task, client });
}

let activeClient = null;
const resumeBaseMs = selected.length * preloadIntervalMs + resumeDelayMs;
for (let index = 0; index < selectedRecords.length; index++) {
  const { task, client } = selectedRecords[index];
  const site = task.sites[0];
  const atMs = resumeBaseMs + index * switchIntervalMs;
  if (activeClient !== client) {
    if (activeClient !== null) {
      events.push({
        atMs,
        action: 'lifecycle',
        client: activeClient,
        state: 'hidden',
      });
    }
    events.push({ atMs, action: 'lifecycle', client, state: 'active' });
    activeClient = client;
  }
  events.push({
    atMs,
    action: 'request',
    client,
    session: `task-${task.task_id}`,
    parent: 'previous',
    prompt: [
      `User intent: ${task.intent}`,
      'Return a concise answer grounded in the retained page context.',
    ].join('\n'),
    nPredict: outputTokens,
    phase: 'foreground-resume',
    sourceTaskId: task.task_id,
    site,
  });
  events.push({
    atMs: atMs + 1,
    action: 'interaction',
    client,
    samples: 1,
    intervalMs: 0,
  });
}

const trace = {
  schemaVersion: 'wllm.browser-service.trace.v1',
  traceId:
    args['trace-id'] ??
    `webarena-readonly-${selected.length}tasks-${clientCount}tabs`,
  source: {
    name: 'WebArena test.raw.json',
    inputFile: path.basename(input),
    inputSha256,
    sourceRevision: args['source-revision'] ?? null,
    contextsFile: contextsFile ? path.basename(contextsFile) : null,
    contextsSha256,
    selection:
      'site-balanced single-site string_match tasks with state-changing intents excluded',
    lifecycleConstruction:
      'one logical tab per site, background page-context preload, then deterministic foreground tab returns',
    siteToClient,
    taskIds: selected.map((task) => task.task_id),
  },
  clients,
  events,
};

await writeFile(output, `${JSON.stringify(trace, null, 2)}\n`);
console.log(
  JSON.stringify({
    output,
    traceId: trace.traceId,
    taskCount: selected.length,
    clientCount,
    eventCount: events.length,
    sourceSha256: inputSha256,
  })
);
