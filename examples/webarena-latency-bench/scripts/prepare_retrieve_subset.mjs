import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
const input = path.resolve(args.input ?? '');
const output = path.resolve(
  args.output ??
    'examples/webarena-latency-bench/public/datasets/webarena/retrieve_info_subset.json'
);
if (!args.input) throw new Error('Missing --input=/path/to/test.raw.json');
const sites = new Set(
  String(args.sites ?? 'shopping_admin,shopping,gitlab,reddit')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
);
const limit = Math.max(1, Number(args.limit ?? 240));
const buffer = await readFile(input);
const raw = JSON.parse(buffer.toString('utf8'));
const mutatingTerms = [
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
];
const candidates = raw
  .filter((task) => {
    if (!Array.isArray(task.sites) || task.sites.length !== 1) return false;
    if (!sites.has(task.sites[0])) return false;
    if (!(task.eval?.eval_types ?? []).includes('string_match')) return false;
    const intent = ` ${String(task.intent ?? '').toLowerCase()} `;
    return !mutatingTerms.some((term) => intent.includes(term));
  })
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
const selectedTasks = [];
let cursor = 0;
while (selectedTasks.length < limit && orderedSites.length > 0) {
  const siteIndex = cursor % orderedSites.length;
  const site = orderedSites[siteIndex];
  const bucket = buckets.get(site);
  selectedTasks.push(bucket.shift());
  if (bucket.length === 0) {
    orderedSites.splice(siteIndex, 1);
  } else {
    cursor += 1;
  }
}
const selected = selectedTasks.map((task) => ({
  id: `webarena-${task.task_id}`,
  taskId: task.task_id,
  site: task.sites[0],
  sites: task.sites,
  startUrls: [task.start_url],
  intent: task.intent,
  intentTemplateId: task.intent_template_id,
  intentTemplate: task.intent_template,
  instantiationDict: task.instantiation_dict,
  revision: task.revision ?? null,
  evaluation: task.eval,
}));
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(selected, null, 2)}\n`);
const sourceSha256 = createHash('sha256').update(buffer).digest('hex');
await writeFile(
  `${output}.meta.json`,
  `${JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      sourceFile: path.basename(input),
      sourceSha256,
      sourceRevision: args['source-revision'] ?? null,
      sites: [...sites],
      selection:
        'site-balanced single-site string_match tasks with state-changing intents excluded',
      selectedTaskIds: selected.map((task) => task.taskId),
    },
    null,
    2
  )}\n`
);
console.log(
  JSON.stringify({
    output,
    count: selected.length,
    sourceSha256,
  })
);
