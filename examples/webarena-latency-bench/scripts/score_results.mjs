import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=') || 'true'];
  })
);
if (!args.input) throw new Error('Missing --input=/path/to/report.json');
const input = path.resolve(args.input);
const output = path.resolve(
  args.output ?? path.join(path.dirname(input), 'accuracy.json')
);
const raw = JSON.parse(await readFile(input, 'utf8'));
const report = raw.report ?? raw;
const metrics = report.webarena
  ? [
      ...(report.webarena.sampleMetricsFlat ?? []),
      ...(report.webarena.sampleMetricsTree ?? []),
    ]
  : [];
const normalize = (value) =>
  String(value ?? '')
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}.+-]+/gu, ' ')
    .trim();
const includes = (output, answer) =>
  normalize(output).includes(normalize(answer));
const scoreOne = (metric) => {
  const evaluation = metric.referenceAnswers ?? {};
  const answers = evaluation.reference_answers ?? evaluation;
  const outputText = metric.outputText ?? '';
  if (answers.exact_match !== undefined) {
    return {
      supported: true,
      correct: normalize(outputText) === normalize(answers.exact_match),
      criterion: 'exact_match',
    };
  }
  if (Array.isArray(answers.must_include)) {
    return {
      supported: true,
      correct: answers.must_include.every((answer) =>
        includes(outputText, answer)
      ),
      criterion: 'must_include',
    };
  }
  if (answers.fuzzy_match !== undefined) {
    const candidates = Array.isArray(answers.fuzzy_match)
      ? answers.fuzzy_match
      : [answers.fuzzy_match];
    return {
      supported: true,
      correct: candidates.some((answer) => includes(outputText, answer)),
      criterion: 'normalized_contains_proxy',
      note: 'Use the official WebArena evaluator for final fuzzy-match accuracy.',
    };
  }
  return { supported: false, correct: false, criterion: 'unsupported' };
};
const scored = metrics.map((metric) => ({ ...metric, ...scoreOne(metric) }));
const grouped = new Map();
for (const row of scored) {
  const key = `${row.mode}\u0000${row.subject ?? 'unknown'}`;
  const bucket = grouped.get(key) ?? [];
  bucket.push(row);
  grouped.set(key, bucket);
}
const summary = [...grouped.entries()].map(([key, rows]) => {
  const [mode, site] = key.split('\u0000');
  const supported = rows.filter((row) => row.supported);
  const correct = supported.filter((row) => row.correct);
  return {
    mode,
    site,
    sampleCount: rows.length,
    scoredCount: supported.length,
    correctCount: correct.length,
    accuracy: supported.length ? correct.length / supported.length : null,
  };
});
await writeFile(
  output,
  `${JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      source: input,
      summary,
      samples: scored,
    },
    null,
    2
  )}\n`
);
console.log(JSON.stringify({ output, summary }));
