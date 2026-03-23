import shopping from '../../../../AgenticBrowserBenchmark/dataset_v2.0/shopping_dataset.json';
import gitlab from '../../../../AgenticBrowserBenchmark/dataset_v2.0/gitlab_dataset.json';
import cross from '../../../../AgenticBrowserBenchmark/dataset_v2.0/cross_dataset.json';
import type { BenchmarkTask, EvalType } from './types';

type RawReferenceAnswers = {
  fuzzy_match?: string[] | string;
  must_include?: string[] | string;
  required_values?: string[] | string;
};

type RawEval = {
  eval_types?: string[];
  reference_answers?: RawReferenceAnswers;
};

type RawTask = {
  task_id?: string | number;
  intent?: string;
  start_url?: string;
  open_url?: string[];
  require_reset?: boolean;
  intent_template_id?: string | number;
  eval?: RawEval;
};

function toEvalTypes(raw?: RawEval): EvalType[] {
  const values = raw?.eval_types ?? [];
  if (!values.length) {
    return ['unknown'];
  }
  return values.map((v) => {
    if (v === 'string_match' || v === 'program_html' || v === 'url_match') {
      return v;
    }
    return 'unknown';
  });
}

function normalizeTask(dataset: string, row: RawTask, index: number): BenchmarkTask {
  return {
    id: row.task_id != null ? String(row.task_id) : `${dataset}-${index + 1}`,
    dataset,
    intent: row.intent || '',
    startUrl: row.start_url,
    openUrls: Array.isArray(row.open_url) ? row.open_url.filter(Boolean) : [],
    requireReset: Boolean(row.require_reset),
    templateId: row.intent_template_id != null ? String(row.intent_template_id) : undefined,
    evalTypes: toEvalTypes(row.eval),
    referenceHints: extractReferenceHints(row.eval),
  };
}

function toStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((v) => String(v || '').trim()).filter(Boolean);
  }
  if (typeof value === 'string') {
    const s = value.trim();
    return s ? [s] : [];
  }
  return [];
}

function extractReferenceHints(raw?: RawEval): string[] {
  const ref = raw?.reference_answers;
  if (!ref) {
    return [];
  }
  const hints = [
    ...toStringList(ref.fuzzy_match),
    ...toStringList(ref.must_include),
    ...toStringList(ref.required_values),
  ];
  return Array.from(new Set(hints));
}

export function loadAllBenchmarkTasks(): BenchmarkTask[] {
  const allRows: Array<{ dataset: string; rows: RawTask[] }> = [
    { dataset: 'shopping', rows: shopping as RawTask[] },
    { dataset: 'gitlab', rows: gitlab as RawTask[] },
    { dataset: 'cross', rows: cross as RawTask[] },
  ];

  return allRows.flatMap(({ dataset, rows }) => rows.map((row, i) => normalizeTask(dataset, row, i)));
}
