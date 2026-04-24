import type { HellaSwagItem, MMLUItem } from './types';

const LOCAL_MMLU_SUBJECTS = [
  'abstract_algebra',
  'anatomy',
  'astronomy',
  // 'business_ethics',
  // 'clinical_knowledge',
  // 'college_biology',
  // 'college_chemistry',
  // 'college_computer_science',
  // 'college_mathematics',
  // 'college_medicine',
  // 'college_physics',
  // 'computer_security',
  // 'conceptual_physics',
  // 'econometrics',
  // 'electrical_engineering',
  // 'elementary_mathematics',
  // 'formal_logic',
  // 'global_facts',
  // 'high_school_biology',
  // 'high_school_chemistry',
  // 'high_school_computer_science',
  // 'high_school_european_history',
  // 'high_school_geography',
  // 'high_school_government_and_politics',
  // 'high_school_macroeconomics',
  // 'high_school_mathematics',
  // 'high_school_microeconomics',
  // 'high_school_physics',
  // 'high_school_psychology',
  // 'high_school_statistics',
  // 'high_school_us_history',
  // 'high_school_world_history',
  // 'human_aging',
  // 'human_sexuality',
  // 'international_law',
  // 'jurisprudence',
  // 'logical_fallacies',
  // 'machine_learning',
  // 'management',
  // 'marketing',
  // 'medical_genetics',
  // 'miscellaneous',
  // 'moral_disputes',
  // 'moral_scenarios',
  // 'nutrition',
  // 'philosophy',
  // 'prehistory',
  // 'professional_accounting',
  // 'professional_law',
  // 'professional_medicine',
  // 'professional_psychology',
  // 'public_relations',
  // 'security_studies',
  // 'sociology',
  // 'us_foreign_policy',
  // 'virology',
  // 'world_religions',
] as const;

export function getAllLocalMmluSubjects(): string[] {
  return [...LOCAL_MMLU_SUBJECTS];
}

function sanitizeText(text: unknown): string {
  return String(text ?? '').replace(/\s+/g, ' ').trim();
}

function parseHellaRow(raw: Record<string, unknown>, idx: number): HellaSwagItem | null {
  const ctx = sanitizeText(raw.ctx);
  const endings = Array.isArray(raw.endings) ? raw.endings.map((x) => sanitizeText(x)) : [];
  const label = Number(raw.label);

  if (!ctx || endings.length < 4 || !Number.isFinite(label)) return null;
  return {
    id: `hella-${raw.ind ?? idx}`,
    ctx,
    endings: [endings[0], endings[1], endings[2], endings[3]],
    label: Math.max(0, Math.min(3, label)) as 0 | 1 | 2 | 3,
  };
}

export async function loadRealHellaSwag(path: string, maxItems: number): Promise<HellaSwagItem[]> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load HellaSwag jsonl: ${path} (${res.status})`);
  }
  const text = await res.text();
  const lines = text.split('\n');
  const out: HellaSwagItem[] = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    try {
      const row = JSON.parse(line) as Record<string, unknown>;
      const parsed = parseHellaRow(row, i);
      if (parsed) out.push(parsed);
      if (out.length >= maxItems) break;
    } catch {
      // Skip malformed lines and continue.
    }
  }

  return out;
}

export async function getHellaSwagLineCount(path: string): Promise<number> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to count HellaSwag jsonl: ${path} (${res.status})`);
  }
  const text = await res.text();
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .length;
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuote = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuote && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuote = !inQuote;
      }
      continue;
    }
    if (ch === ',' && !inQuote) {
      out.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

async function loadMmluCsv(path: string, subject: string, limit: number, offset = 0): Promise<MMLUItem[]> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load MMLU CSV: ${path} (${res.status})`);
  }

  const text = await res.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  const out: MMLUItem[] = [];

  for (let i = 0; i < lines.length; i += 1) {
    const cols = splitCsvLine(lines[i]);
    if (cols.length < 6) continue;

    const question = sanitizeText(cols[0]);
    const c0 = sanitizeText(cols[1]);
    const c1 = sanitizeText(cols[2]);
    const c2 = sanitizeText(cols[3]);
    const c3 = sanitizeText(cols[4]);
    const ansRaw = sanitizeText(cols[5]).toUpperCase();
    const answer = ['A', 'B', 'C', 'D'].indexOf(ansRaw);

    if (!question || !c0 || !c1 || !c2 || !c3 || answer < 0) continue;

    out.push({
      id: `${subject}-${offset + out.length + 1}`,
      subject,
      question,
      choices: [c0, c1, c2, c3],
      answerIndex: answer as 0 | 1 | 2 | 3,
    });

    if (out.length >= limit) break;
  }

  return out;
}

async function countCsvRows(path: string): Promise<number> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to count MMLU CSV: ${path} (${res.status})`);
  }
  const text = await res.text();
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .length;
}

export async function getLocalMmluSubjectCounts(subject: string): Promise<{ valCount: number; testCount: number }> {
  const valPath = `/datasets/mmlu/data/val/${subject}_val.csv`;
  const testPath = `/datasets/mmlu/data/test/${subject}_test.csv`;
  const [valCount, testCount] = await Promise.all([
    countCsvRows(valPath),
    countCsvRows(testPath),
  ]);
  return { valCount, testCount };
}

export async function loadRealMmluFromLocal(subject: string, shots: number, evalCount: number): Promise<MMLUItem[]> {
  const valPath = `/datasets/mmlu/data/val/${subject}_val.csv`;
  const testPath = `/datasets/mmlu/data/test/${subject}_test.csv`;

  const shotRows = shots > 0
    ? await loadMmluCsv(valPath, subject, shots, 0)
    : [];
  const evalRows = evalCount > 0
    ? await loadMmluCsv(testPath, subject, evalCount, shotRows.length)
    : [];

  return [...shotRows, ...evalRows];
}
