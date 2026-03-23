import type { BenchmarkTask, PrefixConfig } from './types';

export function buildSharedPrefix(task: BenchmarkTask, cfg: PrefixConfig): string {
  const chunks: string[] = [];

  chunks.push(cfg.systemPrompt.trim());

  if (cfg.includeStartUrl && task.startUrl) {
    chunks.push(`START_URL: ${task.startUrl}`);
  }

  if (cfg.includeOpenUrls && task.openUrls.length > 0) {
    const line = task.openUrls.map((u, idx) => `TAB_${idx + 1}: ${u}`).join('\n');
    chunks.push(`OPEN_TABS:\n${line}`);
  }

  const templ = cfg.webContentTemplate;
  const webContext = templ
    .split('{{DATASET}}')
    .join(task.dataset)
    .split('{{EVAL}}')
    .join(task.evalTypes.join(',') || 'unknown');
  chunks.push(webContext);

  return chunks.filter(Boolean).join('\n\n');
}

export function buildTaskPrompt(task: BenchmarkTask): string {
  const intent = task.intent.replace(/\s+/g, ' ').trim();
  return [
    'TASK:',
    intent,
    '',
    'Rules:',
    '- Use the shared web context above as primary evidence.',
    '- Return concise final answer only.',
  ].join('\n');
}

export function buildPrefixKey(task: BenchmarkTask, cfg: PrefixConfig): string {
  return `${task.dataset}|${task.templateId || 'none'}|${buildSharedPrefix(task, cfg)}`;
}

export function buildPrefixAnchorPrompt(sharedPrefix: string): string {
  return [
    'Memorize the following context for follow-up tasks.',
    '',
    sharedPrefix,
  ].join('\n');
}
