import type { WebArenaTask } from './types';

type SitePageContexts = Record<string, string>;
type TaskPageContextRecord = {
  pageContext?: string;
  pageContextKey?: string;
  renderedStartUrls?: string[];
};
type TaskPageContexts = Record<string, TaskPageContextRecord>;

async function loadSitePageContexts(path: string): Promise<SitePageContexts> {
  const sidecarPath = path.replace(/\/[^/]+$/, '/site_page_contexts.json');
  try {
    const res = await fetch(sidecarPath);
    if (!res.ok) {
      return {};
    }
    return (await res.json()) as SitePageContexts;
  } catch {
    return {};
  }
}

async function loadTaskPageContexts(path: string): Promise<TaskPageContexts> {
  const sidecarPath = path.replace(/\/[^/]+$/, '/task_page_contexts.json');
  try {
    const res = await fetch(sidecarPath);
    if (!res.ok) {
      return {};
    }
    return (await res.json()) as TaskPageContexts;
  } catch {
    return {};
  }
}

export async function loadWebArenaRetrieveSubset(path: string): Promise<WebArenaTask[]> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load WebArena subset: ${path} (${res.status})`);
  }
  const tasks = (await res.json()) as WebArenaTask[];
  const taskContexts = await loadTaskPageContexts(path);
  const pageContexts = await loadSitePageContexts(path);
  return tasks.map((task) => ({
    ...task,
    renderedStartUrls: taskContexts[task.id]?.renderedStartUrls ?? task.renderedStartUrls,
    pageContext: taskContexts[task.id]?.pageContext ?? pageContexts[task.site] ?? '',
    pageContextKey: taskContexts[task.id]?.pageContextKey
      ?? task.pageContextKey
      ?? task.startUrls.join(' | '),
  }));
}

export function summarizeSiteCounts(tasks: WebArenaTask[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const task of tasks) {
    counts[task.site] = (counts[task.site] ?? 0) + 1;
  }
  return counts;
}

export function filterTasksBySites(tasks: WebArenaTask[], sites: string[]): WebArenaTask[] {
  if (!sites.length) return tasks.slice();
  const wanted = new Set(sites);
  return tasks.filter((task) => wanted.has(task.site));
}
