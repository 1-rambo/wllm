import type { InferenceBridgeOptions } from '../client/inference-bridge';

/**
 * Numeric knobs the engine page passes through to the coordinator worker.
 *
 * Keep the read and write paths on one list. A knob that exists in the
 * coordinator but is missing here is silently dropped at the page boundary.
 */
export const COORDINATOR_NUMERIC_OPTION_KEYS = [
  'nCtx',
  'nBatch',
  'nParallel',
  'nThreads',
  'nGpuLayers',
  'leaseBatchSize',
  'leaseHandoffBatchSize',
  'handoffPrefillTokens',
  'schedulerForegroundBurstUnits',
  'schedulerResidentBurstUnits',
  'schedulerRetentionWindowMs',
  'leaseMaxBudgetMs',
  'leaseWorkConservingQuantumMs',
  'continuationTokenCap',
  'l1TokenCap',
  'l1NodeCap',
  'l2TokenCap',
  'l3ByteCap',
  'repairBudget',
  'leaseMaxPauseMs',
  'leasePausePerDebtMs',
  'leaseMinPressureDecodeTokens',
  'leaseBackgroundDecodeTokens',
  'leaseMinPressurePrefillTokens',
  'leaseMaxPrefillTokens',
  'leaseMaxDecodeTokens',
  'leaseMaxPressurePrefillTokens',
  'leasePrefillChunkTokens',
  'leaseMaxCommandBatchSize',
  'leaseMinPressureCommandBatchSize',
  'leaseInflightCap',
  'leaseMildPressureFramePeriods',
  'leaseSeverePressureFramePeriods',
  'leaseQueueWaitToleranceMs',
  'leaseQueueBackpressureSamples',
  'leaseQueueReliefSamples',
  'webgpuCommandSubmitBatchSize',
  'webgpuMaxInflightBatches',
  'queueWaitTimeoutMs',
] as const satisfies readonly (keyof InferenceBridgeOptions)[];

export function readCoordinatorNumericOptions(
  source: URLSearchParams,
  options: InferenceBridgeOptions
): void {
  for (const key of COORDINATOR_NUMERIC_OPTION_KEYS) {
    const raw = source.get(key);
    if (raw === null || raw.trim() === '') {
      continue;
    }
    const value = Number(raw);
    if (Number.isFinite(value) && value >= 0) {
      options[key] = value;
    }
  }
}

export function writeCoordinatorNumericOptions(
  url: URL,
  options: InferenceBridgeOptions
): void {
  for (const key of COORDINATOR_NUMERIC_OPTION_KEYS) {
    const value = options[key];
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }
}
