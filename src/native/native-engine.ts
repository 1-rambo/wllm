import type {
  NativeSlotCheckpoint,
  GpuBurstPolicy,
  NativeSlotStats,
  NativeSubmitRequest,
  NativeUnitBudget,
  NativeUnitProgress,
} from '../contracts';

export interface NativeEngine {
  readonly bound?: boolean;
  prewarm(): Promise<void>;
  // Best-effort runtime control for the WebGPU submission burst. Backends
  // without WebGPU may treat this as a no-op.
  setGpuBurstPolicy(policy: GpuBurstPolicy): Promise<void>;
  submit(request: NativeSubmitRequest): Promise<NativeUnitProgress>;
  advance(
    nativeRequestId: string,
    budget: NativeUnitBudget
  ): Promise<NativeUnitProgress>;
  status(nativeRequestId: string): Promise<NativeUnitProgress>;
  cancel(nativeRequestId: string): Promise<void>;
  // End the current native stream at its next safe boundary without
  // discarding the request's KV state. The coordinator uses this to cut a
  // work-conserving unit short when a tab becomes foreground.
  interrupt(nativeRequestId: string): Promise<void>;
  slotExport(slotId: number): Promise<NativeSlotCheckpoint>;
  // Copy a live slot without consuming it. A child fork imports this
  // snapshot while the parent remains addressable in its original slot.
  slotSnapshot(slotId: number): Promise<NativeSlotCheckpoint>;
  slotImport(
    checkpoint: NativeSlotCheckpoint,
    targetSlotId?: number
  ): Promise<NativeSlotStats>;
  // tier two: park a live slot in the native heap and release the serving slot.
  // The returned checkpoint carries a handle, not the state bytes.
  slotPark(slotId: number): Promise<NativeSlotCheckpoint>;
  // tier two from tier three: park serialized bytes without keeping a host copy
  slotParkCheckpoint(
    checkpoint: NativeSlotCheckpoint
  ): Promise<NativeSlotCheckpoint>;
  // tier three: serialize a parked state out of the native heap, keeping the
  // native copy so a failed write can fall back to it
  slotReadParked(checkpoint: NativeSlotCheckpoint): Promise<NativeSlotCheckpoint>;
  // release a native parked state once its serialized copy is durable
  slotReleaseParked(checkpoint: NativeSlotCheckpoint): Promise<void>;
  slotDrop(slotId: number): Promise<void>;
  slotStats(): Promise<NativeSlotStats[]>;
}
