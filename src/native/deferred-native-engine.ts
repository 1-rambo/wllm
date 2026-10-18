import type {
  GpuBurstPolicy,
  NativeSlotCheckpoint,
  NativeSlotStats,
  NativeSubmitRequest,
  NativeUnitBudget,
  NativeUnitProgress,
} from '../contracts';
import { WllmError } from '../contracts';
import type { NativeEngine } from './native-engine';

/**
 * A SharedWorker cannot spawn a dedicated worker, so the inference worker is
 * created by a page and its MessagePort is transferred into the coordinator.
 * This holder lets the coordinator keep a stable NativeEngine reference while
 * the port arrives (and while a replacement port is being negotiated).
 */
export class DeferredNativeEngine implements NativeEngine {
  private delegate: NativeEngine | null = null;

  get bound(): boolean {
    return this.delegate !== null;
  }

  attach(delegate: NativeEngine): void {
    this.delegate = delegate;
  }

  detach(): void {
    this.delegate = null;
  }

  prewarm(): Promise<void> {
    return this.require().prewarm();
  }

  setGpuBurstPolicy(policy: GpuBurstPolicy): Promise<void> {
    return this.require().setGpuBurstPolicy(policy);
  }

  submit(request: NativeSubmitRequest): Promise<NativeUnitProgress> {
    return this.require().submit(request);
  }

  advance(
    nativeRequestId: string,
    budget: NativeUnitBudget
  ): Promise<NativeUnitProgress> {
    return this.require().advance(nativeRequestId, budget);
  }

  status(nativeRequestId: string): Promise<NativeUnitProgress> {
    return this.require().status(nativeRequestId);
  }

  cancel(nativeRequestId: string): Promise<void> {
    return this.require().cancel(nativeRequestId);
  }

  interrupt(nativeRequestId: string): Promise<void> {
    return this.require().interrupt(nativeRequestId);
  }

  slotExport(slotId: number): Promise<NativeSlotCheckpoint> {
    return this.require().slotExport(slotId);
  }

  slotSnapshot(slotId: number): Promise<NativeSlotCheckpoint> {
    return this.require().slotSnapshot(slotId);
  }

  slotImport(
    checkpoint: NativeSlotCheckpoint,
    targetSlotId?: number
  ): Promise<NativeSlotStats> {
    return this.require().slotImport(checkpoint, targetSlotId);
  }

  slotDrop(slotId: number): Promise<void> {
    return this.require().slotDrop(slotId);
  }

  slotPark(slotId: number): Promise<NativeSlotCheckpoint> {
    return this.require().slotPark(slotId);
  }

  slotParkCheckpoint(
    checkpoint: NativeSlotCheckpoint
  ): Promise<NativeSlotCheckpoint> {
    return this.require().slotParkCheckpoint(checkpoint);
  }

  slotReadParked(
    checkpoint: NativeSlotCheckpoint
  ): Promise<NativeSlotCheckpoint> {
    return this.require().slotReadParked(checkpoint);
  }

  slotReleaseParked(checkpoint: NativeSlotCheckpoint): Promise<void> {
    return this.require().slotReleaseParked(checkpoint);
  }

  slotStats(): Promise<NativeSlotStats[]> {
    return this.require().slotStats();
  }

  private require(): NativeEngine {
    if (!this.delegate) {
      throw new WllmError(
        'inference worker is not connected',
        'native_unavailable'
      );
    }
    return this.delegate;
  }
}
