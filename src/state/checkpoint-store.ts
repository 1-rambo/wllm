import type { BlobStore } from '../contracts';

export class MemoryBlobStore implements BlobStore {
  readonly values = new Map<string, Uint8Array>();

  async put(key: string, value: Uint8Array): Promise<void> {
    this.values.set(key, new Uint8Array(value));
  }

  async get(key: string): Promise<Uint8Array | null> {
    const value = this.values.get(key);
    return value ? new Uint8Array(value) : null;
  }

  async delete(key: string): Promise<void> {
    this.values.delete(key);
  }

  async listKeys(): Promise<string[]> {
    return [...this.values.keys()].sort();
  }
}

export class VolatileBlobStore implements BlobStore {
  private readonly values = new Map<string, Uint8Array>();
  readonly operationErrors: Array<'put' | 'get' | 'delete' | 'listKeys'> = [];

  async put(key: string, value: Uint8Array): Promise<void> {
    this.maybeFail('put');
    this.values.set(key, new Uint8Array(value));
  }

  async get(key: string): Promise<Uint8Array | null> {
    this.maybeFail('get');
    const value = this.values.get(key);
    return value ? new Uint8Array(value) : null;
  }

  async delete(key: string): Promise<void> {
    this.maybeFail('delete');
    this.values.delete(key);
  }

  async listKeys(): Promise<string[]> {
    this.maybeFail('listKeys');
    return [...this.values.keys()].sort();
  }

  failNext(operation: 'put' | 'get' | 'delete' | 'listKeys'): void {
    this.operationErrors.push(operation);
  }

  private maybeFail(operation: 'put' | 'get' | 'delete' | 'listKeys'): void {
    const index = this.operationErrors.indexOf(operation);
    if (index >= 0) {
      this.operationErrors.splice(index, 1);
      throw new Error(`simulated store ${operation} failure`);
    }
  }
}

export class ResilientBlobStore implements BlobStore {
  readonly fallback = new VolatileBlobStore();
  fallbackCount = 0;
  lastError: unknown = null;
  private useFallback = false;

  constructor(private readonly primary: BlobStore) {}

  async put(key: string, value: Uint8Array): Promise<void> {
    if (this.useFallback) {
      await this.fallback.put(key, value);
      return;
    }
    try {
      await this.primary.put(key, value);
    } catch (error) {
      this.activateFallback(error);
      await this.fallback.put(key, value);
    }
  }

  async get(key: string): Promise<Uint8Array | null> {
    if (this.useFallback) {
      return this.fallback.get(key);
    }
    try {
      return await this.primary.get(key);
    } catch (error) {
      this.activateFallback(error);
      return this.fallback.get(key);
    }
  }

  async delete(key: string): Promise<void> {
    if (this.useFallback) {
      await this.fallback.delete(key);
      return;
    }
    try {
      await this.primary.delete(key);
    } catch (error) {
      this.activateFallback(error);
      await this.fallback.delete(key);
    }
  }

  async listKeys(): Promise<string[]> {
    if (this.useFallback) {
      return this.fallback.listKeys();
    }
    try {
      return await this.primary.listKeys();
    } catch (error) {
      this.activateFallback(error);
      return this.fallback.listKeys();
    }
  }

  get usingFallback(): boolean {
    return this.useFallback;
  }

  private activateFallback(error: unknown): void {
    this.useFallback = true;
    this.fallbackCount += 1;
    this.lastError = error;
  }
}

interface OpfsWritable {
  write(data: Uint8Array): Promise<void>;
  close(): Promise<void>;
}

interface OpfsFileHandle {
  createWritable(): Promise<OpfsWritable>;
  getFile(): Promise<{ arrayBuffer(): Promise<ArrayBuffer> }>;
}

interface OpfsDirectoryHandle {
  getDirectoryHandle(
    name: string,
    options?: { create?: boolean }
  ): Promise<OpfsDirectoryHandle>;
  getFileHandle(
    name: string,
    options?: { create?: boolean }
  ): Promise<OpfsFileHandle>;
  removeEntry(name: string): Promise<void>;
  entries(): AsyncIterableIterator<[string, unknown]>;
}

interface OpfsStorageManager {
  getDirectory(): Promise<OpfsDirectoryHandle>;
}

export class BrowserOpfsBlobStore implements BlobStore {
  constructor(private readonly namespace = 'wllm-checkpoints') {}

  async put(key: string, value: Uint8Array): Promise<void> {
    const directory = await this.directory();
    const file = await directory.getFileHandle(this.fileName(key), {
      create: true,
    });
    const writable = await file.createWritable();
    await writable.write(value);
    await writable.close();
  }

  async get(key: string): Promise<Uint8Array | null> {
    const directory = await this.directory();
    try {
      const file = await directory.getFileHandle(this.fileName(key));
      const buffer = await (await file.getFile()).arrayBuffer();
      return new Uint8Array(buffer);
    } catch (error) {
      if (isNotFound(error)) {
        return null;
      }
      throw error;
    }
  }

  async delete(key: string): Promise<void> {
    const directory = await this.directory();
    try {
      await directory.removeEntry(this.fileName(key));
    } catch (error) {
      if (!isNotFound(error)) {
        throw error;
      }
    }
  }

  async listKeys(): Promise<string[]> {
    const directory = await this.directory();
    const keys: string[] = [];
    for await (const [name] of directory.entries()) {
      if (name.startsWith('checkpoint-')) {
        keys.push(decodeURIComponent(name.slice('checkpoint-'.length)));
      }
    }
    return keys.sort();
  }

  private async directory(): Promise<OpfsDirectoryHandle> {
    const storage = (
      globalThis.navigator as unknown as { storage?: OpfsStorageManager }
    ).storage;
    if (!storage?.getDirectory) {
      throw new Error('OPFS is not available');
    }
    const root = await storage.getDirectory();
    return root.getDirectoryHandle(this.namespace, { create: true });
  }

  private fileName(key: string): string {
    return `checkpoint-${encodeURIComponent(key)}`;
  }
}

function isNotFound(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    (error as { name?: unknown }).name === 'NotFoundError'
  );
}
