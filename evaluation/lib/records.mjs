import { createWriteStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

export class RunArtifacts {
  constructor(directory) {
    this.directory = directory;
    this.recordsStream = null;
    this.eventsStream = null;
    this.closed = false;
  }

  async open() {
    await mkdir(this.directory, { recursive: true });
    this.recordsStream = createWriteStream(path.join(this.directory, "records.jsonl"), {
      flags: "a"
    });
    this.eventsStream = createWriteStream(path.join(this.directory, "events.jsonl"), {
      flags: "a"
    });
    return this;
  }

  async record(record) {
    if (this.closed) throw new Error("artifact recorder is closed");
    await writeLine(this.recordsStream, record);
  }

  async event(event) {
    if (this.closed) throw new Error("artifact recorder is closed");
    await writeLine(this.eventsStream, event);
  }

  async writeJson(name, value) {
    await writeJsonAtomic(path.join(this.directory, name), value);
  }

  // Post-processing stages (for example the judge) enrich records after the
  // streaming records.jsonl has already been written. Persisting that
  // enriched copy keeps the per-record diagnostics auditable instead of
  // leaving them only in memory.
  async writeJsonl(name, values) {
    const file = path.join(this.directory, name);
    const temporary = `${file}.tmp`;
    const stream = createWriteStream(temporary);
    try {
      for (const value of values) {
        await writeLine(stream, value);
      }
    } finally {
      await closeStream(stream);
    }
    const { rename } = await import("node:fs/promises");
    await rename(temporary, file);
  }

  async close() {
    if (this.closed) return;
    this.closed = true;
    await Promise.all([
      closeStream(this.recordsStream),
      closeStream(this.eventsStream)
    ]);
  }
}

export async function writeJsonAtomic(file, value) {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  const { rename } = await import("node:fs/promises");
  await rename(temporary, file);
}

async function writeLine(stream, value) {
  if (!stream) throw new Error("artifact stream is not open");
  const line = `${JSON.stringify(value)}\n`;
  if (stream.write(line)) return;
  await new Promise((resolve, reject) => {
    const onDrain = () => {
      stream.off("error", onError);
      resolve();
    };
    const onError = (error) => {
      stream.off("drain", onDrain);
      reject(error);
    };
    stream.once("drain", onDrain);
    stream.once("error", onError);
  });
}

async function closeStream(stream) {
  if (!stream) return;
  await new Promise((resolve, reject) => {
    stream.end(() => resolve());
    stream.once("error", reject);
  });
}
