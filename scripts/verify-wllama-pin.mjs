import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pinPath = path.join(root, 'vendor', 'wllama-pin.json');
const vendoredRoot = path.join(root, 'vendor', 'wllama');
const pin = JSON.parse(await readFile(pinPath, 'utf8'));

async function requireFile(filePath, message) {
  try {
    await access(filePath);
  } catch {
    throw new Error(`${message}: ${filePath}`);
  }
}

async function readTrimmed(filePath, message) {
  await requireFile(filePath, message);
  return (await readFile(filePath, 'utf8')).trim();
}

try {
  const upstreamPath = path.join(vendoredRoot, 'UPSTREAM_COMMIT');
  const llamaCppPath = path.join(vendoredRoot, 'LLAMA_CPP_COMMIT');
  const actualCommit = await readTrimmed(
    upstreamPath,
    'missing vendored wllama commit'
  );
  const actualLlamaCpp = await readTrimmed(
    llamaCppPath,
    'missing vendored llama.cpp commit'
  );

  if (actualCommit !== pin.commit) {
    throw new Error(
      `wllama commit mismatch: expected ${pin.commit}, found ${actualCommit}`
    );
  }
  if (actualLlamaCpp !== pin.llamaCppCommit) {
    throw new Error(
      `llama.cpp commit mismatch: expected ${pin.llamaCppCommit}, found ${actualLlamaCpp}`
    );
  }

  for (const patch of pin.patches ?? []) {
    await requireFile(path.join(root, patch), 'missing wllama pin patch');
  }

  console.log(
    `wllama pin OK: ${actualCommit.slice(0, 12)}; llama.cpp pin OK: ${actualLlamaCpp.slice(0, 12)}`
  );
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
