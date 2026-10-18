/**
 * Normalize a configured target order into a complete, duplicate-free order
 * and rotate it once per measured trial. Rotation matters for fairness: a
 * fixed prefix would always give the same first, middle, and last priority
 * windows to the same sessions, even though the protocol requires every app
 * to participate in the target pool over the run.
 */
export function normalizedTargets(sequence, count, maxSwitches, rotation = 0) {
  if (count <= 0) return [];

  const order = [];
  for (const value of sequence) {
    const index = ((Math.trunc(value) % count) + count) % count;
    if (!order.includes(index)) order.push(index);
  }
  for (let index = 0; index < count; index += 1) {
    if (!order.includes(index)) order.push(index);
  }

  const limit = Math.min(
    order.length,
    Math.max(0, Math.trunc(maxSwitches))
  );
  if (limit === 0) return [];

  const offset =
    ((Math.trunc(rotation) % order.length) + order.length) % order.length;
  return Array.from(
    { length: limit },
    (_, index) => order[(offset + index) % order.length]
  );
}
