export function finiteValues(values) {
  return values.filter((value) => Number.isFinite(value));
}

export function mean(values) {
  const finite = finiteValues(values);
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

export function median(values) {
  const finite = finiteValues(values).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  const middle = Math.floor(finite.length / 2);
  if (finite.length % 2 === 1) return finite[middle];
  return (finite[middle - 1] + finite[middle]) / 2;
}

export function percentile(values, q) {
  const finite = finiteValues(values).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  const index = Math.min(
    finite.length - 1,
    Math.max(0, Math.floor((finite.length - 1) * q))
  );
  return finite[index];
}

export function summarize(values) {
  const finite = finiteValues(values);
  return {
    n: finite.length,
    mean: mean(finite),
    median: median(finite),
    p50: percentile(finite, 0.5),
    p95: percentile(finite, 0.95),
    p99: percentile(finite, 0.99),
    min: finite.length === 0 ? null : Math.min(...finite),
    max: finite.length === 0 ? null : Math.max(...finite)
  };
}

export function pairedBootstrapDifference(pairs, options = {}) {
  const usable = pairs.filter(
    ([left, right]) => Number.isFinite(left) && Number.isFinite(right)
  );
  if (usable.length === 0) {
    return {
      n: 0,
      medianDifference: null,
      ci95: [null, null],
      meanDifference: null
    };
  }
  const differences = usable.map(([left, right]) => left - right);
  const iterations = options.iterations ?? 5000;
  const random = mulberry32(options.seed ?? 20261015);
  const resampled = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let total = 0;
    for (let index = 0; index < differences.length; index += 1) {
      total += differences[Math.floor(random() * differences.length)];
    }
    resampled.push(total / differences.length);
  }
  resampled.sort((a, b) => a - b);
  const lowerIndex = Math.floor(resampled.length * 0.025);
  const upperIndex = Math.min(
    resampled.length - 1,
    Math.ceil(resampled.length * 0.975) - 1
  );
  return {
    n: usable.length,
    medianDifference: median(differences),
    meanDifference: mean(differences),
    ci95: [resampled[lowerIndex], resampled[upperIndex]]
  };
}

export function ratioDifference(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return null;
  }
  return (numerator - denominator) / denominator;
}

function mulberry32(seed) {
  return function random() {
    let value = (seed += 0x6d2b79f5);
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
