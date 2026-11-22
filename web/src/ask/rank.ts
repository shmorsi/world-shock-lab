// Ranking helpers, the twins of wsl/rag/retrieve.py.

/** Indices of the `depth` highest positive scores; ties keep index order. */
export function topByScore(scores: ArrayLike<number>, depth: number): number[] {
  const idx: number[] = [];
  for (let i = 0; i < scores.length; i++) if (scores[i] > 0) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a] || a - b);
  return idx.slice(0, depth);
}

/** Reciprocal rank fusion: score = sum of 1 / (k + rank), rank from 1. Ties: better best rank, then earlier list, then index. */
export function rrf(rankings: number[][], k: number): number[] {
  const score = new Map<number, number>();
  const first = new Map<number, [number, number]>();
  rankings.forEach((ranking, li) =>
    ranking.forEach((idx, r0) => {
      const r = r0 + 1;
      score.set(idx, (score.get(idx) ?? 0) + 1 / (k + r));
      if (!first.has(idx)) first.set(idx, [r, li]);
    }),
  );
  return [...score.keys()].sort((a, b) => {
    const d = score.get(b)! - score.get(a)!;
    if (d !== 0) return d;
    const [ra, la] = first.get(a)!;
    const [rb, lb] = first.get(b)!;
    return ra - rb || la - lb || a - b;
  });
}

/** Statement order from a chunk ranking: a statement ranks where its best chunk does. */
export function toStatements(rankedChunks: number[], chunkDoc: ArrayLike<number>): number[] {
  const seen = new Set<number>();
  const out: number[] = [];
  for (const c of rankedChunks) {
    const d = chunkDoc[c];
    if (!seen.has(d)) {
      seen.add(d);
      out.push(d);
    }
  }
  return out;
}
