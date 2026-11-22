// Okapi BM25, the twin of wsl/rag/bm25.py: idf = ln(1 + (N - df + 0.5) / (df + 0.5)),
// query terms counted once.
export class BM25 {
  private readonly tf: Map<string, number>[];
  private readonly lengths: Float64Array;
  private readonly avgdl: number;
  private readonly idf = new Map<string, number>();
  private readonly postings = new Map<string, number[]>();

  constructor(docs: string[][], private readonly k1 = 1.2, private readonly b = 0.75) {
    this.tf = docs.map((d) => {
      const m = new Map<string, number>();
      for (const t of d) m.set(t, (m.get(t) ?? 0) + 1);
      return m;
    });
    this.lengths = Float64Array.from(docs, (d) => d.length);
    this.avgdl = docs.length ? this.lengths.reduce((a, x) => a + x, 0) / docs.length : 0;
    const df = new Map<string, number>();
    this.tf.forEach((m, i) => {
      for (const t of m.keys()) {
        df.set(t, (df.get(t) ?? 0) + 1);
        const p = this.postings.get(t);
        if (p) p.push(i);
        else this.postings.set(t, [i]);
      }
    });
    const n = docs.length;
    for (const [t, f] of df) this.idf.set(t, Math.log(1 + (n - f + 0.5) / (f + 0.5)));
  }

  idfOf(t: string): number {
    return this.idf.get(t) ?? 0;
  }

  scores(query: string[]): Float64Array {
    const out = new Float64Array(this.tf.length);
    for (const t of new Set(query)) {
      const idf = this.idf.get(t);
      if (idf === undefined) continue;
      for (const i of this.postings.get(t)!) {
        const f = this.tf[i].get(t)!;
        const norm = this.k1 * (1 - this.b + (this.b * this.lengths[i]) / this.avgdl);
        out[i] += (idf * f * (this.k1 + 1)) / (f + norm);
      }
    }
    return out;
  }
}
