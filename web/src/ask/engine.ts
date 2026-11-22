// The Ask view's search: the same retrievers as wsl/rag/retrieve.py, run in the browser.
// Keyword search (BM25) works as soon as ask.json arrives; the static embedding model
// (about 8 MB) loads on the first question and then the search becomes the hybrid the
// protocol calls hybrid_static (BM25 + static embeddings, reciprocal rank fusion).
import { BM25 } from "./bm25";
import { rrf, topByScore, toStatements } from "./rank";
import { tokenize } from "./text";
import { dotRows, readQMatrix, StaticModel, type QMatrix } from "./wordpiece";

export interface Move {
  asset: string;
  name: string;
  units: string;
  value: number;
}
export interface Statement {
  id: string;
  bank: "boc" | "fed";
  date: string;
  title: string;
  url: string;
  header: string;
  moves: Move[];
}
export interface Metrics {
  "hit@1": number;
  "hit@5": number;
  "hit@10": number;
  "mrr@10": number;
}
export interface Comparison {
  a: string;
  b: string;
  difference: number;
  lo: number;
  hi: number;
  verdict: string;
}
export interface SplitScores {
  n: number;
  summary: Record<string, Metrics>;
  primary: Comparison;
  site_vs_bm25: Comparison;
  git_sha: string;
}
/** ask.json: small, so the view can draw at once. */
export interface AskMeta {
  protocol_tag: string;
  params: { k1: number; b: number; stopwords: string[]; rrf_k: number; depth: number; static_model: string; static_revision: string };
  scores: { dev: SplitScores; test: SplitScores };
  statements: Statement[];
}
/** With ask_text.json's chunks: [statement index, text]. */
export interface AskData extends AskMeta {
  chunks: [number, string][];
}
export type Kind = "bm25" | "static" | "hybrid_static";
export interface Passage {
  statement: Statement;
  chunk: number;
  body: string;
}

export class AskEngine {
  readonly stop: Set<string>;
  readonly bm25: BM25;
  readonly chunkDoc: Int32Array;
  model: StaticModel | null = null;
  chunkVectors: QMatrix | null = null;

  constructor(readonly data: AskData) {
    this.stop = new Set(data.params.stopwords);
    this.chunkDoc = Int32Array.from(data.chunks, (c) => c[0]);
    this.bm25 = new BM25(
      data.chunks.map(([d, body]) => tokenize(`${data.statements[d].header}\n${body}`, this.stop)),
      data.params.k1,
      data.params.b,
    );
  }

  attachStatic(vocab: string[], model: QMatrix, chunkVectors: QMatrix): void {
    this.model = new StaticModel(new Map(vocab.map((p, i) => [p, i])), model);
    this.chunkVectors = chunkVectors;
  }

  get best(): Kind {
    return this.model ? "hybrid_static" : "bm25";
  }

  rankChunks(kind: Kind, query: string): number[] {
    const depth = this.data.params.depth;
    if (kind === "bm25") return topByScore(this.bm25.scores(tokenize(query, this.stop)), depth);
    if (!this.model || !this.chunkVectors) throw new Error("the static model has not loaded");
    if (kind === "static") return topByScore(dotRows(this.chunkVectors, this.model.encode(query)), depth);
    return rrf([this.rankChunks("bm25", query), this.rankChunks("static", query)], this.data.params.rrf_k);
  }

  /** Statement ids, best first. */
  rank(kind: Kind, query: string): string[] {
    return toStatements(this.rankChunks(kind, query), this.chunkDoc).map((d) => this.data.statements[d].id);
  }

  /** The best chunk of each of the top `k` statements, optionally of one bank only (a filter after ranking; the scores use none). */
  passages(kind: Kind, query: string, k = 5, bank: "boc" | "fed" | null = null): Passage[] {
    const out: Passage[] = [];
    const seen = new Set<number>();
    for (const c of this.rankChunks(kind, query)) {
      const d = this.chunkDoc[c];
      if (seen.has(d) || (bank && this.data.statements[d].bank !== bank)) continue;
      seen.add(d);
      out.push({ statement: this.data.statements[d], chunk: c, body: this.data.chunks[c][1] });
      if (out.length === k) break;
    }
    return out;
  }

  /** The sentence of a passage that shares the rarest words with the question (no model writes text here). */
  quote(question: string, p: Passage): string {
    const q = new Set(tokenize(question, this.stop));
    const sentences = p.body.split(/(?<=[.!?])\s+(?=[A-Z“"])/);
    let best = sentences[0];
    let bestScore = -1;
    for (const s of sentences) {
      let score = 0;
      for (const t of new Set(tokenize(s, this.stop))) if (q.has(t)) score += this.bm25.idfOf(t);
      if (score > bestScore) {
        best = s;
        bestScore = score;
      }
    }
    return best.trim();
  }
}

const base = () => `${import.meta.env.BASE_URL}data/`;

async function get(name: string): Promise<Response> {
  const r = await fetch(base() + name);
  if (!r.ok) throw new Error(`Could not load ${name} (HTTP ${r.status})`);
  return r;
}

export async function loadEngine(meta: Promise<AskMeta>): Promise<AskEngine> {
  const [m, chunks] = await Promise.all([meta, get("ask_text.json").then((r) => r.json() as Promise<[number, string][]>)]);
  return new AskEngine({ ...m, chunks });
}

export async function loadStatic(engine: AskEngine): Promise<void> {
  const [vocab, model, chunks] = await Promise.all([
    get("ask_vocab.json").then((r) => r.json() as Promise<string[]>),
    get("ask_static.bin").then((r) => r.arrayBuffer()),
    get("ask_chunks.bin").then((r) => r.arrayBuffer()),
  ]);
  engine.attachStatic(vocab, readQMatrix(model), readQMatrix(chunks));
}
