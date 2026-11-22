// The browser search must match the Python it twins (wsl/rag/): the shared cases in
// tests/fixtures/rag_cases.json exactly, and, on the 64 fixed questions, the rankings Python
// computes with float32 vectors (tests/fixtures/rag_site_rankings.json) closely: the browser
// uses int8 vectors, so a near-tie can swap.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { BM25 } from "./bm25";
import { AskEngine, type AskData } from "./engine";
import { rrf } from "./rank";
import { sStem, tokenize } from "./text";
import { fromFloat, normalize, preTokenize, readQMatrix, StaticModel } from "./wordpiece";

const root = resolve(__dirname, "../../..");
const json = (p: string) => JSON.parse(readFileSync(resolve(root, p), "utf8"));
const cases = json("tests/fixtures/rag_cases.json");
const data = { ...json("web/public/data/ask.json"), chunks: json("web/public/data/ask_text.json") } as AskData;
const stop = new Set(data.params.stopwords);

describe("keyword tokens and ranking", () => {
  it("tokenizes as Python does", () => {
    for (const c of cases.tokenize) expect(tokenize(c.text, stop), c.text).toEqual(c.tokens);
    expect(sStem("policies")).toBe("policy");
    expect(sStem("s")).toBe("");
  });
  it("scores BM25 as Python does", () => {
    const s = new BM25(cases.bm25.docs).scores(cases.bm25.query);
    cases.bm25.scores.forEach((x: number, i: number) => expect(s[i]).toBeCloseTo(x, 9));
  });
  it("fuses as Python does", () => {
    for (const c of cases.rrf) expect(rrf(c.lists, 60)).toEqual(c.fused);
  });
});

describe("static embeddings", () => {
  const st = cases.static;
  const model = new StaticModel(new Map(Object.entries(st.vocab as Record<string, number>)), fromFloat(st.vectors));
  it("splits and encodes as Python does", () => {
    for (const c of st.cases) {
      expect(model.tokenIds(c.text), c.text).toEqual(c.ids);
      const v = model.encode(c.text);
      c.vector.forEach((x: number, i: number) => expect(v[i]).toBeCloseTo(x, 5));
    }
    expect(normalize("Café\tCRÈME 東京")).toBe("cafe creme  東  京 ");
    expect(preTokenize("u.s.-mexico 2¼%")).toEqual(["u", ".", "s", ".", "-", "mexico", "2¼", "%"]);
  });
});

describe("the exported search, against Python's rankings", () => {
  const engine = new AskEngine(data);
  const buf = (p: string) => {
    const b = readFileSync(resolve(root, p));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  };
  engine.attachStatic(json("web/public/data/ask_vocab.json"), readQMatrix(buf("web/public/data/ask_static.bin")), readQMatrix(buf("web/public/data/ask_chunks.bin")));
  const rankings = json("tests/fixtures/rag_site_rankings.json") as Record<string, { question: string; gold: string[]; hybrid_static: string[]; bm25: string[] }>;
  const qs = Object.values(rankings);

  it("keyword search gives the same top 10", () => {
    const same = qs.filter((q) => engine.rank("bm25", q.question).slice(0, 10).join() === q.bm25.join()).length;
    expect(same).toBe(qs.length);
  });
  it("hybrid search agrees closely despite int8 vectors", () => {
    let top1 = 0;
    let hitsTs = 0;
    let hitsPy = 0;
    for (const q of qs) {
      const ts = engine.rank("hybrid_static", q.question);
      if (ts[0] === q.hybrid_static[0]) top1++;
      if (ts.slice(0, 5).some((d) => q.gold.includes(d))) hitsTs++;
      if (q.hybrid_static.slice(0, 5).some((d) => q.gold.includes(d))) hitsPy++;
    }
    expect(top1).toBeGreaterThanOrEqual(Math.ceil(0.9 * qs.length));
    expect(Math.abs(hitsTs - hitsPy)).toBeLessThanOrEqual(2);
  });
});
