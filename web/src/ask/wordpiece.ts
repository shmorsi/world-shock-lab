// The static embedding model's tokenizer and encoder, the twin of wsl/rag/static.py:
// BERT normalisation (clean text, CJK spacing, lowercase, strip accents), split on whitespace
// and punctuation, greedy longest-match WordPiece, unknown words dropped, mean of the pieces'
// vectors, L2-normalised. Vectors arrive as int8 with one scale per row (wsl/rag/site.py).

const isPunct = (ch: string): boolean => {
  const cp = ch.codePointAt(0)!;
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(ch);
};

const isCjk = (cp: number): boolean =>
  (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) ||
  (cp >= 0x2a700 && cp <= 0x2b73f) || (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) ||
  (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);

export function normalize(text: string): string {
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    const ws = ch === "\t" || ch === "\n" || ch === "\r";
    if (cp === 0 || cp === 0xfffd || (/\p{C}/u.test(ch) && !ws)) continue;
    if (ws || /\p{Zs}/u.test(ch)) out += " ";
    else if (isCjk(cp)) out += ` ${ch} `;
    else out += ch;
  }
  return out.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "");
}

export function preTokenize(text: string): string[] {
  const words: string[] = [];
  for (const chunk of text.split(/\s+/)) {
    let cur = "";
    for (const ch of chunk) {
      if (isPunct(ch)) {
        if (cur) words.push(cur);
        cur = "";
        words.push(ch);
      } else cur += ch;
    }
    if (cur) words.push(cur);
  }
  return words;
}

/** Greedy longest-match-first; null when the word cannot be covered. Lengths count code points, as in Python. */
export function wordpiece(word: string, vocab: ReadonlyMap<string, number>): number[] | null {
  const chars = [...word];
  if (chars.length > 100) return null;
  const ids: number[] = [];
  let start = 0;
  while (start < chars.length) {
    let end = chars.length;
    let found: number | undefined;
    while (start < end) {
      const sub = chars.slice(start, end).join("");
      const piece = start === 0 ? sub : "##" + sub;
      found = vocab.get(piece);
      if (found !== undefined) break;
      end--;
    }
    if (found === undefined) return null;
    ids.push(found);
    start = end;
  }
  return ids;
}

/** Rows of int8 values with a float32 scale per row: value = scale * q. */
export interface QMatrix {
  rows: number;
  dims: number;
  scales: Float32Array;
  q: Int8Array | Float32Array;
}

export function readQMatrix(buf: ArrayBuffer): QMatrix {
  const head = new DataView(buf);
  const rows = head.getUint32(0, true);
  const dims = head.getUint32(4, true);
  const scales = new Float32Array(buf.slice(8, 8 + 4 * rows));
  const q = new Int8Array(buf, 8 + 4 * rows, rows * dims);
  return { rows, dims, scales, q };
}

/** Unquantized vectors (the shared test cases): scale 1, float values. */
export function fromFloat(rows: number[][]): QMatrix {
  const dims = rows[0]?.length ?? 0;
  return { rows: rows.length, dims, scales: new Float32Array(rows.length).fill(1), q: Float32Array.from(rows.flat()) };
}

export class StaticModel {
  constructor(readonly vocab: ReadonlyMap<string, number>, readonly vectors: QMatrix) {}

  tokenIds(text: string): number[] {
    const ids: number[] = [];
    for (const w of preTokenize(normalize(text))) {
      const pieces = wordpiece(w, this.vocab);
      if (pieces) ids.push(...pieces);
    }
    return ids;
  }

  encode(text: string): Float32Array {
    const { dims, scales, q } = this.vectors;
    const v = new Float32Array(dims);
    const ids = this.tokenIds(text);
    if (!ids.length) return v;
    for (const id of ids) {
      const s = scales[id];
      for (let j = 0; j < dims; j++) v[j] += s * q[id * dims + j];
    }
    let n = 0;
    for (let j = 0; j < dims; j++) {
      v[j] /= ids.length;
      n += v[j] * v[j];
    }
    n = Math.sqrt(n);
    if (n > 0) for (let j = 0; j < dims; j++) v[j] /= n;
    return v;
  }
}

/** Dot product of every row of `m` with `v`. */
export function dotRows(m: QMatrix, v: Float32Array): Float64Array {
  const out = new Float64Array(m.rows);
  for (let i = 0; i < m.rows; i++) {
    let s = 0;
    const o = i * m.dims;
    for (let j = 0; j < m.dims; j++) s += m.q[o + j] * v[j];
    out[i] = s * m.scales[i];
  }
  return out;
}
