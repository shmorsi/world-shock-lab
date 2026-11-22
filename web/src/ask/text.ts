// Keyword tokens for BM25, the TypeScript twin of wsl/rag/text.py (config/rag_protocol.yaml):
// NFKD, drop combining marks, lowercase, split on anything not a-z or 0-9, drop stopwords,
// then Harman's S-stemmer. tests/fixtures/rag_cases.json holds cases both must pass.

export function sStem(w: string): string {
  if (w.endsWith("ies") && !w.endsWith("eies") && !w.endsWith("aies")) return w.slice(0, -3) + "y";
  if (w.endsWith("es") && !w.endsWith("aes") && !w.endsWith("ees") && !w.endsWith("oes")) return w.slice(0, -1);
  if (w.endsWith("s") && !w.endsWith("us") && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}

export function tokenize(text: string, stopwords: ReadonlySet<string>): string[] {
  const folded = text.normalize("NFKD").replace(/\p{Mn}/gu, "").toLowerCase();
  return folded
    .split(/[^a-z0-9]+/)
    .filter((t) => t !== "" && !stopwords.has(t))
    .map(sStem);
}
