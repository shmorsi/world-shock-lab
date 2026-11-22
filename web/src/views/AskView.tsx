import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { AskEngine, loadEngine, loadStatic, type AskMeta, type Kind, type Passage, type SplitScores } from "../ask/engine";
import { load, useData } from "../data";
import { Loading, Panel, Seg, TableToggle } from "../components/ui";

// Ask the decisions: search 438 Fed and Bank of Canada statements in the browser. Nothing is
// sent anywhere and no model writes text here: the answer line is a sentence quoted from the
// best statement. The Python version (`uv run wsl rag ask`) can also write an answer with
// Claude, under a rule that withholds any answer printing a number its sources do not contain.

type Bank = "all" | "boc" | "fed";
const EXAMPLES: { q: string; bank: Bank }[] = [
  { q: "When did the Bank of Canada raise rates by 100 basis points?", bank: "boc" },
  { q: "What did the Fed say about banks after the failures in March 2023?", bank: "fed" },
  { q: "How did the Bank of Canada describe the invasion of Ukraine?", bank: "boc" },
  { q: "What did the Fed say about Hurricane Katrina?", bank: "fed" },
  { q: "When did the Bank of Canada end quantitative easing?", bank: "boc" },
];
const KIND_TEXT: Record<Kind, string> = {
  bm25: "keyword search (BM25)",
  static: "embedding search",
  hybrid_static: "keyword + embedding search, fused",
};
const BANK = { boc: "Bank of Canada", fed: "Federal Reserve" } as const;
const pct = (x: number) => `${Math.round(x * 1000) / 10}%`;
const pts = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(Math.round(x * 1000) / 10)}`;
const longDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-CA", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });

const params = () => new URLSearchParams(window.location.hash.split("?")[1] ?? "");
const fromHash = (): string => params().get("q") ?? "";
const bankFromHash = (): Bank => {
  const b = params().get("bank");
  return b === "boc" || b === "fed" ? b : "all";
};

export default function AskView() {
  const meta = useData<AskMeta>("ask");
  const [engine, setEngine] = useState<AskEngine | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meaning, setMeaning] = useState<"idle" | "loading" | "ready" | "failed">("idle");
  const [draft, setDraft] = useState(fromHash);
  const [question, setQuestion] = useState(fromHash);
  const [kind, setKind] = useState<Kind>("bm25");
  const [bank, setBank] = useState<Bank>(bankFromHash);
  const started = useRef(false);

  useEffect(() => {
    loadEngine(load<AskMeta>("ask")).then(setEngine, (e: Error) => setError(e.message));
  }, []);

  // The embedding model (about 8 MB) loads only once someone asks something.
  useEffect(() => {
    if (!engine || !question || started.current) return;
    started.current = true;
    setMeaning("loading");
    loadStatic(engine).then(
      () => {
        setMeaning("ready");
        setKind("hybrid_static");
      },
      () => setMeaning("failed"),
    );
  }, [engine, question]);

  const results: Passage[] = useMemo(
    () =>
      engine && question.trim()
        ? engine.passages(kind === "bm25" || engine.model ? kind : "bm25", question, 5, bank === "all" ? null : bank)
        : [],
    [engine, question, kind, bank],
  );

  const submit = (e: FormEvent) => {
    e.preventDefault();
    ask(draft, bank);
  };
  const ask = (q: string, b: Bank) => {
    setDraft(q);
    setQuestion(q.trim());
    setBank(b);
    const p = new URLSearchParams();
    if (q.trim()) p.set("q", q.trim());
    if (b !== "all") p.set("bank", b);
    window.history.replaceState(null, "", `#/ask${p.size ? `?${p}` : ""}`);
  };

  if (!meta.data) return <Loading error={meta.error ?? error} />;
  const m = meta.data;
  const test = m.scores.test;
  const top = results[0];

  return (
    <>
      <Panel eyebrow="View 9 · Ask the decisions" title="Search every Fed and Bank of Canada statement since 2000" id="ask-h">
        <p className="lede">
          {m.statements.length} policy statements, from {longDate(m.statements[0].date)} to{" "}
          {longDate(m.statements[m.statements.length - 1].date)}, searched in your browser. Each result links to the bank's
          own page. Where the lab has market data for that announcement, it shows what moved.
        </p>
        <form className="ask-form" onSubmit={submit} role="search">
          <label htmlFor="ask-q" className="sr-only">
            Your question
          </label>
          <input
            id="ask-q"
            type="search"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="When did the Fed begin tapering its bond purchases?"
            autoComplete="off"
          />
          <button className="btn" type="submit">
            Search
          </button>
        </form>
        <div className="controls ask-controls">
          <Seg
            label="Statements from"
            options={[
              { value: "all" as Bank, text: "Both banks" },
              { value: "fed" as Bank, text: "Fed" },
              { value: "boc" as Bank, text: "Bank of Canada" },
            ]}
            value={bank}
            onChange={(b) => ask(question, b)}
          />
        </div>
        <div className="ask-examples" aria-label="Example questions">
          {EXAMPLES.map((e) => (
            <button key={e.q} type="button" className="btn ghost" onClick={() => ask(e.q, e.bank)}>
              {e.q}
            </button>
          ))}
        </div>
        <p className="muted ask-status" aria-live="polite">
          {error
            ? `The statements did not load: ${error}`
            : !engine
            ? "Loading the statements…"
            : question
            ? `Ranked by ${KIND_TEXT[engine.model ? kind : "bm25"]}.` +
              (bank !== "all" ? ` Showing ${BANK[bank]} statements only.` : "") +
              (meaning === "loading" ? " Loading the embedding model (about 8 MB) to add meaning-based search…" : "") +
              (meaning === "failed" ? " The embedding model did not load, so this is keyword search only." : "")
            : "Type a question, or pick an example."}
        </p>
      </Panel>

      {question && engine && (
        <Panel eyebrow="Answer" title={top ? `${BANK[top.statement.bank]}, ${longDate(top.statement.date)}` : "No statement matched"} id="ask-a">
          {top ? (
            <>
              <blockquote className="ask-quote">
                “{engine.quote(question, top)}” <a href={top.statement.url}>Source</a>
              </blockquote>
              <p className="muted">
                Quoted from the best-matching statement, not written by a model. Check the other statements below: the search finds the right
                one in its top five about {pct(test.summary.hybrid_static["hit@5"])} of the time on the test questions (scored with both banks), not always first.
              </p>
              <ol className="ask-results">
                {results.map((p, i) => (
                  <li key={p.statement.id}>
                    <div className="ask-meta">
                      <span className={`ask-bank ${p.statement.bank}`}>{BANK[p.statement.bank]}</span>
                      <span className="mono">{p.statement.date}</span>
                      <span className="muted">#{i + 1}</span>
                    </div>
                    <h3>
                      <a href={p.statement.url}>{p.statement.title}</a>
                    </h3>
                    <p>{p.body}</p>
                    {p.statement.moves.length > 0 && (
                      <p className="ask-moves">
                        <span className="muted">What moved, first observation after the announcement:</span>{" "}
                        {p.statement.moves.map((m) => (
                          <span key={m.asset} className="mono">
                            {m.name} {m.value > 0 ? "+" : m.value < 0 ? "−" : ""}
                            {Math.abs(m.value).toFixed(2)}
                            {m.units === "bp" ? " bp" : m.units}
                          </span>
                        ))}
                      </p>
                    )}
                  </li>
                ))}
              </ol>
            </>
          ) : (
            <p>Try other words: the search needs at least one word that appears in a statement, or the embedding model to load.</p>
          )}
        </Panel>
      )}

      <Panel eyebrow="How well it works" title="Scored once, on questions fixed before the search was written" id="ask-s">
        <Scores test={test} dev={m.scores.dev} tag={m.protocol_tag} />
      </Panel>
    </>
  );
}

function Scores({ test, dev, tag }: { test: SplitScores; dev: SplitScores; tag: string }) {
  const rows: [string, string][] = [
    ["bm25", "Keyword (BM25)"],
    ["static", "Static embeddings"],
    ["transformer", "Transformer embeddings (bge-small)"],
    ["hybrid_static", "Keyword + static, fused (this page)"],
    ["hybrid", "Keyword + transformer, fused (Python)"],
  ];
  const p = test.primary;
  return (
    <>
      <p>
        {test.n} test questions, written with their answers and tagged <span className="mono">{tag}</span> in git before any retriever existed,
        then scored once. Adding transformer embeddings to keyword search finds a right statement in the top five{" "}
        <strong>{pts(p.difference)} points</strong> more often ({pct(test.summary.bm25["hit@5"])} to {pct(test.summary.hybrid["hit@5"])}; 90%
        interval {pts(p.lo)} to {pts(p.hi)}): <strong>{p.verdict}</strong>, by the rule fixed in advance. This page runs the lighter static model,
        which fits in a browser: {pct(test.summary.hybrid_static["hit@5"])}, {test.site_vs_bm25.verdict} from keyword search alone.
      </p>
      <p className="muted">
        Misses are mostly questions about the first time something happened, which need reasoning over dates that search cannot do, and
        questions whose words differ from the statement's ("a full percentage point" against "100 basis points").
      </p>
      <TableToggle summary="All five retrievers, test and dev questions">
        <table>
          <thead>
            <tr>
              <th scope="col">Retriever</th>
              <th scope="col">Test hit@1</th>
              <th scope="col">Test hit@5</th>
              <th scope="col">Test hit@10</th>
              <th scope="col">Test MRR@10</th>
              <th scope="col">Dev hit@5</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, name]) => (
              <tr key={k}>
                <th scope="row">{name}</th>
                <td className="num">{pct(test.summary[k]["hit@1"])}</td>
                <td className="num">{pct(test.summary[k]["hit@5"])}</td>
                <td className="num">{pct(test.summary[k]["hit@10"])}</td>
                <td className="num">{test.summary[k]["mrr@10"].toFixed(3)}</td>
                <td className="num">{pct(dev.summary[k]["hit@5"])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableToggle>
      <p className="muted">
        Protocol: <a href="https://github.com/shmorsi/world-shock-lab/blob/main/config/rag_protocol.yaml">config/rag_protocol.yaml</a> · Questions:{" "}
        <a href="https://github.com/shmorsi/world-shock-lab/blob/main/config/rag_questions.yaml">config/rag_questions.yaml</a> · Write-up:{" "}
        <a href="https://github.com/shmorsi/world-shock-lab/blob/main/docs/RAG.md">docs/RAG.md</a>
      </p>
    </>
  );
}
