import type { ReactNode } from "react";
import { LABEL_TEXT } from "../format";

export function Chip({ label }: { label: string }) {
  return <span className={`chip ${label}`}>{LABEL_TEXT[label] ?? label}</span>;
}

export function Seg<T extends string | number>({
  label,
  options,
  value,
  onChange,
  disabled,
}: {
  label: string;
  options: { value: T; text: string }[];
  value: T;
  onChange: (v: T) => void;
  disabled?: (v: T) => boolean;
}) {
  const id = `seg-${label.replace(/\W+/g, "-").toLowerCase()}`;
  return (
    <div className="ctl">
      <span className="lbl" id={id}>
        {label}
      </span>
      <div className="seg" role="group" aria-labelledby={id}>
        {options.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            aria-pressed={o.value === value}
            disabled={disabled?.(o.value)}
            onClick={() => onChange(o.value)}
          >
            {o.text}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Select({
  id,
  label,
  value,
  onChange,
  children,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  children: ReactNode;
}) {
  return (
    <div className="ctl">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {children}
      </select>
    </div>
  );
}

export function Panel({ eyebrow, title, chip, children, id }: { eyebrow?: string; title: string; chip?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="panel" aria-labelledby={id}>
      <div className="panel-head">
        <div>
          {eyebrow && <div className="eyebrow">{eyebrow}</div>}
          <h2 id={id}>{title}</h2>
        </div>
        {chip}
      </div>
      {children}
    </section>
  );
}

export function Loading({ error }: { error: string | null }) {
  return <p className="muted">{error ? `Could not load results: ${error}` : "Loading results…"}</p>;
}

export function TableToggle({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details className="tbl">
      <summary>{summary}</summary>
      <div className="tablewrap">{children}</div>
    </details>
  );
}
