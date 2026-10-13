import { useEffect, useState } from "react";

// Every number the app shows comes from these files (web/public/data, written by
// `uv run wsl export site`). They are fetched lazily, once, and cached.
const cache = new Map<string, Promise<unknown>>();

export function load<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    const url = `${import.meta.env.BASE_URL}data/${name}.json`;
    cache.set(
      name,
      fetch(url).then((r) => {
        if (!r.ok) throw new Error(`Could not load ${name}.json (HTTP ${r.status})`);
        return r.json();
      }),
    );
  }
  return cache.get(name) as Promise<T>;
}

export type Loaded<T> = { data: T; error: null } | { data: null; error: string | null };

/** One data file; pass null to hold off (e.g. until it is needed). */
export function useData<T>(name: string | null): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null });
  useEffect(() => {
    if (!name) return;
    let live = true;
    load<T>(name).then(
      (data) => live && setState({ data, error: null }),
      (e: Error) => live && setState({ data: null, error: e.message }),
    );
    return () => {
      live = false;
    };
  }, [name]);
  return state;
}

export function useAll<T extends Record<string, unknown>>(names: (keyof T & string)[]): T | null | string {
  const [state, setState] = useState<T | null | string>(null);
  const key = names.join(",");
  useEffect(() => {
    let live = true;
    Promise.all(names.map((n) => load(n))).then(
      (vals) => {
        if (!live) return;
        const out: Record<string, unknown> = {};
        names.forEach((n, i) => (out[n] = vals[i]));
        setState(out as T);
      },
      (e: Error) => live && setState(e.message),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
