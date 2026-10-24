import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { load, useData } from "./data";
import { ScenarioProvider } from "./scenario";
import type { Meta } from "./types";
import Overview from "./views/Overview";

const MapView = lazy(() => import("./views/MapView"));
const ExplorerView = lazy(() => import("./views/ExplorerView"));
const ScenarioView = lazy(() => import("./views/ScenarioView"));
const ReplayView = lazy(() => import("./views/ReplayView"));
const BriefView = lazy(() => import("./views/BriefView"));
const MethodView = lazy(() => import("./views/MethodView"));
const TrackView = lazy(() => import("./views/TrackView"));
const ModelsView = lazy(() => import("./views/ModelsView"));

const ROUTES = [
  { path: "", text: "Overview", short: "Overview" },
  { path: "map", text: "World map", short: "Map" },
  { path: "explorer", text: "Explorer", short: "Explorer" },
  { path: "scenario", text: "Scenario", short: "Scenario" },
  { path: "replay", text: "Event replay", short: "Replay" },
  { path: "brief", text: "Brief", short: "Brief" },
  { path: "method", text: "Methodology", short: "Method" },
  { path: "models", text: "Models", short: "Models" },
  { path: "track", text: "Track record", short: "Track" },
] as const;

// The data each view needs, requested as soon as the route is known so the downloads run in
// parallel with the view's code chunk instead of after it.
const ROUTE_DATA: Record<string, string[]> = {
  "": ["summary", "mlstudy", "live"],
  map: ["catalog", "meta"],
  explorer: ["catalog", "ml"], // regimes and m4 load after the first chart is drawn
  scenario: ["catalog"],
  replay: ["replay"],
  brief: ["catalog", "m4"],
  method: ["m3", "m4"],
  models: ["mlstudy"],
  track: ["live"],
};
const prefetch = (route: string) => ROUTE_DATA[route]?.forEach((n) => void load(n).catch(() => undefined));
const currentRoute = () => window.location.hash.replace(/^#\/?/, "").split("?")[0];
prefetch(currentRoute());

function useRoute(): string {
  const get = currentRoute;
  const [route, setRoute] = useState(get);
  useEffect(() => {
    const on = () => {
      prefetch(get());
      setRoute(get());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

type Theme = "system" | "light" | "dark";
function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const stored = localStorage.getItem("wsl-theme");
      return stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
    } catch {
      return "system";
    }
  });
  // "System" leaves the page to prefers-color-scheme, and only clears a theme this toggle set,
  // so a theme chosen by an embedding host is respected.
  const setByUs = useRef(false);
  useEffect(() => {
    const root = document.documentElement;
    if (theme !== "system") {
      root.setAttribute("data-theme", theme);
      setByUs.current = true;
    } else if (setByUs.current) {
      root.removeAttribute("data-theme");
      setByUs.current = false;
    }
    try {
      localStorage.setItem("wsl-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);
  const next = () => setTheme((t) => (t === "system" ? "dark" : t === "dark" ? "light" : "system"));
  return [theme, next];
}

export default function App() {
  const route = useRoute();
  const [theme, nextTheme] = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);
  const meta = useData<Meta>("meta");
  const overview = route === "" || !ROUTES.some((r) => r.path === route);
  useEffect(() => {
    setMenuOpen(false);
  }, [route]);
  useEffect(() => {
    document.body.classList.toggle("nav-open", menuOpen);
    const main = document.getElementById("main");
    const footer = document.querySelector("footer.site");
    for (const el of [main, footer]) {
      if (!el) continue;
      if (menuOpen) el.setAttribute("inert", "");
      else el.removeAttribute("inert");
    }
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.classList.remove("nav-open");
      document.getElementById("main")?.removeAttribute("inert");
      document.querySelector("footer.site")?.removeAttribute("inert");
    };
  }, [menuOpen]);
  const themeName = theme === "system" ? "Auto" : theme === "dark" ? "Dark" : "Light";
  const view = (() => {
    switch (route) {
      case "map":
        return <MapView />;
      case "explorer":
        return <ExplorerView />;
      case "scenario":
        return <ScenarioView />;
      case "replay":
        return <ReplayView />;
      case "brief":
        return <BriefView />;
      case "method":
        return <MethodView />;
      case "models":
        return <ModelsView />;
      case "track":
        return <TrackView />;
      default:
        return <Overview />;
    }
  })();
  return (
    <ScenarioProvider>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar-in">
          <a className="brand" href="#/">
            <svg width="16" height="16" viewBox="0 0 32 32" aria-hidden="true" focusable="false">
              <rect width="31" height="31" x="0.5" y="0.5" fill="none" stroke="currentColor" />
              <path d="M4 20 C9 20 10 8 16 8 S23 24 28 18" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            World Shock Lab
          </a>
          <button
            type="button"
            className="menu"
            aria-expanded={menuOpen}
            aria-controls="view-nav"
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? "Close" : "Menu"}
          </button>
          <nav id="view-nav" className={menuOpen ? "views open" : "views"} aria-label="Views">
            {ROUTES.map((r) => (
              <a key={r.path} href={`#/${r.path}`} aria-current={route === r.path ? "page" : undefined}>
                <span className="nav-short">{r.short}</span>
                <span className="nav-long">{r.text}</span>
              </a>
            ))}
          </nav>
          <button type="button" className="theme" onClick={nextTheme} aria-label={`Colour theme: ${theme}. Change theme.`}>
            {themeName}
          </button>
        </div>
      </header>
      <main id="main">
        <Suspense fallback={<div className="frame"><p className="muted">Loading view…</p></div>}>
          {overview ? view : <div className="frame">{view}</div>}
        </Suspense>
      </main>
      <footer className="site">
        <span>
          Every number on this site is produced by the project's code from pinned public data; nothing is typed in by hand.
          {meta.data && ` Built ${meta.data.built_at.replace("T", " ").replace("Z", " UTC")} from commit ${meta.data.git_sha.slice(0, 7)}.`}
        </span>
        <span>
          <a href="https://github.com/shmorsi/world-shock-lab">Source</a>
          {" · "}
          <a href="https://github.com/shmorsi/world-shock-lab/blob/main/docs/RESEARCH_NOTE.md">Research note</a>
          {" · "}
          <a href="https://github.com/shmorsi/world-shock-lab/blob/main/docs/METHOD.md">Methodology</a>
          {" · "}
          <a href="https://github.com/shmorsi/world-shock-lab/blob/main/docs/MODEL_CARD.md">Model card</a>
          {" · "}
          <a href="https://github.com/shmorsi/world-shock-lab/blob/main/docs/PROVENANCE.md">Provenance</a>
          {" · "}
          <a href="https://github.com/shmorsi/world-shock-lab/blob/main/docs/LIMITATIONS.md">Limitations</a>
        </span>
        <span>
          Data: Jarociński &amp; Karadi (CC BY 4.0), Bauer &amp; Swanson, Känzig (CC BY 4.0), Baumeister &amp; Hamilton, Caldara &amp; Iacoviello, Bank of Canada, Statistics Canada, OECD, World Bank, FRED. This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis. Not investment advice.
        </span>
      </footer>
    </ScenarioProvider>
  );
}
