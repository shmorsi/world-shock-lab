import { lazy, Suspense, useEffect, useState } from "react";
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

type Theme = "light" | "dark";
// The same storage key and values as every other page of shmorsi.github.io, so a theme chosen
// anywhere on the site carries over. Dark unless the visitor picked light, as on those pages.
function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return localStorage.getItem("theme") === "light" ? "light" : "dark";
    } catch {
      return "dark";
    }
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  const next = () =>
    setTheme((t) => {
      const chosen = t === "dark" ? "light" : "dark";
      try {
        localStorage.setItem("theme", chosen);
      } catch {
        /* ignore */
      }
      return chosen;
    });
  return [theme, next];
}

// The header hides while the reader scrolls down and returns on the way up, as on the project pages.
function useTuckedHeader(disabled: boolean) {
  useEffect(() => {
    let last = window.scrollY;
    const on = () => {
      const y = window.scrollY;
      if (Math.abs(y - last) < 6) return;
      document.body.classList.toggle("tucked", !disabled && y > last && y > 120);
      last = y;
    };
    if (disabled) document.body.classList.remove("tucked");
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, [disabled]);
}

// World Shock Lab is one section of shmorsi.github.io. These are that site's pages, relative to
// this app's folder, in the order of its header (scripts/sync_chrome.py there).
const SITE = [
  { href: "../#work", text: "Work", here: false },
  { href: "../projects/deep-portfolio-constructor.html", text: "Portfolio", here: false },
  { href: "../projects/pension-lab.html", text: "Pension", here: false },
  { href: "../projects/tsfm.html", text: "TSFM", here: false },
  { href: "../projects/world-shock-lab.html", text: "Shock Lab", here: true },
  { href: "../aurvane/", text: "Aurvane", here: false },
] as const;

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
  useTuckedHeader(menuOpen);
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
      <header className="site-header">
        <a className="wordmark" href="../">
          Sherif Morsi
        </a>
        <div className="header-tools">
          <nav id="site-nav" className={menuOpen ? "site-nav open" : "site-nav"} aria-label="Main">
            {SITE.map((l) => (
              <a key={l.href} href={l.href} aria-current={l.here ? "true" : undefined}>
                {l.text}
              </a>
            ))}
            <a className="nav-resume" href="../resume/Sherif_Morsi_Resume.pdf">
              Resume
            </a>
          </nav>
          <button
            type="button"
            className="theme-toggle"
            onClick={nextTheme}
            aria-label={`Switch to the ${theme === "dark" ? "light" : "dark"} theme`}
          >
            <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">
              <circle cx="10" cy="10" r="7.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <path d="M10 2.75a7.25 7.25 0 0 1 0 14.5Z" fill="currentColor" />
            </svg>
          </button>
          <button
            type="button"
            className="menu-button"
            aria-expanded={menuOpen}
            aria-controls="site-nav"
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? "Close" : "Menu"}
          </button>
        </div>
      </header>
      <div className="topbar">
        <div className="topbar-in">
          <a className="brand" href="#/">
            <svg width="16" height="16" viewBox="0 0 32 32" aria-hidden="true" focusable="false">
              <rect width="31" height="31" x="0.5" y="0.5" fill="none" stroke="currentColor" />
              <path d="M4 20 C9 20 10 8 16 8 S23 24 28 18" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            World Shock Lab
          </a>
          <nav id="view-nav" className="views" aria-label="World Shock Lab">
            {ROUTES.map((r) => (
              <a key={r.path} href={`#/${r.path}`} aria-current={route === r.path ? "page" : undefined}>
                {r.short}
              </a>
            ))}
          </nav>
        </div>
      </div>
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
        <span className="site-links">
          <span>© 2026 Sherif Morsi, Montreal</span>
          <a href="../#work">All projects</a>
          <a href="../projects/world-shock-lab.html">Case study</a>
          <a href="../resume/Sherif_Morsi_Resume.pdf">Resume</a>
          <a href="../#contact">Contact</a>
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
