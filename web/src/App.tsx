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
  { path: "", text: "Overview" },
  { path: "map", text: "World map" },
  { path: "explorer", text: "Explorer" },
  { path: "scenario", text: "Scenario" },
  { path: "replay", text: "Event replay" },
  { path: "brief", text: "Brief" },
  { path: "method", text: "Methodology" },
  { path: "models", text: "Models" },
  { path: "track", text: "Track record" },
] as const;

// The data each view needs, requested as soon as the route is known so the downloads run in
// parallel with the view's code chunk instead of after it.
const ROUTE_DATA: Record<string, string[]> = {
  "": ["summary"],
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
      return (localStorage.getItem("wsl-theme") as Theme) || "system";
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
  const meta = useData<Meta>("meta");
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
            World Shock Lab
          </a>
          <nav className="views" aria-label="Views">
            {ROUTES.map((r) => (
              <a key={r.path} href={`#/${r.path}`} aria-current={route === r.path ? "page" : undefined}>
                {r.text}
              </a>
            ))}
          </nav>
          <button type="button" className="theme" onClick={nextTheme} aria-label={`Colour theme: ${theme}. Change theme.`}>
            theme: {theme}
          </button>
        </div>
      </header>
      <main id="main">
        <Suspense fallback={<p className="muted">Loading view…</p>}>{view}</Suspense>
      </main>
      <footer className="site">
        <span>
          Every number on this site is produced by the project's code from pinned public data; nothing is typed in by hand.
          {meta.data && ` Built ${meta.data.built_at.replace("T", " ").replace("Z", " UTC")} from commit ${meta.data.git_sha.slice(0, 7)}.`}
        </span>
        <span>
          Data: Jarociński &amp; Karadi (CC BY 4.0), Bauer &amp; Swanson, Känzig (CC BY 4.0), Baumeister &amp; Hamilton, Caldara &amp; Iacoviello, Bank of Canada, Statistics Canada, OECD, World Bank, FRED. This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis. Not investment advice.
        </span>
      </footer>
    </ScenarioProvider>
  );
}
