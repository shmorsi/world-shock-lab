import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

// The scenario the user is building, shared by the Composer and the Brief, remembered in this
// browser only (it never leaves the device).
export interface Scenario {
  sizes: Record<string, number>;
  horizon: number;
}
const KEY = "wsl-scenario-v1";
const DEFAULT: Scenario = { sizes: { fed_mp: 10 }, horizon: 5 };

function read(): Scenario {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? { ...DEFAULT, ...JSON.parse(raw) } : DEFAULT;
  } catch {
    return DEFAULT;
  }
}

const Ctx = createContext<{ scenario: Scenario; setScenario: (s: Scenario) => void }>({
  scenario: DEFAULT,
  setScenario: () => undefined,
});

export function ScenarioProvider({ children }: { children: ReactNode }) {
  const [scenario, setScenario] = useState<Scenario>(read);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(scenario));
    } catch {
      /* storage unavailable: keep it in memory */
    }
  }, [scenario]);
  return <Ctx.Provider value={{ scenario, setScenario }}>{children}</Ctx.Provider>;
}

export const useScenario = () => useContext(Ctx);
