import { useEffect, useRef } from "react";

// The still sky behind every page of shmorsi.github.io (assets/site.js there, "the sky"): small
// seeded stars in three tones, drawn once per window size. The lab is one section of that site,
// so it opens on the same sky. Dark theme only (styles.css hides it in the light one), never in
// the embedded view, where the page around the frame already has it.
function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function Sky() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const g = canvas?.getContext("2d");
    if (!canvas || !g) return;
    let seenW = 0;
    let seenH = 0;
    const paint = () => {
      const w = window.innerWidth;
      const h = Math.max(window.innerHeight, 480);
      if (w === seenW && Math.abs(h - seenH) < 140) return; // a phone's address bar sliding away is not worth a repaint
      seenW = w;
      seenH = h;
      const d = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * d);
      canvas.height = Math.round(h * d);
      g.setTransform(d, 0, 0, d, 0, 0);
      g.clearRect(0, 0, w, h);
      const r = seeded(82);
      const count = Math.round(230 * Math.min(1.6, (w * h) / (1440 * 900)));
      for (let i = 0; i < count; i += 1) {
        const x = r() * w;
        const y = r() * h;
        const rad = 0.35 + Math.pow(r(), 3) * 1.25;
        const a = 0.25 + r() * 0.6;
        const tone = r();
        const rgb = tone < 0.1 ? "255,214,170" : tone < 0.4 ? "170,205,255" : "235,240,255";
        g.fillStyle = `rgba(${rgb},${a.toFixed(2)})`;
        g.beginPath();
        g.arc(x, y, rad, 0, 6.2832);
        g.fill();
        if (rad > 0.95) {
          g.fillStyle = `rgba(${rgb},0.1)`;
          g.beginPath();
          g.arc(x, y, rad * 3.2, 0, 6.2832);
          g.fill();
        }
      }
    };
    paint();
    window.addEventListener("resize", paint);
    return () => window.removeEventListener("resize", paint);
  }, []);
  return <canvas ref={ref} className="sky" aria-hidden="true" />;
}
