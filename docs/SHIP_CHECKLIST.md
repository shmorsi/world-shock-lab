# Ship checklist

**Status (2026-10-13):** the repository is public at https://github.com/shmorsi/world-shock-lab with all protocol tags, GitHub Pages is on, and workflows may commit. Still open: the `FRED_API_KEY` secret (step 3.1) and the licensing questions in step 1, which were settled for the published record (see data/SOURCES.md O-3/O-4) but remain open for the site's display. Until the secret exists, the daily `live` workflow downloads nothing; its log and job summary say so and list every forecast and score it skipped, and the run fails if a forecast is due (a forecast that misses its 24-hour deadline can never count). Run `uv run wsl live run` locally and push instead.

## 1. Decide three licensing questions (data/SOURCES.md §6)

The repository currently follows the conservative option for each, so it is safe to publish as is.

| Item | Question | What the site does now | If the answer is "yes" |
|---|---|---|---|
| **O-3** | May statistics derived from CanDeal-sourced BoC Valet yields (including the BoC proxy) be published? | Publishes derived responses. Shows no per-event CA yield values, and live scores give only hit/miss for the CA 2-year. | Add `D` to `publish_realized_policies` in `config/live.yaml` (a protocol change: bump its version and log it) and widen the Event Replay filter in `wsl/export/site.py`. |
| **O-4** | Is showing realized returns on specific dates acceptable under Yahoo's terms? | Event Replay and live scores show per-event values only for public-domain series. | Same two places, for policy `L`. |
| **O-9** | May anything derived from VIX (CBOE) or Moody's spreads via FRED be published? | Nothing derived from them is published; the VIX regime toggle is absent from the public site. | `uv run wsl export site` already includes VIX regimes when `public=False`. Make that the public build. |

If you would rather remove even the derived statistics for O-3, say so and the BoC responses can be dropped from the public export.

## 2. Create the GitHub repository and push (done 2026-10-13)

```bash
git remote add origin git@github.com:<you>/world-shock-lab.git
git push -u origin main --tags
```

Pushing the tags (`protocol-v1` … `protocol-v4`) publishes the pre-registration history. The commit dates show that the protocol and its v4 live addendum came before any forecast.

**Timing matters for the first live forecasts.** The forecasts for the 2026-10-28 FOMC and BoC decisions were committed on 2026-10-13, but they only count as pre-registered if the repository is public before **2026-10-27 09:45 ET** (24 hours before the BoC announcement). If you push later, leave the files in place and add a note to the protocol change log that they were not publicly timestamped in time. The 2026-12-09 forecasts are then the first that count.

## 3. Configure the repository (Settings): 2–4 done 2026-10-13; 1 open

1. **Secrets and variables → Actions → New repository secret:** `FRED_API_KEY` = your FRED key. The key is never committed; `.env` is git-ignored.
2. **Pages → Build and deployment → Source:** "GitHub Actions". The `pages` workflow then deploys the site on every push that touches `web/`.
3. **Actions → General → Workflow permissions:** "Read and write permissions". The `live` workflow commits new forecast and score files.
4. After the first deploy, put the Pages URL (`https://<you>.github.io/world-shock-lab/`) in the README's **Use:** line and the repository's "About" box.

## 4. Check the live workflow

- Run it once by hand (**Actions → live → Run workflow**). On a normal day it prints "No new forecasts or scores today."
- Every run ends with a "Live mode" job summary: forecasts due, forecasts missed, decisions waiting for scores, and the next forecast window. `python -m wsl.live.pending` prints the same locally.
- From 2026-11-29 it should publish the 2026-12-09 forecasts on its own, ten days ahead.
- After 2026-10-28, it should add `live/scores/2026-10-28_*_score1.json` once all 5-day outcomes are published. That is typically within 6–10 days, because H.10 exchange rates are released weekly.
- **If Yahoo blocks the GitHub runner** (the job fails while downloading `yahoo/*`), run `uv run wsl live run` locally and push. Forecasts must be committed at least 24 hours before each decision.

## 5. Optional

- `make all` on a fresh clone confirms the one-command reproduction (see docs/REPRODUCING.md). It needs the FRED key and about 150 MB of downloads.
- `make web-setup web-check pdf gif` regenerates the site checks, the PDF and the demo GIF.
- Ask CBOE and Moody's (O-9) and CanDeal (O-3) about redistribution, if you want the fuller public build.
