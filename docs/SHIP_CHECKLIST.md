# Ship checklist

**Status (2026-10-23):** the repository is public at https://github.com/shmorsi/world-shock-lab with all protocol tags, GitHub Pages is on at https://shmorsi.github.io/world-shock-lab/ (also the README **Site** line and the repository homepage), and workflows may commit. The 2026-10-28 forecasts were committed on 2026-10-13, and the repository was already public then, which is before the 2026-10-27 09:45 ET timestamp in step 2.

Still open: the `FRED_API_KEY` secret (step 3.1) and the licensing questions in step 1, which were settled for the published record (see data/SOURCES.md O-3/O-4) but remain open for the site's display. The last scheduled `live` run before this note (2026-10-23 00:44 UTC) still had an empty secret. Until the secret exists the job downloads nothing, lists every forecast and score it could not touch, and **fails**. Run `uv run wsl live run` locally and push instead. Add the secret before merging a change to `.github/workflows/live.yml`, or the next weekday run will be red.

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

**Timing matters for the first live forecasts.** The forecasts for the 2026-10-28 FOMC and BoC decisions were committed on 2026-10-13. They count as pre-registered only if the repository is public before **2026-10-27 09:45 ET** (24 hours before the BoC announcement). The repository has been public since 2026-10-13, so that condition is met. Leave the forecast files as they are.

## 3. Configure the repository (Settings): 2–4 done 2026-10-13; 1 open

1. **Secrets and variables → Actions → New repository secret:** `FRED_API_KEY` = your FRED key. The key is never committed; `.env` is git-ignored.
2. **Pages → Build and deployment → Source:** "GitHub Actions". The `pages` workflow then deploys the site on every push that touches `web/`.
3. **Actions → General → Workflow permissions:** "Read and write permissions". The `live` workflow commits new forecast and score files.
4. **Done.** The Pages URL is the README **Site** line and the repository homepage (`https://shmorsi.github.io/world-shock-lab/`). There is no separate "Use:" line.

## 4. Check the live workflow

The workflow file is `.github/workflows/live.yml`.

- **When it runs.** Cron `30 22 * * 1-5`: 22:30 UTC, Monday through Friday. That is 18:30 America/New_York through the 2026-11-01 clock change and 17:30 afterwards. GitHub may start it an hour or two late. It does not run on Saturday or Sunday. It can also be started by hand (**Actions → live → Run workflow**), and that manual run has to be on `main`.
- **What the 2026-10-28 run does.** The BoC announcement is 09:45 America/Toronto (13:45 UTC) and the FOMC statement is 14:00 America/New_York (18:00 UTC), so both are over before that day's 22:30 UTC run. The forecasts are already committed. Score 1 is written only once every h = 5 outcome is in the download. USD/CAD comes from the weekly H.10 release, so that is typically 6–10 days after the decision, on the first later weekday run that sees a complete row. The files are `live/scores/2026-10-28_boc_score1.json` and `live/scores/2026-10-28_fomc_score1.json`. FOMC score 2 (`live/scores/2026-10-28_fomc_score2.json`) waits until the JK surprise file lists that meeting. Published files are not edited; a later run that finds them does nothing to them.
- **How a run with the secret proceeds.** `FRED_API_KEY` is read from the Actions secret. The job installs the lockfile, runs `uv run wsl live run` (download, score, publish due forecasts, rewrite `web/public/data/live.json`), and commits only when `live/` gained a file. `actions/checkout` leaves a detached HEAD, so the commit is pushed with `git push origin HEAD:main`. The bot push does not start other workflows, so the job then runs `gh workflow run pages.yml` to redeploy the site.
- **If the secret is missing.** The job prints a "Live mode" summary of what it skipped and exits with an error. It does not download or write files.
- **Forecasts after these.** From 2026-11-29 the same weekday job should publish the 2026-12-09 forecasts, ten days ahead and at least 24 hours before the announcements.
- **If Yahoo blocks the GitHub runner** (the job fails while downloading `yahoo/*`), run `uv run wsl live run` locally and push. Forecasts must be committed at least 24 hours before each decision. A failed download does not write a score.
- On a day with nothing new, the commit step prints "No new forecasts or scores today." Every run that has the secret ends with a "Live mode" job summary: forecasts due, forecasts missed, decisions waiting for scores, and the next forecast window. `python -m wsl.live.pending` prints the same locally.

## 5. Optional

- `make all` on a fresh clone confirms the one-command reproduction (see docs/REPRODUCING.md). It needs the FRED key and about 150 MB of downloads.
- `make web-setup web-check pdf gif` regenerates the site checks, the PDF and the demo GIF.
- Ask CBOE and Moody's (O-9) and CanDeal (O-3) about redistribution, if you want the fuller public build.
