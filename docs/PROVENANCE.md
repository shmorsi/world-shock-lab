# Data provenance

A result file says which inputs, which code and which protocol produced it. This page is the map. It does not restate estimates.

## Pins

Each download is stored once under `data/raw/`, read-only, and hashed. `data/manifest.json` holds the SHA-256, the retrieval time and the URL. A source that revises its history changes the hash. The build fails closed instead of quietly adopting the new bytes. `wsl data fetch --repin` is the only command that moves a pin, and it is a deliberate edit.

The long inventory, with the licence and the redistribution rule for every series, is [data/SOURCES.md](../data/SOURCES.md). Coverage and the checks that ran on the built tables are in [DATA_QUALITY.md](DATA_QUALITY.md).

## What a result file records

`wsl/export/results.py` writes the provenance block on every file under `results/`: the series hashes, the code commit, and the sha256 of the protocol file that was in force. Live forecasts and scores carry their own block, including the hash of the shock file they used. The site's JSON under `web/public/data/` is generated from those files and keeps the block. A test fails if a result file is missing it (`tests/test_data_integration.py`).

Commit hashes inside older provenance blocks can point at history that was rewritten before publication. [COMMIT_MAP.md](COMMIT_MAP.md) is the lookup. Those blocks are left as written.

## What is withheld

Three licence questions are still open. Until they are answered, the published record is the narrower one.

- **O-3.** Canadian yields come through the Bank of Canada's Valet service and CanDeal. Derived responses are published. Per-event yield levels are not on the site, and a live score says whether the outcome fell inside the forecast range.
- **O-4.** Yahoo prices are published only as derived statistics. Event replay shows a realized path only for public-domain series. The pre-registered final-evaluation file is kept complete, because it is the evidence, and it is not copied onto the site.
- **O-9.** VIX and Moody's spreads are used in local research only. Nothing derived from them is in the repository's published outputs.

The decisions are tracked as open items in [data/SOURCES.md](../data/SOURCES.md) §6.
