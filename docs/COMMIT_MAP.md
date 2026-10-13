# Commit hash map

Before the repository was first published (2026-10-13), its history was re-authored so that every commit and protocol tag is attributed to the owner's GitHub account (`shmorsi`). The original commits had been authored under a second account on the same machine.

Only the author and committer identity changed. Every commit's file tree, message and dates are identical, and the protocol tags (`protocol-v1` … `protocol-v4`) keep their original timestamps and messages. Re-authoring changes commit hashes, though. Several records committed earlier cite the old hashes, and they are deliberately left unedited: the final ML evaluation in `results/final_eval/`, the live forecasts in `live/forecasts/`, and the provenance blocks in `results/` and `web/public/data/`. Use this table to resolve them.

| Old hash (cited in records) | Published hash | Commit |
|---|---|---|
| `dfe3b89b6818cc34a276ca58284a63e95ec14d9b` | `8d73fc49ce0fae00674403fdcd61353eaa075137` | M0: verified data sources, scope and pre-registered evaluation protocol |
| `8732db47a92234ebe330dbb56f5286d6f88c8d68` | `40b8d13ebf9feb9a10ffe522e947179585320b75` | M1: data layer with immutable raw cache, validated build and BoC announcement list |
| `d2e9b74c3bfb1481ab3821afb4078c967d4daed7` | `790339c89913d2f94c5d0ac46a2c6c067d76ebde` | Report: show when a build came from uncommitted code |
| `d9fbcea80881eb0d20783ce6c3489433ed61d561` | `f0da88b9f4c1d588c23a6c952b521b855ebc296a` | Regenerate data-quality report from the committed M1 code |
| `c30b6b87f1fe2514e07659f68739b47a563d2e03` | `e319562274b5da99d91c348bd4d84171599b4e19` | FRED data, copyright tags and protocol v2 |
| `44f0c536aab2028f83a4f32333a5e34b5cd7a3f4` | `2c437afb53ab8a5d36044f8e6e8984cb675d2fb2` | M2: estimation core (LP, event-sample LP, LP-IV) with simulation recovery |
| `dc8617798efc2f317d284a2d785f7d332f2eee6a` | `f866ae8b4dc938b96c00b3537c86231f96d26d56` | M3 code: event windows, placebo, BoC proxy validation, replication claims |
| `1b641614150c6be27a8f8a09fecbdd113c1491ca` | `2ad2ae566f01767644db37a0263ef854c9bb5e53` | Fix line length in identification test |
| `9e55fabd57f37be8d35d2c138d25bd61105733f3` | `8e361f1e1238ab3fc159cbffde86a7e4e4b7c9a5` | M3 analysis notes, O-1 resolution logic, reproducible preview page |
| `9106abdd43731e96b4ebb81cbc66c317130ac6b2` | `5ceb058edbec8d355747427ba2bb927280602297` | Provenance: generated outputs do not mark the code dirty |
| `878007d51fade97c577e45868af492f08542aae4` | `8c54081027e3338ecbfe90b73cf2a1676d0b5f39` | M3 outputs: replication, BoC proxy validation, timing check, preview page |
| `04c9699273515e890d66aa80aaccc5e55c9fef79` | `b79e0a3820ba6afca37c2c62818ce3853de3f713` | Protocol v3: M4 specifications fixed before any M4 estimation |
| `34893d78e55d86bd1dcdc08f11866f64fa3fac18` | `7743b338f1d6ec2362bfb50985c729955ca065cc` | M4 code: proxy BVAR, state dependence, Canada lens, exporter panel, network, robustness |
| `ecc870be89de1b1a4d2457258ac0a1530bf0902a` | `c8d312e4f3fb319073f99516ba04295049cf43a3` | M4 report: weak-instrument flags, Anderson-Rubin sets, notes written after the run |
| `ed293ca2cb87c3703f7c73e6c68093b1568a7fa6` | `63f7b6282354b5971bd0d72db6e87ddc8c3ca7b9` | M4 outputs regenerated at a clean commit (RESULTS, SIMULATION, REPLICATION, results/, preview) |
| `e2ea81a4ca22e713c4049953579e0c4acc617218` | `8f60f73cee96336759d99c57334668ede7020686` | M5 code: event dataset, benchmarks, MLP, metrics, guarded final evaluation |
| `ad8c4e5ce887fba4235bf42b2733d453ef336929` | `830c68fb03882528817d14bf804ffbdf7db38d79` | M5 validation phase: candidates chosen on 2015-2018 (before any test-set use) |
| `794dceb9bf53299e60049a4664a50d48db6c1864` | `aac549d15cce112d9f2a518c83c9be26b8cf1143` | M5 final test evaluation (run 1): No evidence ML adds value |
| `30f4199eee054c697ed5b8465fe18d88442723bb` | `255283aad1252abfad4a85786ec645d3ade787d8` | M6: response catalog, scenario engine (with TS fixtures), ML-implied responses, site export with licence filter |
| `24a30c67ab10cf17d3688abd15733629d7b9c66c` | `30731948e4c15b774cc141dfd82c0abd787550c0` | M7: static web app (six views, scenario engine, brief) and split site data |
| `1356edb19cf55dadb42f3e86a542d43b3e9c77bd` | `4eaad322b4851d1e74d1858df177d041b123383e` | Web: self-hosted fonts, parallel data prefetch, compact replay file, a11y and load checks |
| `81c7925cf26e9a6f2874a55d75ddc90511b7926e` | `7d9864622e55ea4fa2e0377c0f40aefaeabea97b` | M8: live pre-registered mode (protocol v4) |
| `d974f4d665f81099c0f3348214972f5472496834` | `d485d5b8ebc5e5db44a2df7ecb6eebb1a00e09f3` | git_sha: ignore live outputs, site data and all generated reports when checking for uncommitted code |
| `ff0403185570c6a4d61152c174c55327f7988386` | `f1cc5b27a81ea3c39f8d0da123c07b5565c28afd` | Live: pre-registered forecasts for the 2026-10-28 FOMC and BoC decisions (collision day) |
| `5429aa5b40ff5b09734552a8d7f5fc117b35a19a` | `cc76ed9a96e87f5166a226e954cb6e7c4b2d1cfd` | CI: web typecheck/test/build/a11y job; Pages deploy; daily live workflow; CPU-only torch on Linux |
| `775e7114bb18bf4b70e6d0fba800f3dc387703b1` | `c6c3cd3532637135acfe2b8b802acdd4d2c99f8a` | Docs: generated README findings, research note (+PDF, figures), resume bullets; limitations, reproduction guide, ship checklist |
| `bcf104eed163882eb78c24a0da4211a0e191dc9e` | `7b925ab387e62e12c45de015c7ecc3c9ef8d96c5` | Reproduction check: full pipeline rerun from the pinned cache reproduces every result |
