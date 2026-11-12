# Phase 0, carry the Flash-Next batch path onto main (re-scoped by the 2026-11-11 run)

Back-link: [overview.md](overview.md).

The original scope, `git merge main` into the batch branch and prove parity,
died in the conflicts. This file records why, what the run measured instead,
and what phase 0 actually costs.

## As found (2026-11-11/12, the quiet-window run)

- The merge (`batch-rebase` worktree) hit 37 conflicts, and the shape of them
  was fatal to the plan: `main` replaced the serving layer under the branch.
  `b00d6da64` (2026-10-30, "Phase 6") deleted `serve/` (Python) for the C++
  `lamina-server`, and main's `generate.cpp` never had `--parallel`/BGEN
  (`git log -S` is empty). Half the conflicts were two batch designs meeting.
  The merge is aborted; `batch-rebase` sits at the harness tip, rerere cache
  kept.
- main's `parallel` / `--batch` is the q36 path's: `docs/USING.md:320` says
  "the batch slots of Qwen3.6 and Qwen3.8-27B". Runtime agrees: the v31
  Flash-Next config with `"parallel": 4` booted `lamina --serve ... --batch 4`
  and the engine exited 2 on the unknown flag (`sweep.log:143`). Flash-Next
  concurrency on main is serial queueing: 4 clients at 107.9 total tok/s, each
  waiting its turn.
- So the harness branch remains the only Flash-Next CUDA batch implementation
  (--parallel serve mode, BGEN protocol, batch verify windows, `--batch-mtp`,
  the 16-row kernel work), and it rides a server main deleted.
- Fresh baseline on the current tree, pinned config (expert-cache 7440,
  second-gpu-gib 21, frozen profile), medians of `tools/lamina_bench.py
  --clients 4 --long`, with the off-arm's repeat as the spread: solo decode
  148.1 (124.4-183.8; repeat boot 138.6), 4K prefill 2946.5 (repeat 2948.0),
  16K 3545.9 (repeat 3512.2), 32K 5143.0 (repeat 5147.9), 4-clients 107.9
  total (serial), agent cache hit 70.6%. Prefill repeats within 1%, solo
  decode drifts ~7% boot to boot: the numbers are reproducible only under the
  deterministic recipe (`--adapt-every 0`), which is also why the overview's
  ±3% speed gate cannot be met on the prod-shaped config. The old program's
  tables sit at 126-127 solo tok/s on the pre-rewrite base; every one of its
  numbers needs restating against this.
- Robustness find, logged not fixed: `"parallel"` on a Flash-Next config
  crashes the engine at boot. `src/server/config.cpp` (the #465 block) appends
  `--batch` without a model-kind check; a refuse-with-warning is the honest
  behavior. Upstream report or one-line fix.

## What carrying the batch path actually is

A merge is off the table; it would graft the dead Python server onto a tree
that replaced it. The honest shape is a port, designed as if the slot routing
had been built on the C++ server from day one:

1. Engine (`src/program/generate.cpp` + cores): the `--parallel` serve mode,
   the BGEN/BT/BDONE protocol lines, draft trim and commit, batch verify
   windows, per-slot MTP draft stores. The harness's 52 commits are the
   reference; re-anchor every hunk against main's rewritten solo path.
2. Server (`src/server/`): Flash-Next slot bookkeeping. Either adapt
   `engine_batch.cpp` (already speaks BGEN to the q36 engine) to the
   Flash-Next engine, or write the Flash-Next path beside it. The harness's
   `serve/server.py` slot logic is the semantics spec.
3. Kernels: the wide-window work lands as upstream would take it (the 8-row
   default stays; the 9-16 materialization and parity tests follow the
   Phase 4 economics probe, not the harness's default flip).
4. Bench: `tools/lamina_bench.py` gains the missing shapes (concurrent
   long-warm replay, per-slot acceptance telemetry) instead of resurrecting
   `bench-batch-ab.py`.

Gates per the overview, restated: byte-identical digests with batch off vs
prod on the deterministic recipe, speed within +/-3%, then the client-4 short
and warm-replay tables on v31.

## As run, warm replay and the CPU prefetch (2026-11-12)

A 4 x 40K-token greedy replay (`.e2e-warm.sh`: conversations seeded, then
decoded in concurrent follow-up turns, thinking off the way
`tools/lamina_bench.py` requests it) against `LAMINA_IQ_PREFETCH` = 2048 (the
code default) / 0 / 8192, three rounds per state, states alternating, first
run of the first boot flagged cold. Per-state means over the six rounds:
2048 -> 38.6, 0 -> 35.0, 8192 -> 33.1 total tok/s, with within-state spread
24.8-50.6 - the state ordering sits well inside the spread, so the flag gets
no win on this box and the default stays. (The first pass of this probe was
re-run and replaces it: it left thinking on, which filled `completion_tokens`
with reasoning and some saved answers empty, and its headline band excluded
the cold first boot without saying so.)

Why no win is not established. The `/metrics` `hit_rate` (~0.98 here) counts
prompt-cache reuse, not expert residency; the residency signal is
`expert_tiers`, which showed 1100-2200 of 2700-6000 second-tier experts
CPU-parked with host CPU near 40% during decode, so CPU-computed experts are
present. Prefetch distance not moving those numbers is unexplained; revisit
with per-tier expert counters before claiming a mechanism. Caveats kept
honest: six rounds per state is thin, one cold boot excluded as flagged, and
the totals swing with answer length round to round.

What the same run settled:

- Warm concurrency hurts hard without batch slots. 4 concurrent warm 40K-token
  follow-ups with thinking off: 25-50 total tok/s with first-token medians
  4-5 s (25.7 s on the cold first boot), while solo decode stays 150-190
  tok/s. With thinking on the totals read higher (100-112) only because long
  reasoning answers amortize the serialized prefills. This is the pain the
  batch port would fix, now quantified.
- The digest-gate machinery works, and placement drift was the diverger, not
  a numerical bug: two fresh boots of the same config with `--adapt-every 0`
  answered byte-identically in all four sessions, while the `--adapt-every 1`
  state runs legitimately diverged. Digest gates must use the deterministic
  recipe; that recipe is now demonstrated end-to-end on this rig
  (`lamina-e2e-det.json`, the d1/d2 pair in `warm/`).

## Cost

Days of focused porting across engine, server, and kernels, with the exactness
harness re-derived as it lands. This is a program decision (is Flash-Next
batching worth days on this single-user box, given the warm regime loses even
when it works?), not a phase to start mid-window.
