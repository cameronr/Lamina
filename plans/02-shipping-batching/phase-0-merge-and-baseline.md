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
  second-gpu-gib 21, frozen profile), medians: solo decode 148.1 tok/s, 4K
  prefill 2946.5, 16K 3545.9, 32K 5259.5, 4-clients 107.9 (serial), agent
  cache hit 70.6%. The old program's tables sit at 126-127 solo tok/s on the
  pre-rewrite base; every one of its numbers needs restating against this.
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

A 4 x 40K-token greedy replay (`.e2e-warm.sh`, conversations seeded then decoded
in concurrent follow-up turns) against `LAMINA_IQ_PREFETCH` = 2048 (the code
default) / 0 / 8192, two rounds per state, states alternating: every state
landed 100-112 total tok/s, differences inside the noise. The per-request
metrics explain it: expert `hit_rate` ~0.98 in this shape, so CPU-resident
experts compute ~2% of the routed work and an L3 prefetch on them cannot move
tokens. That closes the review round's "never tried on the X3D" item:
measured, no win at this residency; worth revisiting only if a shape with real
CPU exposure (single-GPU boot, tier collapse) is what prod actually runs.
Two side observations, useful for the port decision:

- Warm concurrency hurts hard without batch slots: 4 concurrent warm requests
  serialized, first token p50 up to ~26 s cold-boot / ~8-10 s warm, per-request
  decode fine at 150-190 tok/s. That is the pain the port would fix.
- Byte-identity across state runs failed, and it is not evidence of a
  numerical bug: with `--adapt-every 1` the tier placement differs per boot, so
  the greedy text legitimately diverges. A digest gate needs the
  `--adapt-every 0` deterministic recipe; recorded so nobody re-runs this
  invalid gate.

## Cost

Days of focused porting across engine, server, and kernels, with the exactness
harness re-derived as it lands. This is a program decision (is Flash-Next
batching worth days on this single-user box, given the warm regime loses even
when it works?), not a phase to start mid-window.
