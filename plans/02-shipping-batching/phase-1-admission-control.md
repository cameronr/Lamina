# Phase 1, admission control

Back-link: [overview.md](overview.md).

## Goal

Batching cannot kill the server. A request whose K/V does not fit gets refused
with a countable answer while every other request keeps running. Today an
over-full pool reaches `kvg_ensure` failure and the engine exits; the branch
cannot go near prod until that is gone. Design ported from Niko
`docs/BATCHING.md`; the old program scoped it in
`plans/01-engine-batching/phase-11-long-context-batching.md` item 1.

## Changes

Commit order is red, then green: land the G7 script first and let it show the
engine dying, then the fix on top. A reviewer sees the bug and the proof in
one stack.

1. **The oversubscription script (red).** Add G7 to the branch's gate runner:
   four requests of 120K tokens into the 4-slot arm. Expected today: the
   engine exits. This is the failing test; the phase is not done until it
   passes.
2. **Engine refusal.** `src/program/generate.cpp`, the `kvg` path. Three
   `kvg_ensure` failure sites (pre-merge lines 4421, 4807, 5283; re-anchor
   after the Phase 0 merge) print an `ERR` line the server treats as fatal.
   At request admission instead, compute the need (prompt plus growth margin
   of `max_tokens` plus headroom), let `kvg_trim` give back idle slots and
   parkable conversations first, and answer the protocol with
   `REFUSED <needed_cells>` when the need still cannot be met. The mid-run
   `kvg_ensure` sites keep their `ERR`: a slot growing past its admitted
   margin after the fact stays a fatal engine bug, because it means the
   admission math was wrong, and silent degradation would hide that.
3. **Server mapping.** `serve/server.py`: `REFUSED` maps through the existing
   `STATUS` table to a 400 naming the limit the slot cannot hold, queue order
   stays FIFO, and `/metrics` gains a refusal count. The prompt-too-long
   rejection already has this shape (the 400 path near the `STATUS` table);
   this reuses it.
4. **Startup fit.** Engine INFO block: name each slot that fits at boot and
   the cells it was sized for, as phase 9's slot-fit work already computes.
   Fewer than two fitting slots: batching starts off with a log line, instead
   of carving slots the pool cannot feed.
5. **Batch-arm config.** The `.ab` config raises `--cache-ram` 8192 to 16384;
   the old warm sweep showed the 8 GiB stash cannot hold 4 x 64K sessions.

## Data structures

No new engine types. The batch request record gains a refused terminal state
(next to its done and failed states in the server's slot bookkeeping); the
engine's answer is one protocol line. One new counter.

## Verification

- **Static.** ctest green; the deterministic-config digest gate still
  byte-identical (`--parallel 1`): admission must not move greedy solo tokens.
- **Runtime, G7 (green).** Four 120K requests into the 4-slot arm: four 400s
  with the limit named, the server alive, the refusal count at 4, and the
  other slots still serving (send a short chat request during the storm and
  get an answer).
- **Runtime, startup.** Boot with a pool too small for 4 slots: the INFO
  block names the fitting slots, and with fewer than two, batching is off
  with the log line. Same boot inside the container via the Phase 0 runner.
- **Runtime, give-back.** Saturate the pool with parked conversations plus
  three active slots, admit a fourth whose need is met only after give-back:
  it must admit, not refuse, and the refusal count must not move.
