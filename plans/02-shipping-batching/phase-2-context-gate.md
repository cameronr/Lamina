# Phase 2, the context gate

Back-link: [overview.md](overview.md).

## Goal

Batching engages only where it measurably wins. A request whose resident
context is large takes the one-at-a-time path while short requests keep the
batch slots. This ships the +19% short-prompt regime as the default instead of
waiting for a residency fix to the -16% warm regime that the old program's own
doc lists under Non-goals.

## The open question, first

The phase starts by measuring what already exists, not by writing the gate:
can the serve path run one long request through the non-batch control path
while batch slots stay active and correct? It must, because real arrivals
interleave. If yes, the gate is one threshold check at routing. If no, the
work is making that coexistence safe (it is the same machinery Phase 8 built
in the other direction: prompt reads beside running slots), and this phase
grows to cover it; scope does not move, the estimate does.

## Changes

1. **Threshold knob.** Engine option `--batch-max-context <tokens>` (mirror of
   `STRATA_BATCH_WINDOW`'s shape; env knob allowed for the A/B, default set at
   the measured crossover, 0 disables the gate and restores old behavior).
2. **Routing decision.** `serve/server.py`: at request admission, compare the
   request's resident prompt tokens (the cache-aware figure the warm replay
   already reports) plus its projected output against the threshold. Over it:
   route to the non-batch path, do not touch a batch slot. The decision is
   per request and made once; a session that grows past the threshold
   mid-stream finishes wherever it runs, because re-routing mid-request would
   move K/V.
3. **Telemetry.** `/metrics` and the per-request DONE line gain a batched
   or solo marker, so the mixed-load runs below can be audited from the log
   alone.
4. **Calibration.** With the warm sweep standing as a permanent arm (Phase 0),
   sweep the threshold over 4K, 8K, 16K, 32K on the client-warm replay and
   record the short-wins/warm-loses crossover. Default is the crossover minus
   a safety band, written into `serve/README` and `testing.md` with the table
   that produced it.

## Data structures

One threshold and one per-request marker. No new engine state: the gate
chooses a path at admission and never revisits the choice.

## Verification

- **Static.** ctest; deterministic digest gate byte-identical (`--parallel 1`
  and with the gate off at `--parallel 4`).
- **Runtime, mixed load.** Interleave 64K warm sessions with short chats
  through the server: the marker lines must show shorts batched and warm ones
  solo; no digest drift on either; refusal count stays 0.
- **Runtime, the two headline rows.** Short-prompt 4x row unchanged from the
  Phase 0 re-measure (within round noise). Warm 64K row moves from the Phase 0
  batch number to within noise of the solo number, because it now runs solo.
  Those two rows together are the definition of done for this phase.
- **Runtime, G7 still green.** Admission (Phase 1) and the gate interact at
  the boundary: a refused-then-shortened request must re-enter the same
  routing decision.
