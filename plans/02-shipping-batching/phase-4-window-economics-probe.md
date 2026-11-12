# Phase 4, wide-window economics probe

Back-link: [overview.md](overview.md).

Needs the Phase 0 branch. One day of measurement that ends in a written
verdict: build the 9-16 column interleaved path, or close the question.
No kernel work happens in this phase.

## Why this exists

The review round that produced this program treated "windows over 8 columns
fall off the fast path" as the one thing blocking wide windows, with the
ceiling treated as proven. It is not. `phase-10-verify-window-16.md` gives
two causes for the 4x regression: the lost q8_1 path, and "the extra
verification rows outweigh the deeper drafts they let the scheduler batch."
The same-binary A/B (window 8: 104.7/105.5, window 16: 98.0/97.4) does not
separate them; and the 2x row gained only +8% at 10 columns, also on the
slow path. Attention, softmax, routing, and accept scale with rows whatever
path the experts take, so a perfect wide kernel fixes one cause and not the
other. Also register evidence says wide is not free to build:
`gr_up_multi_kernel<16>` already sits at 106 REG / 32 KB. Decide with the
telemetry, before paying for the kernel.

## Changes

1. **Depth histogram.** The batch draft telemetry (`c2f044954`, the BDONE
   drafts fields) counts accepted against offered, not per depth. One counter
   array: accepted draft number 1, 2, 3, 4 at 4 slots, printed with the
   window size. Smallest instrumentation that answers the question.
2. **Two runs, one knob.** Short-prompt 4-slot sweeps (Phase 0's scenario)
   at `STRATA_BATCH_WINDOW=8` and `=16`, same binary, 3 alternating rounds,
   plus the depth histogram on the 16 arm.
3. **Cost side.** The marginal cost of a row inside the fast path, measured
   rather than assumed: the same scenario at slot counts that hold the window
   at 2, 5, and 8 rows, so the fast-path cost curve is known per row up to 8.
4. **The arithmetic, written down.** Expected extra commits per window at 4
   slots from depth 1 to depth 4, from the histogram chain. Compare against
   8 extra rows priced at the fast-path marginal cost. The wide-kernel bet is
   alive only if expected extra commits, at fast-path row cost, clear the
   measured window-8 to window-16 gap. If they do, opening the kernel phase
   gets a predicted gain to be measured against. If they do not, the
   As built closes the CUDA question at 8 rows, matching Niko's independent
   verdict, and `kVerifyMaxT = 16` stays the experiment knob.

## Data structures

One counter array in the batch accept path; a results table appended to this
file's As built. Nothing shipped by this phase but numbers.

## Verification

- **Static.** ctest unchanged (counters are additive).
- **Runtime.** The two runs and the cost-curve runs go through
  `run-e2e.sh` in one window; the histogram must reconcile with the existing
  BDONE drafts totals (sum of per-depth accepts equals the old total, else
  the instrument is wrong and nothing above means anything).
- **Decision gate.** The verdict paragraph in the As built, with the four
  numbers: expected extra commits per window, measured window-cost gap,
  fast-path marginal row cost, and the resulting prediction. No follow-up
  phase opens without them.
