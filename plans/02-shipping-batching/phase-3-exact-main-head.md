# Phase 3, exact two-stage main head

Back-link: [overview.md](overview.md).

Lands on `main`, not the batch branch. It is a solo-path win, independent of
Phases 0-2, and can start any time.

## Goal

Stop reading all 248k head rows every window. Score an interval bound per row,
exact-score only the rows the bound cannot eliminate, keep the argmax provably
identical. Worth ~0.4-0.6 ms per window at 126 tok/s decode: a couple of
percent, and it is arithmetic, not a hardware bet (`plans/external-engine-ideas.md`
item 6).

## What is precedent and what is not

`src/core/mtp.cpp` scores its draft head through a 2-bit copy
(`LAMINA_DRAFT_PROXY`, ~line 309) and rescores neighbors exactly. That is a
lossy proxy: its copy can rank the true argmax out of the neighborhood, and
`plans/external-engine-ideas.md:81` says by name not to make that the main
head's default. The exact variant, conservative bounds that no true argmax can
fall through, has no in-repo precedent. The bounds are the new work; the
rescore-and-merge plumbing after them is shape the MTP path already shows.

## Changes

1. **Bound store.** At head load, one float per (row, block): the maximum
   absolute weight in the block, blocks aligned to the quantization blocks.
   A row's bound against input `x` is the sum over blocks of the block's max
   times the block's `max|x|`; it cannot fall under the true dot product.
   Size it before building: rows x hidden/blocks x 4 B. At 248k rows, hidden
   5120, block 32, that is ~159 MB, against a pinned pool with no slack, so
   the store's home (VRAM beside the head, or host memory streamed like the
   head's rows are today, or block 64) is chosen by measured boot fit, and
   the boot line prints what it cost. The bound pass costs bytes far under
   the head's own rows (4 B per block against the quantized weights' 16-32),
   which is the whole margin being spent.
2. **Bound-and-prune scoring.** `src/core/verify.cpp`, the head section
   (~1011-1022: `native_mmvq` over every row, then `argmax_rows`). New pass
   order: block-max of `x`; bound scan accumulating per-row, abandoning a row
   once its bound plus the best exact so far cannot win; exact `native_mmvq`
   only over surviving rows. `--head-prune auto|off`, default auto once its
   own gate has passed a full soak, off for the A/B until then.
3. **Split head.** The head runs split in prod (`--head-split 0.45`,
   `src/core/split_head.cpp`, the `shead_` join at verify.cpp ~1365). The
   bound store must be sharded with the head and the prune run per shard
   before the join, so the join's input shrinks the same way. Run the **how**
   skill over this subsystem before touching it; the interaction is the
   phase's main risk, not the math.
4. **Counter.** Windows report pruned-over-total rows and exact rows per
   column, so the A/B can show the mechanism and not just the speed (basalt's
   expectation: about 2 exact rows per column).

## Data structures

One bounded float table keyed (row, block), sharded with the head; one small
survivor list per window column. No new engine types.

## Verification

- **Static.** ctest; a new unit test draws random `x`, runs the prune path
  and the full read, asserts identical argmax and identical logits for
  survivors, over quant block sizes 16/32/64 and both head splits.
- **Runtime, exactness.** Deterministic-config digest byte-identical with
  `--head-prune auto` vs `off` (the `--parallel 1` recipe). A digest move
  here is a correctness bug, not noise: the variant must be the exact one.
- **Runtime, speed.** `tools/lamina_bench.py` steady load, 3 alternating
  rounds, plus the warm replay: the win must show at solo decode tok/s,
  where the head read is not amortized over window rows. Kill rule: under
  1% measured steady, the store's boot cost and memory are not earned;
  record that in the As built and drop the store to host-resident or stop.
- **Runtime, mechanism.** Counter shows survivors near basalt's ~2 per
  column. If survivors run into the thousands, the bound is too loose to
  pay; report the histogram in the As built before any tuning.
