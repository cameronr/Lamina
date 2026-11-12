# Phase 3, exact two-stage main head

Back-link: [overview.md](overview.md).

Lands on `main`, not the batch branch. It is a solo-path win (`plans/external-engine-ideas.md`
item 6). The 2026-11-11 verdict "dead for sampled traffic" was wrong and is
retracted below; the work is alive and gated.

## Correction (2026-11-12): why the drop was wrong

The drop read the call site: `sample_tokens` is handed full `T x n_vocab`
logit rows, so "bound-and-prune cannot skip the read". It did not read what
the sampler does with them. The sampler's pipeline is
`penalties -> top_k -> top_p -> min_p -> temperature -> pick`
(`src/kernels/cuda/sampler.cu:3`), and its shortlist is capped at 64 rows for
every config, "off" included (`sampled_k`, `sampler.cu:393`); the split top_k
merge is exact because the first k of a union are within the first k of each
part (`sampler.cu:563-567`). So the pick depends only on the exact top-64 of
a row plus whatever rows the acceptance path names. A conservative screen
that certifies the exact top-64 can skip most of the 248k-row read and stay
byte-identical. Two consumers do need whole rows and fall back to the full
read: `copy_logits` (`verify.cpp:1414`, API logits) and `candidate_logprobs`
(`verify.cpp:1426`, full softmax). Another fact fixed here: prod does **not**
run the split head, the config passes `--head-split 0`; the split path
(`shead_`) is an option, not the default.

## Design v2, norm screen, not the block-max store

The original bound store (per-row, per-block max weight, ~159 MB, loose) is
replaced. Per row keep one fp32 norm `||w_r||` of the dequantized weight row
(1 MB total) and a static permutation of rows sorted by norm descending
(computed once at load, one dequant pass, boot cost printed). The dot the
engine computes is on the q8-quantized head input `xq`, so
`|w_r . xq| <= ||w_r|| * ||xq||` in the very arithmetic the kernel uses
(Cauchy-Schwarz, one safety epsilon for accumulation order). Per window
column, score rows in norm order keeping the running 64th-largest exact logit
`tau` (forced rows - the draft tokens' ids - are always exact-scored); the
column is certified once `||w_next|| * ||xq|| < tau`. The survivors plus
forced rows feed a shortlist entry point into the sampler that reuses
`sampled_tail_warp` verbatim, so the tail arithmetic is the same code, run on
a set that provably contains the sampler's own shortlist. Prune falls back to
the full read for `logits_wanted_`, `candidate_logprobs`, and non-zero
penalties (penalties move logits by per-id constants before top_k; prod uses
none, supporting them is later work).

## Gate order, cheapest evidence first

1. **M0, head share, stock binary.** `--stats` prints `head %.3f ms` per
   window already (`print_profile`, the `w[3]-w[2]` stamp). If the head is
   under ~2% of the window at solo decode, stop; nothing downstream earns it.
2. **M1, survivor histogram, no behavior change.** With the norms and the
   permutation built (load-time, as above) and the full read kept, simulate
   the screen per window: count the rows the bound would let through at the
   real `tau`. Print the histogram with `--stats`. Kill rule: if the median
   survivor count is over ~20% of the vocabulary, the bound is too loose;
   record and stop (or revisit with a rank-`r` proxy: stage-1 logit from a
   low-rank head factorization plus per-row residual norms, much tighter,
   real boot cost - only if the simple screen is close but not enough).
3. **The real path**, behind `--head-prune off|auto` (off default through
   its own soak): the screen replaces the read, shortlist sampler, forced
   draft rows, acceptance-path audit (find every reader of the head's rows
   and put it on the forced list or the fallback list).
4. **Gates.** Unit: random `xq`, screen-certified set vs brute top-64, equal
   including tie-break ids, plus the forced rows always present. Digest: the
   deterministic recipe (`--adapt-every 0`, the pair already proven to
   reproduce), byte-identical `auto` vs `off`, sampled config included, not
   just greedy. Speed: `tools/lamina_bench.py` steady 3 rounds + the warm
   replay; the win must show at solo decode tok/s. Mechanism: the M1
   counter still ships, survivors per column in the report.

## What is precedent and what is not

`src/core/mtp.cpp` scores its draft head through a 2-bit copy
(`LAMINA_DRAFT_PROXY`, ~line 309) and rescores neighbors exactly. That is a
lossy proxy: its copy can rank the true argmax out of the neighborhood, and
`plans/external-engine-ideas.md:81` says by name not to make that the main
head's default. The exact variant, conservative bounds that the sampler's own
shortlist can never fall through, has no in-repo precedent. The bounds are the
new work; the rescore-and-merge plumbing after them is shape the MTP path
already shows.

## Changes

1. **Norm store (M1's first half).** At head load, dequant-stream the rows
   once into fp32 row norms (`248k x 4 B`) and the descending permutation
   (`248k x 4 B`). Both live in VRAM (about 2 MB, printed with the boot
   costs). The head loader already walks every row; the pass is seconds.
2. **Screen simulation (M1).** Keep the full `native_mmvq` read. New cheap
   per-window kernel: from the exact logits take the 64th-largest per column,
   count rows whose norm bound clears it, bucket into a histogram printed by
   `--stats`. This is the real screen's arithmetic with none of its risk; the
   histogram is the go/no-go.
3. **The real path.** Behind `--head-prune off|auto`: the screen kernel walks
   the permutation, raises `tau` per column as exact scores land (chunks of a
   few thousand rows, gather-GEMV into a survivor buffer), forced draft rows
   appended, then argmax and `sampled_tail_warp` run over the survivor list.
   `copy_logits` / `candidate_logprobs` / non-zero penalties take the old
   full-read path. The split head (`--head-split`, off in prod) is out of
   scope until the solo path soaks; when it lands the norms shard with the
   head and the screen runs per shard before the join.
4. **Counter.** Survivors and exact rows per column ship in the report, so
   the A/B shows the mechanism, not just the speed.

## Data structures

Two arrays computed at load (norms, permutation), one survivor buffer and one
`tau` vector per window. No new engine types.

## Verification

- **Static.** ctest; a unit test draws random `xq`, brute-forces the top-64
   with the engine's tie-break order, and asserts the screen's certified set
   contains it, the forced rows are present, and `sampled_tail_warp` on the
   survivor list picks identically to the full-row sampler.
- **Runtime, exactness.** Deterministic-config digest byte-identical `auto`
   vs `off`, sampled params (temperature 1.0) included. A digest move here is
   a correctness bug, not noise.
- **Runtime, speed.** `tools/lamina_bench.py` steady 3 rounds + warm replay;
   the win must show at solo decode tok/s. Kill rule: under 1% measured
   steady, stop; the boot pass and 2 MB are not earned.
- **Runtime, mechanism.** The M1 histogram travels with every speed claim.
