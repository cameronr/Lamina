# Phase 3, exact two-stage main head

Back-link: [overview.md](overview.md).

Status 2026-11-12: **killed with measurement**, after one wrong death and one
resurrection. The measurement kit stays in the tree.

## The arc, so nobody re-litigates it from memory

1. 2026-11-11: dropped as "dead for sampled traffic" because `sample_tokens`
   consumes full logit rows. Wrong: that read the call site, not the
   semantics.
2. 2026-11-12: alive. The sampler's pipeline is
   `penalties -> top_k -> top_p -> min_p -> temperature -> pick`
   (`src/kernels/cuda/sampler.cu:3`) with the shortlist capped at 64 rows for
   every config (`sampled_k`, `:393`), so the pick depends only on the exact
   top-64 plus the acceptance path's named rows. A conservative bound can in
   principle certify the rest away, byte-identically. Prod also does **not**
   run the split head (`--head-split 0`; the old version of this file claimed
   otherwise).
3. 2026-11-12, later the same day: killed by the gates below. The bound that
   certifies is also the bound that costs.

## Gate 1 (M0), head share, stock binary

`--window-profile` prints `head ms per window` (`print_profile`). A
4K-prompt, 300-token generate on the pinned config: **head 0.592 ms of a
14.623 ms window, 4.0%**; the head's 521 MB read in 0.59 ms is bandwidth
bound, so skipping rows would cut it linearly. Full-elimination ceiling
4-6% decode. Bar was 2%: pass, proceed. Log:
`~/strata-data/bench/2026-11-12/m0-generate.log` (`.e2e-m0.sh` in the main
checkout).

## Gate 2 (M1), could a bound prune, measured beside the full read

The kit (branch `head-prune`, commit `2ec502664` on main; kernels
in `src/kernels/cuda/native_head_screen.cu`, hook in `verify.cpp` behind
`LAMINA_HEAD_SCREEN=1`, parity test `head_screen_parity`): one pass at load
computes each dequantized head row's L2 norm (248320 rows, 144 ms, 1 MB);
per window it takes the exact 64th-largest logit per column (bit-level radix
select), the window's q8_1 input norms, and histograms the rows whose
Cauchy-Schwarz bound `||w|| ||x|| x 1.0001 + 1e-4` clears it. Nothing else
runs differently.

Result: **every window column lands in the top bucket (>= 262144 rows of
248320): the norm bound certifies nothing.** 593/593 columns.
`~/strata-data/bench/2026-11-12/m1-generate.log`.

It had to, and that is the general lesson: a head row's dot with the window
input sits at typical magnitude `s`, the 64th-largest of 248K such logits
around `3.6 sigma` of that spread, while `||w|| ||x||` sits near
`sqrt(5120) s` - about twenty times `tau`, for every row. Certifying
non-membership in the top-64 needs an approximation whose *error* is under
the 64th-to-65th gap, not under the typical magnitude. A block-max bound
lands ~10x `tau` (same failure, milder); a rank-`r` residual proxy would
need residuals five times smaller than the full norms, which LM-head weight
spectra do not do. The kill rule's escape ("close but not enough") is not
met, so the proxy is not funded.

## What this item leaves behind

- `LAMINA_HEAD_SCREEN=1` + `--window-profile`: the survivor histogram and the
  norm pass, for any future bound scheme, norms or otherwise, to answer the
  same question in one run before anyone builds plumbing.
- `head_screen_parity` in the kernel test suite: the radix-select k-th
  largest (exact bits, ties, 65..200000 rows, all-negative columns), the
  q8_1 input norms and the reference-dequant row norms against host
  arithmetic, and the histogram's bucket math.
- The rule this item paid for: **for sampled decoding with a capped
  shortlist, a head-skip idea lives or dies on the bound-vs-tau ratio at
  the 64th position; measure that ratio before touching a sampler.**

## What is precedent and what is not

`src/core/mtp.cpp` scores its draft head through a 2-bit copy
(`LAMINA_DRAFT_PROXY`, ~line 309) and rescores neighbors exactly: a lossy
proxy that can rank the true argmax out of the neighborhood, and
`plans/external-engine-ideas.md:81` says by name not to make it the main
head's default. The exact variant had no precedent and now has a measured
reason not to. The head can still get faster - better quantization of the
read itself, kernel efficiency - but that is not skipping rows, and it is a
different item.
