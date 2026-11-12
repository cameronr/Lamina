# Program 02, getting batching shippable, plus two decode wins

Back-link: none, this is the overview. The phases numbered here are internal to
this program; they do not line up with `plans/01-engine-batching/` phase numbers.

## Context

The whole batch path (old program phases 1-10, 52 commits) sits on the
`batch-phase1-harness` branch and has never shipped. Prod runs `main`, which
knows nothing about it: `kVerifyMaxT` is 8 there and the batch slots do not
exist. The branches diverged at `dd0390fbe` (2026-10-23); `main` has moved 181
commits since, including the strata-to-lamina rename (`ea809273d`), and the
branch tip (`1096c382e`, 2026-10-31) predates all of it. The measured case for
batching (2026-10-30/31, old config, `plans/01-engine-batching/phase-11-long-context-batching.md`):
+19% total tok/s on short prompts at 4 concurrent, -16% on warm 64K sessions,
and an over-full K/V pool exits the engine, which is why no one dares enable it.

Three facts set this program's order:

1. Nothing ships while a fourth long request can kill the server. Admission
   control (old phase 11 item 1) is the gate, and its design is ported from
   Niko `docs/BATCHING.md`, so it is proven, not invented.
2. The real traffic (llama-swap, 27 days: 89% of prompt tokens cached, p50
   resident context 82K, 44% of requests arriving during another) is the shape
   where batching loses. The win only has to apply where batching wins, so the
   shipping default is a context gate, not a residency rescue.
3. The branch cannot be trusted as a baseline until it carries today's main.
   181 commits of solo-path work landed under it unseen, and 47 of the 83 files
   the branch touched were also touched by main. Before building on it, prove
   it byte-equal to prod with batching off and re-measure its headline numbers
   on the current v31 model.

Two decode items ride along because they were picked in the same review: the
exact two-stage main head (a `main`-branch win, independent), and a one-day
probe that decides the wide-window kernel question with data instead of a
build. External numbers get no benefit of the doubt here: the disjoint cache
was predicted +15.5% and measured +2.1%, and upstream's same-day sampled-draft
work (`ec30341b3`) found the draft-selection part worth 0.2 points. Measure
before building.

## Scope

In: merge `main` into the batch branch and prove baseline parity; recreate the
pinned A/B configs; admission control; a context gate for batch eligibility;
the exact bound-based main-head verify (`main` branch); a wide-window
economics probe with a build-or-kill verdict.

Out, each with a prior decision behind it:

- Expert residency under concurrency and context-priced drafts. Old phase 11
  items 2-3; the old doc's own Non-goals accept the physics ("accepted
  physics", the honest lever is a slot-count recommendation).
- Relaxed drafting and any sampling-draft work. Upstream is on it
  (`ec30341b3`, 2026-11-11); do not race it.
- A rebase onto Niko main. The nightly watch says this fork is +75.0% (2026-11-06,
  `~/strata-results.md`); revisit only if the ratio crosses ~1.15.
- Tree drafting, FP8, `--kv-resident` streaming.
- Ship decisions. This program ends at shippable-with-gates; enabling batching
  in prod is a separate call.

## Constraints

- Any boot of both cards stops prod. GPU work goes through
  `~/lamina/run-e2e.sh <checkout> <script>`, one atomic foreground command per
  run; it pauses llama-swap inference, and the next generation 503s until it
  finishes. If a run is SIGKILLed, recover with
  `curl -X POST localhost:8020/api/models/resume`. Schedule a maintenance
  window; batches of runs in one window.
- Exactness runs use the deterministic recipe from the fork notes:
  `--adapt-every 0`, `STRATA_IQ_MT_MIN=1`, `--pcie-frac 0`, `--expert-cache`
  pinned below free VRAM, fixed `--second-gpu-gib`,
  `--second-gpu-prefetch-mb 0`, `--no-prefill-borrow`, and a frozen copy of the
  learned profile (prod rewrites `/data/unsloth-ud-iq4_xs-v31-learned.bin`
  continuously; never point a gate at the live file).
- Perf A/Bs use byte-pinned configs so two boots of one arm match, and at
  least 3 alternating rounds, because two rounds missed the 16-row window
  regression.
- Scratch files live in the checkout or `/tmp/opencode`, cleaned after. Never
  write the chat-template control tokens literally into any file or message;
  build them from parts in code.
- Fork convention: develop on fork branches, no upstream PRs for local phases.

## Alternatives

**Carrying main into the branch.** A three-way merge is the pick. The branch
touches 83 files, 47 of them also touched by main, and main renamed every
`strata` path midstream: a 52-commit rebase replays those files onto a tree
where their paths no longer exist and turns one rename into up to 52 rounds of
the same conflicts. Merge detects the rename once and resolves it once, and the
phase history stays bisectable. If linear history matters at landing time,
squash then, the way `wt merge` already does.

**Making batching honest.** Ship the measured +19% regime behind a context
gate (this program) versus chase the -16% long-context loss through residency
tuning (old phase 11 items 2-3, already flagged as a likely negative result by
its own doc). The gate ships the win that exists.

**The wide-window kernel.** Build the 9-16 column interleaved path directly,
versus one day of telemetry with the existing `STRATA_BATCH_WINDOW` knob to
first check that deeper drafts at 4 slots outweigh the extra rows. Phase 4
settles this; phase 10's doc attributes the 4x regression to two causes and
only one of them is the kernel path, so the ceiling above window 8 is a
hypothesis until something measures it.

## Applicable skills for the implementer

- The **how** skill (Explain mode) over any subsystem you touch cold: the
  `kvg_ensure` path in `src/program/generate.cpp`, the slot routing in
  `serve/server.py`, the head scoring in `src/core/verify.cpp`.
- **principle-sequence-verifiable-units**: each phase ends in a named gate, and
  the next phase does not start on a failed one.
- The **unslop** skill on every diff and doc before commit; the **no-comments**
  skill before review.
- The **show-me-your-work** habit via the repo's own convention: append an
  `As built` section to the phase file with the real numbers.

## Phases

- [Phase 0, carry the batch path onto main](phase-0-merge-and-baseline.md),
  re-scoped by the 2026-11-11 run: the merge died because main replaced the
  serving layer; what remains is a multi-day port decision, with the fresh v31
  baseline recorded (solo decode 148.1, repeat boot 138.6 - decode needs the
  deterministic recipe to repeat; 4-clients-serial 107.9 total).
- [Phase 1, admission control](phase-1-admission-control.md) - blocked behind
  the port, and reframed by its forensics: on main, over-cap is a clean 400 at
  the API and within-cap exhaustion is unreachable at prod sizing (a 248K-token
  request was served by yielding 1503 expert-cache slots, decode falling to
  the K/V-bound 33 tok/s floor). What is left is QoS under batch slots, part
  of the port's design, not a crash hunt.
- [Phase 2, the context gate](phase-2-context-gate.md) - already superseded on
  `main`: the harness's gate landed in the new server's design (the queue plus
  the engine's lazy start, `src/server/engine_batch.cpp`'s `admission()`;
  `docs/USING.md:248-251`). Kept as the reference for when the batch port
  needs the same semantics on slot admission.
- [Phase 3, exact two-stage main head](phase-3-exact-main-head.md) - killed
  with measurement on 2026-11-12, after one wrong death (sampler semantics,
  not call sites) and one resurrection. The Cauchy-Schwarz bound sits ~20x
  over the 64th-largest logit for every head row: nothing certifiable.
  Ceiling was 4-6% decode. The measurement kit (norm store, survivor
  histogram, parity tests) is on `main` for the next bound idea to answer
  the same question in one run.
- [Phase 4, wide-window economics probe](phase-4-window-economics-probe.md) -
  needs a Flash-Next batch path to probe; blocked behind the port.

Order after the run: the port decision is the only door - Phases 1 and 4 wait
behind it, Phase 2 already landed, Phase 3 is dead for sampled traffic. What
was measured instead, this window: the fresh baseline, the warm-concurrency
cost of serial serving, and the CPU prefetch flag, in the phase 0 file.

## Verification

Phase 3's gates are in its file. The run infrastructure the port will reuse,
built this window: `lamina-e2e:latest` now carries build deps;
`.e2e-build-sweep.sh` builds a checkout container-locally and runs
`tools/lamina_bench.py --clients 4 --long` arms against pinned configs
(`lamina-e2e-off/on.json`, frozen profile at
`~/strata-data/bench/2026-11-11/`); results land under `bench/results/` and
`~/strata-data/bench/`. Everything goes through `~/lamina/run-e2e.sh` in a
maintenance window.
