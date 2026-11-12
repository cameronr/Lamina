# Read beside decode: a new request's prompt while another request decodes

Status 2026-11-12: baseline measured (P0). The proposal is real but it fixes the minority term
of TTFT on this engine; see "What the measurement settled". Not funded yet.

## The proposal, as received

Split a request into prefill and decode at the protocol level and pipeline them across
requests: decode stays single-stream, the next request's prompt is chunked and its chunks
slot into the running decode's gaps, and the server admits that read as soon as a decode is
running instead of only when it finishes.

## What the code actually does today (verified 2026-11-12)

- Prod runs no `parallel` / `--batch` (`/cfg/lamina.json` in the `lamina` container), so the
  server is on the solo path.
- The native engine's serve loop takes one request at a time: `while (next_line(line))`
  (`src/program/generate.cpp:3395`), commands `GEN` / `GENI` only, and runs prefill *and*
  decode to `DONE` inside the iteration. It has no slots at all: `--batch` / `--slots` are
  parsed only by the other engine, `src/qwen35/q36_engine.cu:4354`.
- The server's slot protocol (`src/server/engine_batch.cpp`: `BGEN`, `BADM`, `BSTOP`,
  `BYIELD`, `kYieldsMax`) is therefore dead code for this model. With no slots, a second
  request waits for the first's `DONE`.

## Three corrections to the received framing

1. **This engine has no "decode falls to 2-3 tok/s" defect.** That figure describes the case
   where the read *does* run beside the decode unsupervised, which is what the old
   `plans/01-engine-batching/phase-8-read-beside-decode.md:7` was written against. Here the
   read never starts, so the running decode is undisturbed. The defect is pure queueing:
   time-to-first-token equals the other request's remaining generation, minutes on a long one.
2. **It is not dead time.** The decode holds about 89% of the main GPU's wall clock: window
   14.623 ms of GPU time, 156.5 tok/s at 2.56 tokens/round = 61 rounds/s = 0.89 GPU-seconds
   per wall-second (`~/strata-data/bench/2026-11-12/m0-generate.log`). Both workloads are
   bandwidth-bound, so a second stream's share comes out of the first's rate near-linearly.
   The 11% outside the verify windows is the only *idle* time; whether a bandwidth-bound
   decode and a heavier prefill co-schedule better than linearly is an open question, and it
   is the single biggest unknown in the sizing (gate P2).
3. **Warm reads are already nearly free, so the cliff is a cold-prompt phenomenon.** A repeat
   of a 10,459-token prompt cost 0.09 s to first token (P0, `read-beside.log`). Any design
   must not disturb prompt-cache hits, which is where today's cheap wins already are.

## The trade, from measured rates

Solo decode 156-162 tok/s; cold prefill 2,300-3,076 tok/s. Hand the reader a fraction `f` of
the GPU and B prefills at about `f x 2800` tok/s while A loses a comparable share: `f = 0.3`
is B reading a 32K prompt in ~39 s with A around 110-120 tok/s, against today's A at full
rate with B waiting out A's entire generation. This is a model, not a measurement; P0 and a
prototype are what make it real.

## Scope: this is much smaller than the parked batching port

Needed: per-slot session state, a preemptible chunked read beside verify windows, a thin
engine command, and server early-admission. Explicitly *not* needed: multi-sequence verify
windows and the segment kernels, which were the expensive half of
`plans/01-engine-batching/` (phases 3-4). Decode stays single-sequence, so the kernels that
made batching hard are untouched.

Phase 2 of that plan (the slot state machine, per-slot GDN state, K/V arena, PLE history,
draft state) is already validated on this rig (2026-10-26, `batch-phase1-harness`):
`slots_test` passing, sizing table printed, **0.56 GiB per slot at 32K context, and a clean
fit refusal at prod's expert-cache sizing with 0.72 GiB free**.

## Design space, cheapest first

- **A. Server-side conversation preemption, no engine change.** STOP the decoding request,
  park it (K/V to host: 0.56 GiB at 26.7 GB/s is ~21 ms), run the queued request, re-issue
  the stopped one with the ids it already produced and let the caches restore it. B's wait
  drops from "A's whole generation" to "B's own read", and A pays a park/restore per switch.
  Gating question: is a resumed conversation byte-identical to an uninterrupted one? That is
  testable on the phase-1 digest harness before any of it is built.
- **B. Host-staged read.** The engine reads B's prompt in chunks between A's windows, with
  B's K/V parked in host RAM between chunks and staged around each one. No VRAM growth; PCIe
  traffic about 1 GiB each way for a 32K read in 16K chunks (~0.1 s, overlappable with
  decode). Costs engine work in the park/restore path plus the interleaving.
- **C. Concurrent-KV read (the original phases 2 + 8).** B's K/V resident beside A's. Needs
  about 0.6 GiB trimmed from the expert cache at 32K, and 4.5 GiB at prod's 262K max context,
  so its price is a decode-rate question (gate P1).
- **D. Peer-card read.** Run the reader's chunks on the second GPU: during A's decode that
  card spends 1.8-5.2 ms of the 14.6 ms window on A's rows, its expert cache holds 9,233
  experts (21.00 GiB), and the prompt path already borrows 577 slots there, and
  `--prompt-split` already runs a prompt's layers 20-47 on it. It leaves A's expert
  residency on the main card undisturbed, which is the tax every other option pays. The old
  phase-8 doc put the reader on the same cards at low stream priority and never considered
  this; stream priorities do exist in the tree (`src/prefill/split.cpp:465`).

## Gate order

1. **P0, measured 2026-11-12.** Driver `.e2e-readbeside.py`, log
   `~/strata-data/bench/2026-11-12/read-beside.log`, shipping binary, prod-shaped config.
   A generates 4,579 tokens (25.3 s decoding, 171 tok/s average, steady 178-190 per 2 s bucket
   the whole time B and C queued: premise 1 confirmed, the decode is untouched). B arrives at
   t=10.0 with a cold 29,022-token prompt; C arrives at t=12.0 with 278 tokens.

   | request | prompt | arrived | TTFT | queue for a decode turn | a read ahead of it | its own read |
   | --- | --- | --- | --- | --- | --- | --- |
   | A, decodes 4,579 tokens | 4,114 | 0.0 | 1.53 s | - | - | 1.4 s |
   | B | 29,022 | 10.0 | **22.6 s** | 16.8 s | 0 | 5.8 s |
   | C | 278 | 12.0 | **21.6 s** | 14.8 s | 6.8 s | ~0.1 s |

   Same 29,022-token prompt a second time: **TTFT 0.14 s**. The same prompt alone, cold: 6.6 s
   (4,400 tok/s).

2. **P1, conversation resume fidelity and cost (option A's gate).** Measured 2026-11-12, results in
   the section after this list. Cost answered (cheap: 0.16-0.93 s), mechanism answered (it has to be
   ids, not a re-spelled chat turn), identity still open: `.e2e-p1c.py` cuts a generation at 1/2 and
   3/4 of what the model actually produced, small (4K) and big (16K) conversations, so identity and
   how the cost scales with context come back in one run.
3. **P2, the peer card's prefill headroom** (design D), only if P1 shows preemption is not
   enough. Same question as before: how fast can a read run on GPU 0 while the main card
   decodes.
4. **P3, prototype** behind a flag. Gates: solo runs token-identical (the phase-1 digest
   harness), then the P0 driver as the A/B, with warm-hit rate pinned as a guard.

## P1 results (2026-11-12): the preemption card is real, at the id level

Two probes on the shipping binary, engine-direct (`GEN` on stdin, `.e2e-p1.py`) and through the API
(`.e2e-p1b.py`); logs `~/strata-data/bench/2026-11-12/p1.log`, `p1b.log`.

- **A parked conversation's K/V covers what decoding wrote.** Re-issuing a 5,596-token conversation
  (prompt plus everything it had decoded) reached its first token in **0.16 s**, against 1.15-1.57 s to
  cold-read a *smaller* prompt. The restore does not re-read the generation.
- **Cheap with depth too:** 0.55 s resuming at 5,096 tokens, 0.16 s at 5,596, 0.93 s through the API at
  about 6K including tokenization. Against the 15-17 s of decode-turn queue a preemption dodges, the
  switch costs 1-6% of what it saves.
- **A chat-template continuation is not byte-safe, so the mechanism has to be ids.** Feeding the partial
  answer back as a trailing assistant turn was a coin flip: cut at 1,369 chars the model *restarted its
  own count from 1*; cut at 1,200 chars it continued identically for all 500 compared characters. The
  template closes the assistant turn, so whether the model reads "continue" or "answer again" depends on
  where the cut landed. The server holds the ids already, and `GEN prompt+produced` is both the cheap
  path and the only exact one.
- Parking is on by default and prod does not override it: `prompt_cache = 16`,
  `conversation_cache_mib = 16384`, `conversation_cache_min_tokens = 1024`
  (`src/program/generate.cpp:252-264`).

For implementation shape, the server already has the hard part: requests are *legs* in
`src/server/engine_batch.cpp` (the `#1869` carry, `may_go_solo`), which already resume one request across
a STOP and a re-issue. Preemption is a policy on top of legs that exist, holding the client's stream open
while the preempted conversation waits its turn, with the switch gated on the waiting request's work
clearing the ~0.2-0.9 s restore plus a cap on preemptions per conversation so two long generations cannot
ping-pong.

## The host OOM this investigation caused (2026-11-12)

P1c took the machine down. Boot `-1` ends in a kernel OOM report (85 GiB anonymous, 113 MiB free, 86 GiB
of swap untouched) after `lamina: page allocation failure` at 12:49:53, and the sequence is legible: P1b's
wrapper resumed inference at 12:47:56, a request arrived 7 s later and llama-swap did its job and started
prod at 12:48:03, and the e2e container began loading its own 56 GiB pinned arena about a minute after
that. `pause` asks llama-swap to unload but does not wait for the 120 s `docker stop` grace and the arena
to come back, so the two engines overlapped. The earlier runs of the same wrapper were lucky: nothing had
just summoned prod, so there was nothing to wait for.

`run-e2e.sh` (gitignored, machine-local) now refuses to start a second arena until prod is *gone* and
`MemAvailable` is back above 64 GiB, up to 300 s; runs a 3 s watchdog that `docker kill`s the test
container the moment prod returns or free memory drops below 20 GiB mid-run; and caps the container at
72 GiB so an unexpected overlap is a container OOM rather than a host one. `--probe` reports the same
preconditions without pausing anything (`prod running: yes / 23941 MiB free` right now, so a run would
correctly refuse). The `rare, accept it` line in the machine's AGENTS notes describes this same window and
does not mention that on this box it costs a reboot.

## What the measurement settled (2026-11-12)

TTFT on this engine decomposes into three terms, and the proposal only touches two of them:

- **the queue for the exclusive decode turn**, 14.8-16.8 s of the measured 21.6-22.6 s: the
  majority of the pain, and *deliberately untouched* by read-beside-decode, since keeping
  decode single-stream is that design's central simplification;
- **a read queued ahead of me**, 6.8 s for C: this it removes;
- **my own read**, 5.8 s for B: this it overlaps, so B's TTFT would go 22.6 s to about 17 s
  and C's 21.6 s to about 15 s: a real 26-30% cut, not the 3-10x the framing suggested.

And on this box's actual traffic the two read terms are usually near zero already: warm reads
cost 0.14 s for 29K tokens. So read-beside-decode pays for cold big prompts arriving behind a
*short* generation, where the read term is the long pole; it does nothing for the measured
worst case, which is a request waiting behind a long generation.

That reorders the options: the dominant term is the exclusive decode turn, so the cheap move
worth testing is **option A, conversation preemption in the server** (B's TTFT in the scenario
above drops to about its own read plus a decode turn, ~6 s, a 3.7x cut, with A parked for the
same span), and the only design that fixes the term for everyone at once remains decode
batching, which is parked for the expert-working-set reasons in
`plans/batching-reconsidered.md`, not for scheduling reasons. Read-beside-decode stays worth
doing *after* those, or if cold long prompts become the common case.


## Risks worth naming early

- The adaptive tier (`--adapt-every 1`, 80 swaps/round) thrashing between A's hot set and the
  read's broader working set. Measure A's rate *during* the overlap, not after it: the
  batching verdict's "bounded tax" claim is exactly what this risks, and it is bounded only if
  the read ends.
- Prompt-cache hits: an admitted-early read must not evict or double-read what a warm request
  would have hit for free in 0.09 s.
- Preemption granularity: chunk boundaries only (the `BYIELD` rule), so a single 16K chunk
  still blocks the card for its whole duration unless chunks shrink.
- Long contexts: option C's per-slot cost scales with context (0.56 GiB at 32K, ~4.5 GiB at
  262K), which is the argument for B or D on this box.
