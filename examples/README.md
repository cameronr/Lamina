# Example configs

The configs Lamina is developed with, on a PC with two RTX 3090s, an RTX 5070 Ti and 160 GB of RAM
(their speeds: [COMPARISON.md](../docs/COMPARISON.md)). Setup writes each PC's own configs from them, with its GPUs
and data folder ([SETUP.md](../docs/SETUP.md)):

- `iq3_s.json`: Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S (~53 GB of RAM taken).
- `ud-q4_k_xl.json`: Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL (~81 GB of RAM taken).
- `qwen36-ud-q4_k_xl.json`: Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL, on the engine's qwen35 path: its layers split over
  the fewest GPUs that hold every expert, else on the main GPU with the CPU computing the experts it lacks.
- `qwen38-27b-ud-q6_k_m.json`: Qwen3.8-27B, unsloth's UD-Q6_K_M, on the same path, its layers split over the fewest
  GPUs that hold it at the full 262,144 cells, K and V in q8_0 (~36 GB of VRAM: a 24 GB and a 16 GB GPU).

`run-<model>.bat` starts each from the clone's folder, for models in setup's data folder (`../Lamina-data`). Their
engine options: [SETUP.md](../docs/SETUP.md#engine-options); the server's keys: [USING.md](../docs/USING.md#the-config).

## The development PC's own options

The development PC's configs name their GPUs (`--main-gpu GPU-75af3083 --second-gpu "RTX 5070 Ti,GPU-f26e9ba6"`, its
two RTX 3090s by their UUIDs' starts, where the examples take every other GPU with `auto`), and its Flash-Next configs add four options measured for its hardware, which the examples
leave out. Each pays only where a PC has the same trait: measure it through the server before keeping it. Its
Qwen3.8-27B config puts the RTX 5070 Ti first (`--main-gpu`, which setup sets to a PC's own choice): that card reads
prompts ~1.44x as fast as the RTX 3090 and takes the first 26 layers, the 3090 the other 38 with the head, and the
prompt chunks run on both at once, so the prompts read ~11% faster than in the other order.

- **`--arena-place`** (both): for RAM whose regions read at different rates, as unpaired DIMMs make (here two 32 GB and
  two 48 GB modules leave 32 GB on one channel), the engine times its RAM arena at the start and keeps the experts the
  CPU reads out of the slow region, again between requests. Elsewhere it finds nothing to move and costs ~2 s at the
  start.
- **`--no-file-cache`** (UD-Q4_K_XL): the engine reads the experts from the GGUF past the OS file cache. Where the RAM
  holds the 72 GiB arena and the cached 72 GB file together (this PC's 160 GB), the file settles in the cache on pages
  the later starts' arenas would have had: here generation fell 7-8% over a session's starts, and with the option it
  stays at a fresh boot's speed. Each start then reads the experts from the drive (here 4.3-4.4 GiB/s against 6 from
  the cache, ~5 s more). With less RAM the cache cannot keep the file beside the arena, and the option costs only the
  start time.
- **`--prompt-split 26`** (both): a prompt of more than one chunk runs in two stages, the first tier taking each
  chunk's layers 26-47 whole while the main GPU runs the next chunk. It pays where the tier alone reads a chunk about as
  fast as both GPUs together: this PC's 5070 Ti runs the dense steps 1.44x as fast as its 3090, and prompts of
  32K-128K tokens read 1.13-1.55x as fast. It takes 0.7 GiB more pinned RAM and lends 6-8 GiB of the tier's expert
  cache while such a prompt runs. Its layer needs measuring too, through the server: here generate mode put IQ3_S's
  best layer at 23 (6-11% faster than 26), while through the API 26 was 2-4% faster than 23.
- **`--second-gpu-clocks 0,1395-2100/9751`** (all three): a value per tier, here none for the first and, for the
  second, its graphics clock locked at 1395-2100 MHz and its memory clock at 9,751 while the engine generates (NVML's
  locks, the ones `nvidia-smi -lgc` and `-lmc` set: the engine's process needs administrator rights, and says so
  when it has none). It pays for a tier that gets nothing but the verify windows' short shares: under them the
  Windows driver drops this PC's second RTX 3090 to its idle clocks between shares, its rows come late and its
  moves copy at a quarter of the link's rate, and generation takes 4-22% longer. The engine sets the locks from a
  request's start, or from a long prompt's last chunk, until 5 s after the request, so the card idles at ~12 W
  instead of the ~110 W it draws locked. The clocks are a card's own: its full memory clock, and a graphics range
  that starts well above its idle clock.
