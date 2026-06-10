# Orbit Wars — edge over the public Producer Hybrid v4

Baseline = Tamrazov's public "Producer Hybrid v4" (orbit_lite torch agent), which
many competitors submit unchanged. To rank above identical copies we need an edge.

## Controlled self-play A/B (producer-vs-producer, identical engine)
Harness: `producer_tune.py` — loads the agent once per worker, plays config
variants head-to-head over many seeds (2P), scored by final ship margin.

### Findings (vs stock 2P config, horizon=18)
| variant       | winrate | avg ship margin |
|---------------|---------|-----------------|
| horizon=19    | 0.52    | +121            |
| **horizon=20**| **0.70**| **+739**        |  <- sharp optimum (60 games)
| horizon=21    | 0.58    | +333            |
| horizon=22    | 0.62    | +401            |
| horizon=24    | 0.30    | -757            |  <- collapses
| horizon=26    | 0.23    | -992            |
| roi_threshold 1.3/1.7 | 0.43 | +0          |  <- does not bind in 2P
| max_waves=8   | 0.43    | +0              |  <- does not bind
| min_ships=3   | 0.46    | +842 (mixed)    |

### Decision
2P `ProducerLiteConfig.horizon`: 18 -> 20 (one-line change in main.py).
4P preset (CONFIG_4P, horizon=13) left untouched (not yet validated).
Per-turn timing at horizon 20-26 stays <40ms (Kaggle limit 1000ms) -> safe.

### Next levers to test
- 4P horizon sweep (separate 1-vs-3 self-play harness).
- min_ships_to_launch (big ship margin but mixed winrate — worth a field test).
