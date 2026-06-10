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

## 4P tuning — NEGATIVE result (kept stock)
4P self-play (`producer_tune4p.py`, variant in 1 seat vs 3 stock):
small-sample runs (24-48 games) hinted horizon=20 helped (share 0.30-0.33),
but at 64 games it REVERSED — all horizon changes hurt:
  h18 share 0.193, h20 0.224, h22 0.124  (vs 0.250 baseline).
=> the early 4P signal was noise. CONFIG_4P horizon=13 is already well tuned;
   leave the 4P preset untouched. Only the 2P horizon=20 change ships.

## Reality check on "top 3"
Public Producer Hybrid v4 converges ~1150-1200. Top 3 needs ~1660+.
Tamrazov (the author, a strong competitor) is himself not top 3 with this agent,
so config tweaks alone will NOT close the gap — that requires a stronger
planner/scorer (code-level R&D), higher risk, uncertain payoff in the time left.
The horizon=20 edge is a real but modest improvement over identical public copies.

## Leaderboard reality + medal path (2026-06-10)
4184 teams. Medal cutoffs (1000+ teams band):
  Bronze = top 10% (rank <=418, ~score 1181)
  Silver = top 5%  (rank <=209, ~score 1330)
  Gold   = top 10 + 0.2% (~18 teams, ~1575+)
Our team "⸻AgentZZ ⸻": rank 656 / score 1140.4 (top 15.7%) — short of bronze.
h20 tuned variant rated 1068 (WORSE than stock 1140) — confirms field-tuning
via self-play-vs-stock does not transfer. Dropped.

### Strongest PUBLIC notebooks (author's own LB rank)
  Pilkwang  "ProducerLite Flow-Diff" 1220.5 (rank 194, SILVER zone)  <- adopted
  Marwan A. "Agent Smith +1000LB"    1216.6 (rank 207)
  Penguin   "copied from vkhydras"   1205.3 (rank 263)
  Carbon    "fork of top1"           1201.9 (rank 291)
  ShumingFang exp48                  1167.8 (rank 494)
  Tamrazov  Producer Hybrid v4       1164.1 (rank 526)  <- what we had

=> The producer agent we first adopted is NOT the strongest public code.
   Pilkwang's exp48-lineage flow-diff agent (1220.5) is well above the bronze
   cutoff and in the silver zone. Submitted exp48_public variant (the proven
   default; sha256-verified decode of the notebook's embedded archive).
   Challenger variant exp48_2p_regroup_4p_original kept as backup (unproven on LB).
