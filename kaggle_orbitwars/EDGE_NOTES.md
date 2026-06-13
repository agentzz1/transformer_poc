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

## CORRECTION: the ~1165 "ceiling" was wrong — it's incomplete convergence
Verified: Pilkwang's exp48 archive == slawekbiel dataset orbit_lite + exp48 main.py,
BYTE-IDENTICAL to what we submitted. Same agent (waves=7, term 8, horizon 18,
sizes 0.5/0.75/1.0). So we DID effectively try the "exp48 (exp41+7 waves, 1301.9)"
notebook — re-submitting it changes nothing.

Same-agent authors' CURRENT ranks (4204 teams):
  slawekbiel  1298.5 (rank 76)
  Pilkwang    1207.9 (rank 260)   <- submitted the byte-identical archive to ours
  ShumingFang 1184.6 (rank 389)   <- bronze zone
  US          1165.8 (rank 525)
=> The 42-pt gap to Pilkwang (identical agent) is convergence + variance, NOT a
   better version. Our submission is younger/fewer games and still oscillating
   (it already touched 1192). Byte-identical agents converge to similar ratings.
   Expect ours to drift toward ~1185-1207 = BRONZE ZONE with more games.
   Action: wait for convergence; no resubmit needed.

## 2026-06-13: TOP 1% PUSH (deadline 06-23, 4377 teams, Featured/$50k=medals)
Thresholds: TOP1%=rank43/1391.8 | silver=rank218/1231 | bronze=rank437/1174.
Reality: public exp48 clones DECAYED as field strengthened (us 1165->1100,
Pilkwang 1207->1143, ShumingFang 1184->1114). Fixed clones lose ground.

Strongest ADOPTABLE public base = slawekbiel "The Producer V2" = 1320.7 (top 1.8%).
 (title-claim notebooks are inflated: ramesh888 "1400+ ELO" is really 1161.)
V2 key idea: reinforcement-aware ROI — capture floor inflated by
 beta*rho(eta)*reachable_enemy_mass (opponents can reinforce mid-flight).
 Single-size fleets (dropped exp48 multi-size); NO terminal phase.
=> Phase 1: submitted V2 as floor. Phase 2 levers to exceed 1320->1391:
   (a) multi-size + reinforcement combo, (b) add terminal/endgame phase,
   (c) opponent-response modeling, (d) config (beta/waves/horizon).
   Validate via head-to-head vs V2 in BOTH 2P & 4P; field-test daily.

## 2026-06-13 Phase 2 results (main_lab = V2 + multi-size + terminal, parity-verified)
2P self-play vs stock V2 (60 games each, ship margin in parens):
  beta3 (2.2->3.0)            0.57 (+1123)
  multi-size (0.5,0.75,1.0)   0.57 (+968)
  ms_beta3                    0.58 (+790)
  ms75_beta3 (0.75,1.0;b3)    0.60 (+1074)   <- best, SUBMITTED for field test
  terminal phase              ~0.44 (no help) ; waves8 0.42 ; horizon20 0.55(+50)
Changes affect 2P config ONLY; 4P pinned to stock V2 (no regression). Field test
pending — local self-play is an imperfect predictor, so the LB is the judge.

## 2026-06-13 field verdict: local improvements DO NOT transfer
V2 converged to 1274.4 (still climbing) => est. true rank ~124/4380 = TOP 2.8% = SILVER.
But my local-tested tweaks LOST in the field:
  ms75_beta3 (0.60 vs V2 local) -> field 1170 (BELOW V2 1274)
  breadth_msb3 (0.64 local)     -> still climbing, expected below V2
=> Confirmed (again): self-play-vs-V2 winrate is NOT predictive of field ELO.
   Stock V2 is our best agent. Stop submitting V2 derivatives.
Loop fix: estimate rank from live best score vs CSV distribution (public CSV
lags and showed our ms75 score instead of V2, causing a false drop alert).
Bronze comfortably secured; sitting in SILVER zone on V2 alone.

## 2026-06-13 prize-opportunity scan (other competitions)
Active comps w/ prize (deadline > today):
  arc-prize-2026-arc-agi-3  $850k 2026-11-02 (entered) - abstract reasoning, NOT our strength
  arc-prize-2026-arc-agi-2  $700k 2026-11-02 (entered) - same
  arc-prize-2026-paper-track $450k 2026-11-09
  nvidia-nemotron-reasoning $106k 2026-06-15 (entered) - too late (LLM reasoning)
  neurogolf-2026            $50k  2026-07-15 (entered, 3x ERROR, no standing) - multi-task code/algo
  orbit-wars                $50k  2026-06-23 (entered) - SILVER medal range, NOT prize (top10~1529 vs our ~1333)
Verdict: our edge (adopt strongest public agent -> medal range) yields MEDALS not PRIZE
(prize needs top ~3-10 absolute, far above public-agent ceilings). No easy prize chance.
neurogolf-2026 is the only open question (would need real from-scratch work).
