"""Producer config A/B tournament — controlled self-play on the identical engine.

Loads the Producer Hybrid v4 module ONCE per worker, then plays config variants
head-to-head by calling run_turn directly with each side's own config + memory
(no global monkeypatch, no per-game reload). Because both sides run the SAME
planner/scorer, win-rate differences isolate the config knobs.

Run:  python3 kaggle_orbitwars/producer_tune.py <seeds> [variant1 variant2 ...]
"""
import sys, os, warnings, dataclasses
from multiprocessing import Pool
warnings.filterwarnings("ignore")

PROD_DIR = os.path.join(os.path.dirname(__file__), "producer")

VARIANTS = {
    "stock":     {},
    "horizon22": {"horizon": 22},
    "horizon14": {"horizon": 14},
    "roi1.3":    {"roi_threshold": 1.3},
    "roi1.7":    {"roi_threshold": 1.7},
    "waves8":    {"max_waves_per_turn": 8},
    "minship3":  {"min_ships_to_launch": 3.0},
    "src16":     {"max_sources_per_lane": 16, "max_offensive_targets": 16},
    "aggr":      {"roi_threshold": 1.3, "max_waves_per_turn": 8, "horizon": 20},
    "horizon20": {"horizon": 20},
    "horizon24": {"horizon": 24},
    "horizon26": {"horizon": 26},
    "h24_ms3":   {"horizon": 24, "min_ships_to_launch": 3.0},
    "h24_src16": {"horizon": 24, "max_sources_per_lane": 16, "max_offensive_targets": 16},
}

_M = None  # per-worker module handle


def _init_worker():
    global _M
    import torch
    torch.set_num_threads(1)  # avoid oversubscription across Pool workers
    sys.path.insert(0, PROD_DIR)
    import importlib.util
    spec = importlib.util.spec_from_file_location("prod_main", os.path.join(PROD_DIR, "main.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["prod_main"] = m
    spec.loader.exec_module(m)
    _M = m


def _make_agent(overrides):
    """Closure: own memory + own config, calls run_turn directly (no globals)."""
    import torch
    m = _M
    base = m.ProducerLiteConfig()
    cfg2p = dataclasses.replace(base, **overrides)
    mem = m.ProducerLiteMemory()
    state = {"pc": None}

    def agent(obs):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        pid = int(player)
        ot = m.single_obs_to_tensor(obs, player_id=pid)
        if bool((ot["step"] == 0).all()):
            state["pc"] = None; mem.reset()
        if state["pc"] is None:
            state["pc"] = m.largest_initial_player_count(ot)
        cfg = cfg2p if int(state["pc"]) < 4 else m._config_for(state["pc"])
        with torch.no_grad():
            row = m.run_turn(ot, config=cfg, player_count=int(state["pc"]), memory=mem)
        return m.sparse_action_row_to_moves(row, obs, player_id=pid)
    return agent


def _final_ships(env):
    obs = env.steps[-1][0].observation
    sh = {}
    for p in obs["planets"]:
        if p[1] >= 0:
            sh[p[1]] = sh.get(p[1], 0.0) + p[5]
    for f in obs["fleets"]:
        sh[f[1]] = sh.get(f[1], 0.0) + f[6]
    return sh


def _play(args):
    a_over, b_over, seed = args
    from kaggle_environments import make
    A = _make_agent(a_over); B = _make_agent(b_over)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([A, B])
    sh = _final_ships(env)
    return (sh.get(0, 0.0), sh.get(1, 0.0))  # final ship totals P0, P1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    which = sys.argv[2:] if len(sys.argv) > 2 else [v for v in VARIANTS if v != "stock"]
    base = "stock"
    print(f"A/B vs '{base}', {n} seeds x2 sides each:", flush=True)
    for v in which:
        jobs = []
        for seed in range(n):
            jobs.append((VARIANTS[v], VARIANTS[base], seed))   # v as P0
            jobs.append((VARIANTS[base], VARIANTS[v], seed))   # v as P1
        with Pool(4, initializer=_init_worker) as pool:
            res = pool.map(_play, jobs)
        w = d = g = 0; margin = 0.0
        for i, (s0, s1) in enumerate(res):
            vs, bs = (s0, s1) if i % 2 == 0 else (s1, s0)
            margin += vs - bs
            if vs > bs * 1.02: w += 1
            elif bs > vs * 1.02: pass
            else: d += 1
            g += 1
        print(f"  {v:12s} vs stock: W{w} D{d} L{g-w-d} / {g} = {w/g:.2f}  "
              f"avg ship margin {margin/g:+.0f}", flush=True)
