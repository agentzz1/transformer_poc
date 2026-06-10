"""Producer config A/B tournament — controlled self-play on the identical engine.

Loads the Producer Hybrid v4 agent once, instantiates it under several
ProducerLiteConfig variants, and plays them head-to-head across many seeds in
2P and 4P. Because both sides run the SAME planner/scorer, win-rate differences
isolate the effect of the config knobs (unlike cross-architecture self-play,
which we found non-predictive of the Kaggle field).

NOTE: executing the third-party orbit_lite agent locally requires permission.
Run as:  python3 kaggle_orbitwars/producer_tune.py 24
"""
import sys, os, warnings, itertools, dataclasses
from multiprocessing import Pool
warnings.filterwarnings("ignore")

PROD_DIR = os.path.join(os.path.dirname(__file__), "producer")
sys.path.insert(0, PROD_DIR)

# candidate config overrides to A/B against the stock 2P default
VARIANTS = {
    "stock":        {},
    "horizon22":    {"horizon": 22},
    "roi1.3":       {"roi_threshold": 1.3},
    "roi1.7":       {"roi_threshold": 1.7},
    "waves8":       {"max_waves_per_turn": 8},
    "minship3":     {"min_ships_to_launch": 3.0},
    "aggr":         {"roi_threshold": 1.3, "max_waves_per_turn": 8, "horizon": 20},
}


def _load_agent_factory():
    import importlib.util, torch
    spec = importlib.util.spec_from_file_location("prod_main", os.path.join(PROD_DIR, "main.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["prod_main"] = m
    spec.loader.exec_module(m)
    return m


def make_agent(overrides):
    """Return a fresh agent callable bound to a config with the given overrides."""
    m = _load_agent_factory()
    import torch
    base = m.ProducerLiteConfig()
    cfg2p = dataclasses.replace(base, **overrides)
    runtime = m.ProducerLiteRuntime()
    # monkeypatch _config_for to use our 2P config (keep 4P default)
    orig = m._config_for
    def patched(pc):
        return cfg2p if int(pc) < 4 else orig(pc)
    def agent(obs):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        ot = m.single_obs_to_tensor(obs, player_id=int(player))
        m._config_for = patched
        with torch.no_grad():
            row = runtime.tensor_action(ot)
        return m.sparse_action_row_to_moves(row, obs, player_id=int(player))
    return agent


def play(args):
    a_over, b_over, seed = args
    from kaggle_environments import make
    A = make_agent(a_over); B = make_agent(b_over)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([A, B])
    r = [s.reward if s.reward is not None else -999 for s in env.steps[-1]]
    return (seed, r[0], r[1])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    names = list(VARIANTS)
    base = "stock"
    print(f"A/B vs '{base}', {n} seeds each side:")
    for v in names:
        if v == base:
            continue
        jobs = []
        for seed in range(n):
            jobs.append((VARIANTS[v], VARIANTS[base], seed))
            jobs.append((VARIANTS[base], VARIANTS[v], seed))
        with Pool(4) as pool:
            res = pool.map(play, jobs)
        w = 0; g = 0
        for i, (seed, ra, rb) in enumerate(res):
            # even index: v is side 0; odd: v is side 1
            v_first = (i % 2 == 0)
            vr, br = (ra, rb) if v_first else (rb, ra)
            if vr > br: w += 1
            g += 1
        print(f"  {v:12s} winrate vs stock: {w}/{g} = {w/g:.2f}")
