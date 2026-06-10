"""Producer 4P config A/B — one variant vs three stock agents (FFA).

The variant takes each of the 4 seats in turn (to cancel positional bias).
Scored by: outright-win rate (variant has strictly the most ships) and the
variant's average ship share. Tunes the CONFIG_4P preset.

Run:  python3 kaggle_orbitwars/producer_tune4p.py <seeds> [variant ...]
"""
import sys, os, warnings, dataclasses
from multiprocessing import Pool
warnings.filterwarnings("ignore")

PROD_DIR = os.path.join(os.path.dirname(__file__), "producer")

# 4P variants: overrides applied on top of the stock CONFIG_4P preset
VARIANTS = {
    "stock4p":   {},
    "h15":       {"horizon": 15},
    "h18":       {"horizon": 18},
    "h16":       {"horizon": 16},
    "h20":       {"horizon": 20},
    "h22":       {"horizon": 22},
    "h24":       {"horizon": 24},
    "src10":     {"max_sources_per_lane": 10, "max_offensive_targets": 10},
    "roi1.4":    {"roi_threshold": 1.4},
    "h18_src10": {"horizon": 18, "max_sources_per_lane": 10, "max_offensive_targets": 10},
}

_M = None


def _init_worker():
    global _M
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, PROD_DIR)
    import importlib.util
    spec = importlib.util.spec_from_file_location("prod_main", os.path.join(PROD_DIR, "main.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["prod_main"] = m
    spec.loader.exec_module(m)
    _M = m


def _make_agent(overrides, four_p=True):
    import torch
    m = _M
    base4 = m.CONFIG_4P if four_p else m.ProducerLiteConfig()
    cfg4 = dataclasses.replace(base4, **overrides)
    cfg2 = m.ProducerLiteConfig()
    mem = m.ProducerLiteMemory(); st = {"pc": None}

    def agent(obs):
        pid = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
        ot = m.single_obs_to_tensor(obs, player_id=pid)
        if bool((ot["step"] == 0).all()):
            st["pc"] = None; mem.reset()
        if st["pc"] is None:
            st["pc"] = m.largest_initial_player_count(ot)
        cfg = cfg4 if int(st["pc"]) >= 4 else cfg2
        with torch.no_grad():
            row = m.run_turn(ot, config=cfg, player_count=int(st["pc"]), memory=mem)
        return m.sparse_action_row_to_moves(row, obs, player_id=pid)
    return agent


def _play(args):
    var_over, seat, seed = args
    from kaggle_environments import make
    agents = []
    for i in range(4):
        agents.append(_make_agent(var_over) if i == seat else _make_agent({}))
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    obs = env.steps[-1][0].observation
    sh = [0.0, 0.0, 0.0, 0.0]
    for p in obs["planets"]:
        if p[1] >= 0:
            sh[p[1]] += p[5]
    for f in obs["fleets"]:
        sh[f[1]] += f[6]
    total = sum(sh) or 1.0
    win = 1 if (sh[seat] == max(sh) and sh.count(max(sh)) == 1) else 0
    return (win, sh[seat] / total)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    which = sys.argv[2:] if len(sys.argv) > 2 else [v for v in VARIANTS if v != "stock4p"]
    # baseline: stock seat among 3 stock = 0.25 win, 0.25 share by symmetry
    print(f"4P: variant in 1 seat vs 3 stock, {n} seeds x 4 seats (baseline win=0.25):", flush=True)
    for v in which:
        jobs = [(VARIANTS[v], seat, seed) for seat in range(4) for seed in range(n)]
        with Pool(4, initializer=_init_worker) as pool:
            res = pool.map(_play, jobs)
        g = len(res)
        wins = sum(r[0] for r in res)
        share = sum(r[1] for r in res) / g
        print(f"  {v:10s} win-rate {wins}/{g} = {wins/g:.2f}   avg ship share {share:.3f} (vs 0.250)", flush=True)
