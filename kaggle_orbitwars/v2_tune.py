"""Producer V2 improvement A/B — variant vs stock V2 on the identical engine.

Loads main_lab.py once per worker (verified byte-identical to V2 at default
config) and pits config/structural variants head-to-head against stock V2 in
2P (variant vs V2) and 4P (variant in 1 seat vs 3 V2). Scored by win-rate and
final ship margin/share. Large, consistent margins in BOTH modes are the
filter before spending a Kaggle submission.

Run:  python3 kaggle_orbitwars/v2_tune.py <mode 2p|4p> <seeds> <variant ...>
"""
import sys, os, warnings, dataclasses
from multiprocessing import Pool
warnings.filterwarnings("ignore")

PROD_DIR = os.path.join(os.path.dirname(__file__), "producer_v2")

# Each variant = dict of overrides applied to the 2P ProducerLiteConfig (and,
# if keys prefixed "4p_", to CONFIG_4P). Empty = stock V2.
VARIANTS = {
    "stock":        {},
    # multi-size tiers
    "ms_50_75_100": {"size_multipliers": (0.5, 0.75, 1.0)},
    "ms_60_100":    {"size_multipliers": (0.6, 1.0)},
    "ms_75_100":    {"size_multipliers": (0.75, 1.0)},
    # terminal/endgame phase
    "term40":       {"terminal_phase_turns": 40},
    "term60":       {"terminal_phase_turns": 60},
    "term40_w10":   {"terminal_phase_turns": 40, "terminal_max_waves_per_turn": 10},
    # combos
    "ms_term40":    {"size_multipliers": (0.5, 0.75, 1.0), "terminal_phase_turns": 40},
    # config tweaks
    "beta3":        {"reinforce_size_beta": 3.0},
    "beta1.5":      {"reinforce_size_beta": 1.5},
    "waves8":       {"max_waves_per_turn": 8},
    "horizon20":    {"horizon": 20},
    "roi1.3":       {"roi_threshold": 1.3},
    # combos / beta fine-search
    "beta2.6":      {"reinforce_size_beta": 2.6},
    "beta3.5":      {"reinforce_size_beta": 3.5},
    "ms_beta3":     {"size_multipliers": (0.5, 0.75, 1.0), "reinforce_size_beta": 3.0},
    "ms75_beta3":   {"size_multipliers": (0.75, 1.0), "reinforce_size_beta": 3.0},
    "ms_beta3_h20": {"size_multipliers": (0.5, 0.75, 1.0), "reinforce_size_beta": 3.0, "horizon": 20},
}

_M = None


def _init_worker():
    global _M
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, PROD_DIR)
    import importlib.util
    spec = importlib.util.spec_from_file_location("lab", os.path.join(PROD_DIR, "main_lab.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["lab"] = m
    spec.loader.exec_module(m)
    _M = m


def _make_agent(overrides):
    import torch
    m = _M
    ov2 = {k: v for k, v in overrides.items() if not k.startswith("4p_")}
    ov4 = {k[3:]: v for k, v in overrides.items() if k.startswith("4p_")}
    cfg2 = dataclasses.replace(m.ProducerLiteConfig(), **ov2)
    cfg4 = dataclasses.replace(m.CONFIG_4P, **ov4) if ov4 else m.CONFIG_4P
    mem = m.ProducerLiteMemory(); st = {"pc": None}

    def agent(obs):
        pid = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
        ot = m.single_obs_to_tensor(obs, player_id=pid)
        if bool((ot["step"] == 0).all()):
            st["pc"] = None; mem.reset()
        if st["pc"] is None:
            st["pc"] = m.largest_initial_player_count(ot)
        base = cfg4 if int(st["pc"]) >= 4 else cfg2
        step = int(ot["step"].reshape(-1)[0].item())
        cfg = m._apply_phase_config(base, step)
        with torch.no_grad():
            row = m.run_turn(ot, config=cfg, player_count=int(st["pc"]), memory=mem)
        return m.sparse_action_row_to_moves(row, obs, player_id=pid)
    return agent


def _ships(env, seats):
    obs = env.steps[-1][0].observation
    sh = [0.0] * seats
    for p in obs["planets"]:
        if p[1] >= 0 and p[1] < seats:
            sh[p[1]] += p[5]
    for f in obs["fleets"]:
        if f[1] < seats:
            sh[f[1]] += f[6]
    return sh


def _play2p(args):
    ov, seed, var_first = args
    from kaggle_environments import make
    A = _make_agent(ov); B = _make_agent({})
    agents = [A, B] if var_first else [B, A]
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    sh = _ships(env, 2)
    vs, bs = (sh[0], sh[1]) if var_first else (sh[1], sh[0])
    return (vs, bs)


def _play4p(args):
    ov, seat, seed = args
    from kaggle_environments import make
    agents = [_make_agent(ov) if i == seat else _make_agent({}) for i in range(4)]
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    sh = _ships(env, 4)
    tot = sum(sh) or 1.0
    win = 1 if (sh[seat] == max(sh) and sh.count(max(sh)) == 1) else 0
    return (win, sh[seat] / tot)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "2p"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    which = sys.argv[3:] if len(sys.argv) > 3 else [v for v in VARIANTS if v != "stock"]
    if mode == "2p":
        print(f"2P: variant vs stock V2, {n} seeds x2 sides:", flush=True)
        for v in which:
            jobs = [(VARIANTS[v], s, True) for s in range(n)] + [(VARIANTS[v], s, False) for s in range(n)]
            with Pool(4, initializer=_init_worker) as pool:
                res = pool.map(_play2p, jobs)
            g = len(res); w = sum(1 for vs, bs in res if vs > bs * 1.02)
            d = sum(1 for vs, bs in res if abs(vs - bs) <= bs * 0.02)
            margin = sum(vs - bs for vs, bs in res) / g
            print(f"  {v:14s} W{w} D{d} L{g-w-d}/{g} = {w/g:.2f}  ship margin {margin:+.0f}", flush=True)
    else:
        print(f"4P: variant in 1 seat vs 3 V2, {n} seeds x4 seats (baseline win 0.25):", flush=True)
        for v in which:
            jobs = [(VARIANTS[v], seat, s) for seat in range(4) for s in range(n)]
            with Pool(4, initializer=_init_worker) as pool:
                res = pool.map(_play4p, jobs)
            g = len(res); wins = sum(r[0] for r in res); share = sum(r[1] for r in res) / g
            print(f"  {v:14s} win {wins}/{g} = {wins/g:.2f}  ship share {share:.3f} (vs 0.250)", flush=True)
