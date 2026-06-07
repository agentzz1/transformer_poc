"""Evaluate our agent vs opponents over many seeds (2-player and 4-player)."""
import sys, warnings
warnings.filterwarnings("ignore")
from kaggle_environments import make

MINE = "kaggle_orbitwars/main.py"
SNIPER = "data/orbitwars/main.py"

def play(agents, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    return [s.reward for s in env.steps[-1]]

def duel(opp, n=12, tag=""):
    w = d = l = 0
    for seed in range(n):
        # our agent as player 0
        r = play([MINE, opp], seed)
        if r[0] is None: r[0] = -999
        if r[1] is None: r[1] = -999
        if r[0] > r[1]: w += 1
        elif r[0] == r[1]: d += 1
        else: l += 1
    print(f"  vs {tag or opp}: {n} games  W{w} D{d} L{l}  winrate {w/n:.2f}", flush=True)
    return w / n

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print("2-player duels:")
    duel("random", n, "random")
    duel(SNIPER, n, "sample-sniper")
    # 4-player: us vs 3 snipers
    w = 0
    for seed in range(n):
        r = play([MINE, SNIPER, SNIPER, SNIPER], seed)
        r = [x if x is not None else -999 for x in r]
        if r[0] == max(r) and r.count(max(r)) == 1: w += 1
    print(f"4p vs 3 snipers: {n} games  outright-wins {w}  ({w/n:.2f})")
