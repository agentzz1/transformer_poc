"""Parallel round-robin tournament between agent files (both sides, many seeds)."""
import sys, warnings, itertools
from multiprocessing import Pool
warnings.filterwarnings("ignore")

def play(args):
    a, b, seed = args
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([a, b])
    r = [s.reward if s.reward is not None else -999 for s in env.steps[-1]]
    return (a, b, seed, r[0], r[1])

if __name__ == "__main__":
    agents = sys.argv[1].split(",")
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    jobs = []
    for a, b in itertools.combinations(agents, 2):
        for seed in range(nseeds):
            jobs.append((a, b, seed))
            jobs.append((b, a, seed))
    with Pool(4) as pool:
        results = pool.map(play, jobs)
    import json
    with open("/tmp/tourney_results.json", "w") as fh:
        json.dump(results, fh)
    score = {a: [0, 0] for a in agents}  # wins, games
    pair = {}
    for a, b, seed, ra, rb in results:
        score[a][1] += 1; score[b][1] += 1
        k = tuple(sorted((a, b)))
        pair.setdefault(k, {a: 0, b: 0})
        if ra > rb:
            score[a][0] += 1; pair[k][a] += 1
        elif rb > ra:
            score[b][0] += 1; pair[k][b] += 1
    print("=== overall winrate ===")
    for a in sorted(agents, key=lambda x: -score[x][0] / max(1, score[x][1])):
        w, g = score[a]
        name = a.split('/')[-1]
        print(f"  {name:<16} {w}/{g} = {w/g:.2f}")
    print("=== pairwise ===")
    for k, v in pair.items():
        n1, n2 = k[0].split('/')[-1], k[1].split('/')[-1]
        print(f"  {n1} {v[k[0]]} : {v[k[1]]} {n2}")
