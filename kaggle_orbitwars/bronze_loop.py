"""Bronze-guard loop engine (driven by a persistent Monitor).

Polls our orbit-wars standing every ~10min and prints an ACTIONABLE line only
when something needs a decision: submission error, crossing the bronze 10%
boundary either way, or meaningful convergence movement (best >= +/-40 since
last emit). State persists in /tmp so it survives monitor re-arms. Each printed
line wakes the /loop turn to evaluate and act.
"""
import time, subprocess, glob, csv, os, re, json

os.environ["KAGGLE_API_TOKEN"] = "KGAT_89355067f08958fd6805b6053c24d9e0"
V2, MS, BR = 53632955, 53633979, 53634496
TRACKED = (V2, MS, BR)
STATE = "/tmp/bronze_state.json"


def sub_scores():
    out = subprocess.run(["python3", "-m", "kaggle", "competitions", "submissions", "orbit-wars"],
                         capture_output=True, text=True).stdout
    res, err = {}, False
    for line in out.splitlines():
        for ref in TRACKED:
            if str(ref) in line:
                m = re.search(r"COMPLETE\s+([0-9]+\.[0-9])", line)
                if m:
                    res[ref] = float(m.group(1))
                if "ERROR" in line.upper():
                    err = True
    return res, err


def rank_info():
    d = "/tmp/lb_loop"
    subprocess.run(f"rm -rf {d}", shell=True)
    subprocess.run(["python3", "-m", "kaggle", "competitions", "leaderboard", "orbit-wars",
                    "--download", "-p", d], capture_output=True)
    subprocess.run(f"cd {d} && unzip -o *.zip >/dev/null 2>&1", shell=True)
    fs = glob.glob(d + "/*publicleaderboard*.csv")
    if not fs:
        return None
    scores = sorted((float(r["Score"]) for r in csv.DictReader(open(fs[0])) if r["Score"]),
                    reverse=True)
    n = len(scores)
    cutoff = max(1, int(n * 0.10))
    silver = max(1, int(n * 0.05))
    cutscore = scores[cutoff - 1]
    return (n, cutoff, silver, cutscore, scores)


def rank_from_score(scores, best):
    """Estimate our live rank: teams strictly above our best live score + 1.
    Avoids the public-CSV lag where our row may still show an older/other sub."""
    return sum(1 for s in scores if s > best) + 1


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"last_emit_best": 0.0, "was_in": None}


def save_state(s):
    json.dump(s, open(STATE, "w"))


while True:
    try:
        scores, err = sub_scores()
        ri = rank_info()
    except Exception:
        time.sleep(600)
        continue
    if not ri:
        time.sleep(600)
        continue
    n, cutoff, silver, cutscore, all_scores = ri
    best = max(scores.values()) if scores else 0.0
    rank = rank_from_score(all_scores, best)        # live-score-based, lag-free
    in_bronze = rank <= cutoff
    in_silver = rank <= silver
    st = load_state()
    emit, tag = False, ""
    if err:
        emit, tag = True, "SUBMISSION_ERROR"
    elif st["was_in"] is not None and in_bronze != st["was_in"]:
        emit, tag = True, ("ENTERED_BRONZE" if in_bronze else "DROPPED_OUT_OF_BRONZE")
    elif abs(best - st["last_emit_best"]) >= 40:
        emit, tag = True, ("SILVER_ZONE" if in_silver else "CONVERGENCE")
    st["was_in"] = in_bronze
    if emit:
        st["last_emit_best"] = best
        zone = "SILVER" if in_silver else ("bronze" if in_bronze else "below-bronze")
        print(f"LOOP[{tag}] rank~{rank}/{n}({zone}) bronze_cut={cutoff}(~{cutscore:.0f}) "
              f"best={best:.1f} V2={scores.get(V2,'?')} ms75={scores.get(MS,'?')} "
              f"breadth={scores.get(BR,'?')}", flush=True)
    save_state(st)
    time.sleep(600)
