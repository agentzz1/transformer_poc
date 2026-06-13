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
    rows = [r for r in csv.DictReader(open(fs[0])) if r["Score"]]
    rows.sort(key=lambda r: float(r["Score"]), reverse=True)
    n = len(rows)
    cutoff = max(1, int(n * 0.10))
    cutscore = float(rows[cutoff - 1]["Score"])
    for i, r in enumerate(rows, 1):
        if "agentzz" in (r.get("TeamMemberUserNames") or "").lower():
            return (i, float(r["Score"]), n, cutoff, cutscore)
    return (None, None, n, cutoff, cutscore)


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
    rank, myscore, n, cutoff, cutscore = ri
    best = max(scores.values()) if scores else 0.0
    in_bronze = (rank is not None and rank <= cutoff)
    st = load_state()
    emit, tag = False, ""
    if err:
        emit, tag = True, "SUBMISSION_ERROR"
    elif st["was_in"] is not None and in_bronze != st["was_in"]:
        emit, tag = True, ("ENTERED_BRONZE" if in_bronze else "DROPPED_OUT_OF_BRONZE")
    elif abs(best - st["last_emit_best"]) >= 40:
        emit, tag = True, "CONVERGENCE"
    st["was_in"] = in_bronze
    if emit:
        st["last_emit_best"] = best
        rk = f"{rank}/{n}" if rank else f"?/{n}"
        print(f"LOOP[{tag}] rank={rk} bronze_cut={cutoff}(~{cutscore:.0f}) "
              f"best={best:.1f} V2={scores.get(V2,'?')} ms75={scores.get(MS,'?')} "
              f"breadth={scores.get(BR,'?')}", flush=True)
    save_state(st)
    time.sleep(600)
