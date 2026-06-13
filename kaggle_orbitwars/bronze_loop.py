"""Bronze-guard loop engine (driven by a persistent Monitor).

Polls our orbit-wars standing every ~10min and prints an ACTIONABLE line only
when something needs a decision: submission error, crossing the bronze 10%
boundary either way, or meaningful convergence movement (best >= +/-40 since
last emit). State persists in /tmp so it survives monitor re-arms. Each printed
line wakes the /loop turn to evaluate and act.
"""
import time, subprocess, glob, csv, os, re, json, datetime

# Token from environment only — never hardcode secrets in committed code.
assert os.environ.get("KAGGLE_API_TOKEN"), "set KAGGLE_API_TOKEN in the environment"
V2, MS, BR = 53632955, 53633979, 53634496
TRACKED = (V2, MS, BR)
STATE = "/tmp/bronze_state.json"


def sub_scores():
    """All of our submissions' current COMPLETE public scores, keyed by ref id.
    Generic: captures every submission so `best` reflects our true leaderboard
    value regardless of how many variants are in flight."""
    out = subprocess.run(["python3", "-m", "kaggle", "competitions", "submissions", "orbit-wars"],
                         capture_output=True, text=True).stdout
    today = datetime.date.today().isoformat()  # e.g. 2026-06-13
    res, err = {}, False
    for line in out.splitlines():
        ref_m = re.match(r"\s*(\d{6,})", line)
        sc_m = re.search(r"COMPLETE\s+([0-9]+\.[0-9])", line)
        if ref_m and sc_m:
            res[int(ref_m.group(1))] = float(sc_m.group(1))
        # only flag a genuine error on a submission made TODAY (ignore old/desc text)
        if ref_m and "SubmissionStatus.ERROR" in line and today in line:
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
    top1 = max(1, int(n * 0.01))
    cutscore = scores[cutoff - 1]
    return (n, cutoff, silver, top1, cutscore, scores)


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
    n, cutoff, silver, top1, cutscore, all_scores = ri
    best = max(scores.values()) if scores else 0.0
    rank = rank_from_score(all_scores, best)        # live-score-based, lag-free
    in_bronze = rank <= cutoff
    in_silver = rank <= silver
    in_top1 = rank <= top1
    st = load_state()
    emit, tag = False, ""
    if err:
        emit, tag = True, "SUBMISSION_ERROR"
    elif in_top1 and not st.get("was_top1"):
        emit, tag = True, "TOP_1_PERCENT"
    elif st["was_in"] is not None and in_bronze != st["was_in"]:
        emit, tag = True, ("ENTERED_BRONZE" if in_bronze else "DROPPED_OUT_OF_BRONZE")
    elif abs(best - st["last_emit_best"]) >= 40:
        emit, tag = True, ("SILVER_ZONE" if in_silver else "CONVERGENCE")
    st["was_in"] = in_bronze
    st["was_top1"] = in_top1
    if emit:
        st["last_emit_best"] = best
        zone = "TOP1%" if in_top1 else ("SILVER" if in_silver else ("bronze" if in_bronze else "below-bronze"))
        top3 = sorted(scores.values(), reverse=True)[:3]
        print(f"LOOP[{tag}] rank~{rank}/{n}({zone}) bronze_cut={cutoff}(~{cutscore:.0f}) "
              f"top1%~{top1}(~{all_scores[top1-1]:.0f}) best={best:.1f} "
              f"our_top3={[round(x,1) for x in top3]}", flush=True)
    save_state(st)
    time.sleep(600)
