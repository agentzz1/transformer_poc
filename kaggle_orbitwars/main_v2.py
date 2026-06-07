"""Orbit Wars agent v2 — v1 (interception, sun-avoid, defense) + global multi-target
allocation, step-adaptive reserve (aggressive early & late), and frontier reinforcement.
"""
import math

CENTER = (50.0, 50.0); SUN_R = 10.0; ROT_LIMIT = 50.0; MAX_SPEED = 6.0
_prev = {}

def _speed(ships):
    ships = max(1, int(ships))
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5

def _orbiting(x, y, r): return math.hypot(x-CENTER[0], y-CENTER[1]) + r < ROT_LIMIT

def _rotate(x, y, ang):
    dx, dy = x-CENTER[0], y-CENTER[1]; c, s = math.cos(ang), math.sin(ang)
    return CENTER[0]+dx*c-dy*s, CENTER[1]+dx*s+dy*c

def _seg_hits_sun(x0, y0, x1, y1):
    dx, dy = x1-x0, y1-y0; L2 = dx*dx+dy*dy
    if L2 < 1e-9: return math.hypot(x0-CENTER[0], y0-CENTER[1]) < SUN_R
    t = max(0.0, min(1.0, ((CENTER[0]-x0)*dx+(CENTER[1]-y0)*dy)/L2))
    px, py = x0+t*dx, y0+t*dy
    return math.hypot(px-CENTER[0], py-CENTER[1]) < SUN_R

def agent(obs):
    global _prev
    if isinstance(obs, dict):
        planets = obs.get("planets", []); fleets = obs.get("fleets", [])
        player = obs.get("player", 0); av = obs.get("angular_velocity", 0.0)
        step = obs.get("step", 0)
    else:
        planets = obs.planets; fleets = obs.fleets; player = obs.player
        av = getattr(obs, "angular_velocity", 0.0); step = getattr(obs, "step", 0)

    P = {int(p[0]): {"id": int(p[0]), "owner": int(p[1]), "x": float(p[2]), "y": float(p[3]),
                     "r": float(p[4]), "ships": float(p[5]), "prod": float(p[6])} for p in planets}
    cur = {}
    for pid, p in P.items():
        cur[pid] = (p["x"], p["y"]); sav = 0.0
        if _orbiting(p["x"], p["y"], p["r"]):
            sav = av
            if pid in _prev:
                ox, oy = _prev[pid]
                a0 = math.atan2(oy-CENTER[1], ox-CENTER[0]); a1 = math.atan2(p["y"]-CENTER[1], p["x"]-CENTER[0])
                d = (a1-a0+math.pi) % (2*math.pi) - math.pi
                if abs(d) > 1e-4: sav = max(-0.08, min(0.08, d))
        p["sav"] = sav
    _prev = cur

    def predict(p, dt):
        if abs(p["sav"]) < 1e-9: return p["x"], p["y"]
        return _rotate(p["x"], p["y"], p["sav"]*dt)

    incoming = {pid: {} for pid in P}
    for f in fleets:
        fo, fx, fy, fang, fsh = int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[6])
        hx, hy = math.cos(fang), math.sin(fang); best, bestd = None, 1e9
        for pid, p in P.items():
            rx, ry = p["x"]-fx, p["y"]-fy; proj = rx*hx+ry*hy
            if proj <= 0: continue
            perp = abs(rx*hy-ry*hx)
            if perp < p["r"]+1.5 and proj < bestd: bestd, best = proj, pid
        if best is not None: incoming[best][fo] = incoming[best].get(fo, 0.0)+fsh
    def en_in(pid): return sum(s for o, s in incoming[pid].items() if o != player)
    def fr_in(pid): return incoming[pid].get(player, 0.0)

    mine = [p for p in P.values() if p["owner"] == player]
    if not mine: return []

    # reserve policy by phase
    def reserve_for(src, threat):
        if step >= 470: base = 0.0            # endgame: throw everything
        elif step < 80: base = 0.05           # early: expand hard
        else: base = 0.12
        return max(threat, base*src["ships"], 1.0 if step < 470 else 0.0)

    def intercept(src, tgt):
        ships_guess = max(1.0, tgt["ships"]+1); ax, ay = tgt["x"], tgt["y"]; eta = 1.0
        for _ in range(4):
            dist = math.hypot(ax-src["x"], ay-src["y"]); eta = dist/_speed(ships_guess)
            ax, ay = predict(tgt, eta)
            grow = tgt["prod"]*eta if tgt["owner"] >= 0 else 0.0
            ships_guess = max(1.0, tgt["ships"]+grow+3)
        return ax, ay, eta

    avail = {};
    for s in mine:
        th = en_in(s["id"]);
        if th >= s["ships"]-1: avail[s["id"]] = 0.0  # threatened: hold
        else: avail[s["id"]] = s["ships"] - reserve_for(s, th)

    # build candidate launches (src -> tgt) with value
    cands = []
    for s in mine:
        if avail[s["id"]] < 1: continue
        for t in P.values():
            if t["owner"] == player: continue
            ax, ay, eta = intercept(s, t)
            if _seg_hits_sun(s["x"], s["y"], ax, ay): continue
            grow = t["prod"]*eta if t["owner"] >= 0 and t["owner"] != player else 0.0
            req = t["ships"]+grow+en_in(t["id"])-fr_in(t["id"])+1+2
            if req <= 0 or req > avail[s["id"]]: continue
            val = (t["prod"]+0.2)/(req*(1.0+0.05*eta))
            if t["owner"] == -1: val *= 1.3
            ang = math.atan2(ay-s["y"], ax-s["x"])
            cands.append((val, s["id"], t["id"], int(math.ceil(req)), ang))
    cands.sort(reverse=True)

    moves = []; claimed = set(); src_left = dict(avail)
    for val, sid, tid, send, ang in cands:
        if tid in claimed: continue
        if src_left[sid] < send: continue
        moves.append([sid, ang, send]); claimed.add(tid); src_left[sid] -= send

    # frontier reinforcement: idle surplus -> nearest own planet with enemy incoming
    threatened = [p for p in mine if en_in(p["id"]) > 0]
    if threatened:
        for s in mine:
            if src_left.get(s["id"], 0) > 20 and en_in(s["id"]) == 0:
                tgt = min(threatened, key=lambda q: math.hypot(q["x"]-s["x"], q["y"]-s["y"]))
                if tgt["id"] == s["id"]: continue
                ax, ay, eta = intercept(s, tgt)
                if _seg_hits_sun(s["x"], s["y"], ax, ay): continue
                send = int(src_left[s["id"]]*0.6)
                if send >= 1:
                    moves.append([s["id"], math.atan2(ay-s["y"], ax-s["x"]), send]); src_left[s["id"]] -= send
    return moves
