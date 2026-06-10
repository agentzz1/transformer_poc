"""Orbit Wars v4 — v1 core (decisive packages) + same-turn multi-source coordination,
multi-capture per turn, rear->front funneling with big fleets, amortization check.
"""
import math

CENTER = (50.0, 50.0)
SUN_R = 10.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0
EPISODE_STEPS = 500

_prev = {}

def _speed(ships):
    ships = max(1, int(ships))
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5

def _orbiting(x, y, r):
    return math.hypot(x - CENTER[0], y - CENTER[1]) + r < ROT_LIMIT

def _rotate(x, y, ang):
    dx, dy = x - CENTER[0], y - CENTER[1]
    c, s = math.cos(ang), math.sin(ang)
    return CENTER[0] + dx * c - dy * s, CENTER[1] + dx * s + dy * c

def _seg_hits_sun(x0, y0, x1, y1, margin=0.8):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(x0 - CENTER[0], y0 - CENTER[1]) < SUN_R + margin
    t = ((CENTER[0] - x0) * dx + (CENTER[1] - y0) * dy) / L2
    t = max(0.0, min(1.0, t))
    px, py = x0 + t * dx, y0 + t * dy
    return math.hypot(px - CENTER[0], py - CENTER[1]) < SUN_R + margin

def agent(obs):
    global _prev
    if isinstance(obs, dict):
        planets = obs.get("planets", []); fleets = obs.get("fleets", [])
        player = obs.get("player", 0); av = obs.get("angular_velocity", 0.0)
        step = obs.get("step", 0); comet_ids = set(obs.get("comet_planet_ids", []))
    else:
        planets = obs.planets; fleets = obs.fleets; player = obs.player
        av = getattr(obs, "angular_velocity", 0.0)
        step = getattr(obs, "step", 0); comet_ids = set(getattr(obs, "comet_planet_ids", []))

    P = {int(p[0]): {"id": int(p[0]), "owner": int(p[1]), "x": float(p[2]), "y": float(p[3]),
                     "r": float(p[4]), "ships": float(p[5]), "prod": float(p[6])} for p in planets}

    cur = {}
    for pid, p in P.items():
        cur[pid] = (p["x"], p["y"])
        sav = 0.0
        if _orbiting(p["x"], p["y"], p["r"]):
            sav = av
            if pid in _prev:
                ox, oy = _prev[pid]
                a0 = math.atan2(oy - CENTER[1], ox - CENTER[0])
                a1 = math.atan2(p["y"] - CENTER[1], p["x"] - CENTER[0])
                d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                if abs(d) > 1e-4:
                    sav = max(-0.08, min(0.08, d))
        p["sav"] = sav
    _prev = cur

    def predict(p, dt):
        if abs(p["sav"]) < 1e-9:
            return p["x"], p["y"]
        return _rotate(p["x"], p["y"], p["sav"] * dt)

    incoming = {pid: {} for pid in P}
    for f in fleets:
        fo, fx, fy, fang, fsh = int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[6])
        hx, hy = math.cos(fang), math.sin(fang)
        best, bestd = None, 1e9
        for pid, p in P.items():
            rx, ry = p["x"] - fx, p["y"] - fy
            proj = rx * hx + ry * hy
            if proj <= 0:
                continue
            perp = abs(rx * hy - ry * hx)
            if perp < p["r"] + 1.5 and proj < bestd:
                bestd, best = proj, pid
        if best is not None:
            incoming[best][fo] = incoming[best].get(fo, 0.0) + fsh

    def enemy_in(pid):
        return sum(s for o, s in incoming[pid].items() if o != player)
    def friendly_in(pid):
        return incoming[pid].get(player, 0.0)

    mine = [p for p in P.values() if p["owner"] == player]
    if not mine:
        return []
    others = [p for p in P.values() if p["owner"] != player]
    remaining = EPISODE_STEPS - step

    front_d = {}
    for m in mine:
        front_d[m["id"]] = min((math.hypot(m["x"] - o["x"], m["y"] - o["y"]) for o in others),
                               default=0.0)

    # available surplus per planet (after threat-aware reserve)
    avail = {}
    for m in mine:
        threat = enemy_in(m["id"])
        if threat >= m["ships"] - 1:
            avail[m["id"]] = 0
            continue
        reserve = max(2.0, threat, 0.10 * m["ships"])
        avail[m["id"]] = max(0, int(m["ships"] - reserve))

    # build global target list with value, then assign sources greedily
    cands = []
    for tgt in others:
        # representative eta from nearest of my planets
        near = min(mine, key=lambda m: math.hypot(m["x"] - tgt["x"], m["y"] - tgt["y"]))
        d0 = math.hypot(near["x"] - tgt["x"], near["y"] - tgt["y"])
        eta0 = max(1.0, d0 / _speed(max(1, tgt["ships"] + 1)))
        if tgt["owner"] == -1:
            req = tgt["ships"] + 1 - friendly_in(tgt["id"])
            pay = tgt["prod"] * max(0.0, remaining - eta0)
            if tgt["id"] in comet_ids:
                pay *= 0.35
            if req <= 0 or pay < req:
                continue
            val = (tgt["prod"] + 0.3) * 1.3 / (req * (1.0 + 0.04 * eta0))
        else:
            req = tgt["ships"] + tgt["prod"] * eta0 + enemy_in(tgt["id"]) - friendly_in(tgt["id"]) + 3
            if req <= 0:
                req = 1
            val = (tgt["prod"] + 0.3) * 2.0 / (req * (1.0 + 0.04 * eta0))
        cands.append((val, tgt, req))
    cands.sort(key=lambda c: -c[0])

    moves = []
    for val, tgt, req in cands:
        if req < 1:
            continue
        # collect nearest sources (skip sun-blocked) until req covered
        srcs = sorted([m for m in mine if avail[m["id"]] >= 1],
                      key=lambda m: math.hypot(m["x"] - tgt["x"], m["y"] - tgt["y"]))
        plan = []
        covered = 0.0
        for s in srcs:
            if covered >= req:
                break
            contrib = min(avail[s["id"]], int(math.ceil(req - covered)))
            if contrib < 1:
                continue
            # intercept aim per source
            ships_guess = max(1, contrib)
            ax, ay = tgt["x"], tgt["y"]
            for _ in range(3):
                dist = math.hypot(ax - s["x"], ay - s["y"])
                eta = max(1.0, dist / _speed(ships_guess))
                ax, ay = predict(tgt, eta)
            if _seg_hits_sun(s["x"], s["y"], ax, ay):
                continue
            if eta > remaining:
                continue
            plan.append((s, contrib, ax, ay, eta))
            covered += contrib
        if covered < req:
            continue  # cannot raise a decisive package this turn
        # enemy targets: recompute req with the slowest eta (garrison grows until last arrival)
        if tgt["owner"] != -1 and plan:
            worst_eta = max(pl[4] for pl in plan)
            req2 = tgt["ships"] + tgt["prod"] * worst_eta + enemy_in(tgt["id"]) - friendly_in(tgt["id"]) + 3
            if covered < req2:
                continue
        for s, contrib, ax, ay, eta in plan:
            ang = math.atan2(ay - s["y"], ax - s["x"])
            moves.append([s["id"], ang, contrib])
            avail[s["id"]] -= contrib
        incoming[tgt["id"]][player] = incoming[tgt["id"]].get(player, 0.0) + covered

    # funnel: rear planets with big leftover surplus send it to the friendliest front planet
    for s in mine:
        left = avail[s["id"]]
        if left < 25 or len(mine) < 2:
            continue
        fwd = [m for m in mine if m["id"] != s["id"]
               and front_d[m["id"]] < front_d[s["id"]] - 8.0]
        if not fwd:
            continue
        dest = min(fwd, key=lambda m: math.hypot(m["x"] - s["x"], m["y"] - s["y"]))
        ax, ay = dest["x"], dest["y"]
        for _ in range(3):
            dist = math.hypot(ax - s["x"], ay - s["y"])
            eta = max(1.0, dist / _speed(left))
            ax, ay = predict(dest, eta)
        if _seg_hits_sun(s["x"], s["y"], ax, ay) or eta > remaining:
            continue
        ang = math.atan2(ay - s["y"], ax - s["x"])
        moves.append([s["id"], ang, left])
        avail[s["id"]] = 0

    return moves
