"""Orbit Wars v3 — swarm expansion + funnel reinforcement.

Replay-driven redesign (vs v1 which rated 593):
  * Continuous pressure: every owned planet ships its surplus every turn.
  * Neutrals don't produce -> partial fleets over several turns are fine (swarm).
  * Enemy planets produce -> only attack with a decisive package (incl. friendly inbound).
  * Rear planets funnel surplus to the friendly planet closest to the front.
  * Amortization: only buy a neutral if its production pays back before turn 500.
  * Interception of orbiting targets, sun-avoidance, threat-aware reserves kept from v1.
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

def _seg_hits_sun(x0, y0, x1, y1, margin=1.0):
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

    # track signed angular velocity from previous turn
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

    # attribute in-flight fleets to their target planet
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

    # front distance: for each of my planets, distance to nearest non-mine planet
    front_d = {}
    for m in mine:
        front_d[m["id"]] = min((math.hypot(m["x"] - o["x"], m["y"] - o["y"]) for o in others),
                               default=0.0)

    moves = []
    # remember planned friendly contributions this turn (counts toward decisive packages)
    planned = {pid: 0.0 for pid in P}

    for src in sorted(mine, key=lambda m: front_d[m["id"]]):  # front planets pick targets first
        threat = enemy_in(src["id"])
        garrison_need = threat + 1 if threat > 0 else 0
        reserve = max(1.0, 0.6 * src["prod"], garrison_need)
        surplus = src["ships"] - reserve
        if surplus < 1:
            continue
        surplus = int(surplus)

        # score all candidate targets reachable without the sun
        best = None; best_val = 0.0; best_req = 0; best_aim = (0.0, 0.0); best_eta = 0.0
        for tgt in others:
            # iterate intercept point
            ships_guess = max(1.0, min(surplus, tgt["ships"] + 1))
            ax, ay = tgt["x"], tgt["y"]; eta = 1.0
            for _ in range(3):
                dist = math.hypot(ax - src["x"], ay - src["y"])
                eta = max(1.0, dist / _speed(ships_guess))
                ax, ay = predict(tgt, eta)
            if _seg_hits_sun(src["x"], src["y"], ax, ay):
                continue
            if tgt["owner"] == -1:
                req = tgt["ships"] + 1 - friendly_in(tgt["id"]) - planned[tgt["id"]]
                pay = tgt["prod"] * max(0.0, remaining - eta)
                if tgt["id"] in comet_ids:
                    pay *= 0.35  # comets leave the board; garrison is lost
                if pay < max(1.0, req):  # doesn't amortize
                    continue
                cost = max(1.0, req)
                val = (tgt["prod"] + 0.3) / (cost * (1.0 + 0.04 * eta))
                send = min(surplus, int(math.ceil(req))) if req > 0 else 0
                if req <= 0:
                    continue  # already covered by inbound/planned
                # partial contributions vs neutrals are fine (no regrowth)
                if send < 1:
                    continue
                val *= min(1.0, send / req) ** 0.5  # prefer finishing a capture
            else:
                grow = tgt["prod"] * eta
                req = tgt["ships"] + grow + enemy_in(tgt["id"]) - friendly_in(tgt["id"]) - planned[tgt["id"]] + 2
                if req < 1:
                    req = 1
                if req > surplus:
                    continue  # only decisive packages vs enemies
                send = int(math.ceil(req))
                val = (tgt["prod"] + 0.3) * 2.0 / (req * (1.0 + 0.04 * eta))
            if val > best_val:
                best_val, best, best_req, best_aim, best_eta = val, tgt, send, (ax, ay), eta

        if best is not None and best_req >= 1:
            ang = math.atan2(best_aim[1] - src["y"], best_aim[0] - src["x"])
            moves.append([src["id"], ang, int(best_req)])
            planned[best["id"]] = planned.get(best["id"], 0.0) + best_req
            surplus -= int(best_req)

        # funnel leftover surplus toward the front (only from clearly rear planets)
        if surplus >= 8 and len(mine) > 1:
            fwd = [m for m in mine if m["id"] != src["id"]
                   and front_d[m["id"]] < front_d[src["id"]] - 5.0
                   and enemy_in(m["id"]) <= m["ships"]]
            if fwd:
                dest = min(fwd, key=lambda m: math.hypot(m["x"] - src["x"], m["y"] - src["y"]))
                dist = math.hypot(dest["x"] - src["x"], dest["y"] - src["y"])
                eta = max(1.0, dist / _speed(surplus))
                ax, ay = predict(dest, eta)
                for _ in range(2):
                    dist = math.hypot(ax - src["x"], ay - src["y"])
                    eta = max(1.0, dist / _speed(surplus))
                    ax, ay = predict(dest, eta)
                if not _seg_hits_sun(src["x"], src["y"], ax, ay) and eta < remaining:
                    ang = math.atan2(ay - src["y"], ax - src["x"])
                    moves.append([src["id"], ang, int(surplus)])

    return moves
