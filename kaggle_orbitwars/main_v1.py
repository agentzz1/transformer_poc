"""Orbit Wars agent — value-based expansion with interception, sun-avoidance, defense.

Strategy:
  * Track signed angular velocity per planet across turns -> predict future positions.
  * Intercept moving targets (lead the aim point at fleet arrival time).
  * Skip launches whose straight path crosses the sun (fleet would be destroyed).
  * Attribute in-flight fleets to target planets -> avoid over-sending, estimate threats.
  * Each owned planet picks its best affordable target by value = production / (cost * eta);
    sends minimal sufficient force, keeps a defensive reserve. Threatened planets hold.
"""
import math

CENTER = (50.0, 50.0)
SUN_R = 10.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0

_prev = {}  # planet id -> (x, y) from last turn (module state across turns)

def _speed(ships):
    ships = max(1, int(ships))
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5

def _orbiting(x, y, r):
    return math.hypot(x - CENTER[0], y - CENTER[1]) + r < ROT_LIMIT

def _rotate(x, y, ang):
    dx, dy = x - CENTER[0], y - CENTER[1]
    c, s = math.cos(ang), math.sin(ang)
    return CENTER[0] + dx * c - dy * s, CENTER[1] + dx * s + dy * c

def _seg_hits_sun(x0, y0, x1, y1):
    # min distance from segment to center < SUN_R ?
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(x0 - CENTER[0], y0 - CENTER[1]) < SUN_R
    t = ((CENTER[0] - x0) * dx + (CENTER[1] - y0) * dy) / L2
    t = max(0.0, min(1.0, t))
    px, py = x0 + t * dx, y0 + t * dy
    return math.hypot(px - CENTER[0], py - CENTER[1]) < SUN_R

def agent(obs):
    global _prev
    if isinstance(obs, dict):
        planets = obs.get("planets", []); fleets = obs.get("fleets", [])
        player = obs.get("player", 0); av = obs.get("angular_velocity", 0.0)
    else:
        planets = obs.planets; fleets = obs.fleets; player = obs.player
        av = getattr(obs, "angular_velocity", 0.0)

    # planet tuple: [id, owner, x, y, radius, ships, production]
    P = {int(p[0]): {"id": int(p[0]), "owner": int(p[1]), "x": float(p[2]), "y": float(p[3]),
                     "r": float(p[4]), "ships": float(p[5]), "prod": float(p[6])} for p in planets}

    # signed angular velocity per orbiting planet, tracked from last turn
    cur = {}
    for pid, p in P.items():
        cur[pid] = (p["x"], p["y"])
        sav = 0.0
        if _orbiting(p["x"], p["y"], p["r"]):
            sav = av  # default magnitude
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

    # attribute in-flight fleets to nearest planet along their heading -> incoming by owner
    incoming = {pid: {} for pid in P}  # pid -> {owner: ships}
    for f in fleets:
        fid, fo, fx, fy, fang, ffrom, fsh = (int(f[0]), int(f[1]), float(f[2]), float(f[3]),
                                             float(f[4]), int(f[5]), float(f[6]))
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

    mine = [p for p in P.values() if p["owner"] == player]
    if not mine:
        return []

    # threat per my planet: enemy incoming
    def enemy_incoming(pid):
        return sum(s for o, s in incoming[pid].items() if o != player)
    def friendly_incoming(pid):
        return incoming[pid].get(player, 0.0)

    moves = []
    for src in mine:
        threat = enemy_incoming(src["id"])
        # hold if threatened (keep garrison to defend)
        if threat >= src["ships"] - 1:
            continue
        reserve = max(2.0, 0.15 * src["ships"], threat)  # keep some at home
        avail = src["ships"] - reserve
        if avail < 1:
            continue
        best, best_val, best_send, best_ang = None, -1.0, 0, 0.0
        for tgt in P.values():
            if tgt["id"] == src["id"]:
                continue
            if tgt["owner"] == player:
                continue  # don't attack own (reinforcement handled implicitly by holding)
            # intercept: iterate eta <-> aim
            ships_guess = max(1.0, tgt["ships"] + 1)
            ax, ay = tgt["x"], tgt["y"]; eta = 1.0
            for _ in range(4):
                dist = math.hypot(ax - src["x"], ay - src["y"])
                eta = dist / _speed(ships_guess)
                ax, ay = predict(tgt, eta)
                grow = tgt["prod"] * eta if tgt["owner"] >= 0 else 0.0
                ships_guess = max(1.0, tgt["ships"] + grow + 1 + 2)
            if _seg_hits_sun(src["x"], src["y"], ax, ay):
                continue
            # required accounts for growth, enemy reinforcement, minus friendly already incoming
            grow = tgt["prod"] * eta if tgt["owner"] >= 0 and tgt["owner"] != player else 0.0
            req = tgt["ships"] + grow + enemy_incoming(tgt["id"]) - friendly_incoming(tgt["id"]) + 1 + 2
            if req <= 0:
                continue
            if req > avail:
                continue
            val = (tgt["prod"] + 0.2) / (req * (1.0 + 0.05 * eta))
            # bonus for neutral (cheap expansion) and comets handled as normal planets
            if tgt["owner"] == -1:
                val *= 1.3
            if val > best_val:
                best_val, best, best_send = val, tgt, int(math.ceil(req))
                best_ang = math.atan2(ay - src["y"], ax - src["x"])
        if best is not None and best_send >= 1 and best_send <= src["ships"]:
            moves.append([src["id"], best_ang, best_send])
    return moves
