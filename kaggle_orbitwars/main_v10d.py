"""Orbit Wars v10d — v5 + phase-1 blackout fix without the drip leak:
targets already covered by an inbound friendly capture are skipped entirely;
remaining small targets get a send floor of 3 instead of stalling the turn;
no shots at rotating targets on turn 0 (rotation sign unknown).

Each planet drips its surplus every turn at one focused target until it flips:
neutrals don't regrow, so sequential waves lose nothing and capture happens at
the earliest possible moment, with no dead garrisons. Enemy planets still
require decisive same-turn packages (they reinforce). Rear planets funnel big
surpluses to the front. Amortization check near the end of the game.
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

    # in-flight friendly/enemy ships per target planet (heading-based attribution)
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

    # surplus per planet after threat-aware reserve
    avail = {}
    for m in mine:
        threat = enemy_in(m["id"])
        if threat >= m["ships"] - 1:
            avail[m["id"]] = 0
            continue
        reserve = max(1.0, threat, 0.08 * m["ships"])
        avail[m["id"]] = max(0, int(m["ships"] - reserve))

    # remaining need per neutral target: garrison + 1 - friendly inbound
    need = {}
    for t in others:
        if t["owner"] == -1:
            need[t["id"]] = t["ships"] + 1 - friendly_in(t["id"])

    moves = []

    # ---- phase 1: focus-drip at neutrals ----
    for src in sorted(mine, key=lambda m: front_d[m["id"]]):
        sur = avail[src["id"]]
        if sur < 3:  # tiny fleets are too slow to be useful
            continue
        best = None; best_val = 0.0; best_aim = None
        for tgt in others:
            if tgt["owner"] != -1:
                continue
            rem_need = need.get(tgt["id"], 0.0)
            if rem_need <= 0:
                continue
            if rem_need <= 2 and friendly_in(tgt["id"]) > 0:
                continue  # capture already inbound; residual is rounding noise
            ships_guess = max(3, min(sur, int(math.ceil(rem_need))))
            ax, ay = tgt["x"], tgt["y"]; eta = 1.0
            for _ in range(3):
                dist = math.hypot(ax - src["x"], ay - src["y"])
                eta = max(1.0, dist / _speed(ships_guess))
                ax, ay = predict(tgt, eta)
            if _seg_hits_sun(src["x"], src["y"], ax, ay) or eta > remaining:
                continue
            if step < 1 and abs(tgt.get("sav", 0.0)) > 1e-9:
                continue  # rotation sign unknown before the first delta
            pay = tgt["prod"] * max(0.0, remaining - eta)
            if tgt["id"] in comet_ids:
                pay *= 0.35
            if pay < rem_need:  # won't amortize before game end
                continue
            val = (tgt["prod"] + 0.3) / (rem_need * (1.0 + 0.05 * eta))
            if val > best_val:
                best_val, best, best_aim = val, tgt, (ax, ay)
        if best is None:
            continue
        send = min(sur, max(3, int(math.ceil(need[best["id"]]))))
        ang = math.atan2(best_aim[1] - src["y"], best_aim[0] - src["x"])
        moves.append([src["id"], ang, send])
        avail[src["id"]] -= send
        need[best["id"]] -= send

    # ---- phase 2: decisive packages vs enemy planets (multi-source, same turn) ----
    enemy_t = [t for t in others if t["owner"] != -1]
    enemy_t.sort(key=lambda t: -(t["prod"] + 0.3) / (t["ships"] + 10.0))
    for tgt in enemy_t:
        srcs = sorted([m for m in mine if avail[m["id"]] >= 3],
                      key=lambda m: math.hypot(m["x"] - tgt["x"], m["y"] - tgt["y"]))
        if not srcs:
            continue
        d0 = math.hypot(srcs[0]["x"] - tgt["x"], srcs[0]["y"] - tgt["y"])
        eta0 = max(1.0, d0 / _speed(max(1, int(tgt["ships"]))))
        req = tgt["ships"] + tgt["prod"] * eta0 + enemy_in(tgt["id"]) - friendly_in(tgt["id"]) + 3
        if req < 1:
            req = 1
        plan = []; covered = 0.0; worst_eta = eta0
        for s in srcs:
            if covered >= req:
                break
            contrib = min(avail[s["id"]], int(math.ceil(req - covered)))
            if contrib < 3:
                continue
            ax, ay = tgt["x"], tgt["y"]; eta = 1.0
            for _ in range(3):
                dist = math.hypot(ax - s["x"], ay - s["y"])
                eta = max(1.0, dist / _speed(contrib))
                ax, ay = predict(tgt, eta)
            if _seg_hits_sun(s["x"], s["y"], ax, ay) or eta > remaining:
                continue
            plan.append((s, contrib, ax, ay))
            covered += contrib
            worst_eta = max(worst_eta, eta)
        req2 = tgt["ships"] + tgt["prod"] * worst_eta + enemy_in(tgt["id"]) - friendly_in(tgt["id"]) + 3
        if covered < req2:
            continue
        for s, contrib, ax, ay in plan:
            ang = math.atan2(ay - s["y"], ax - s["x"])
            moves.append([s["id"], ang, contrib])
            avail[s["id"]] -= contrib

    # ---- phase 3: rear -> front funneling of big leftovers ----
    for s in mine:
        left = avail[s["id"]]
        if left < 25 or len(mine) < 2:
            continue
        fwd = [m for m in mine if m["id"] != s["id"]
               and front_d[m["id"]] < front_d[s["id"]] - 8.0]
        if not fwd:
            continue
        dest = min(fwd, key=lambda m: math.hypot(m["x"] - s["x"], m["y"] - s["y"]))
        ax, ay = dest["x"], dest["y"]; eta = 1.0
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
