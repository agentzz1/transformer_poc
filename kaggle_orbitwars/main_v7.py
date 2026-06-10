"""Orbit Wars v7 — v5 focus-drip strategy on top of an exact forward simulator.

Instead of heading-based attribution, every in-flight fleet is simulated turn by
turn against moving planets to get its exact arrival turn and target. Each
planet then gets a garrison timeline (production + all known arrivals, combat
rules applied), giving exact ship requirements at any future arrival turn:
  * defense: if my planet's timeline shows a flip, neighbors send the deficit
    so it lands before the flip;
  * attack: required force = timeline garrison at my arrival + margin;
  * neutrals are dripped at (they don't regrow), enemies get decisive packages.
"""
import math

CENTER = (50.0, 50.0)
SUN_R = 10.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0
EPISODE_STEPS = 500
HORIZON = 70  # how far ahead we simulate

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

def _seg_hits_circle(x0, y0, x1, y1, cx, cy, r):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(x0 - cx, y0 - cy) <= r
    t = ((cx - x0) * dx + (cy - y0) * dy) / L2
    t = max(0.0, min(1.0, t))
    px, py = x0 + t * dx, y0 + t * dy
    return math.hypot(px - cx, py - cy) <= r

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

    # signed angular velocity from previous positions
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

    def planet_pos(p, dt):
        if abs(p["sav"]) < 1e-9:
            return p["x"], p["y"]
        return _rotate(p["x"], p["y"], p["sav"] * dt)

    # --- exact fleet arrival simulation ---
    # arrivals[pid] = list of (turns_from_now, owner, ships)
    arrivals = {pid: [] for pid in P}
    for f in fleets:
        fo, fx, fy, fang, fsh = int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[6])
        sp = _speed(fsh)
        hx, hy = math.cos(fang) * sp, math.sin(fang) * sp
        x, y = fx, fy
        for dt in range(1, HORIZON):
            nx, ny = x + hx, y + hy
            if _seg_hits_sun(x, y, nx, ny, margin=0.0):
                break
            if nx < 0 or nx > 100 or ny < 0 or ny > 100:
                break
            hit = None
            for pid, p in P.items():
                px, py = planet_pos(p, dt)
                if _seg_hits_circle(x, y, nx, ny, px, py, p["r"]):
                    hit = pid
                    break
            if hit is not None:
                arrivals[hit].append((dt, fo, fsh))
                break
            x, y = nx, ny

    # --- garrison timeline per planet ---
    # state[pid][dt] = (owner, ships) after combat at turn dt
    def timeline(pid, extra=None, upto=HORIZON):
        p = P[pid]
        owner, ships = p["owner"], p["ships"]
        evs = {}
        for (dt, fo, fsh) in arrivals[pid]:
            evs.setdefault(dt, {}).setdefault(fo, 0.0)
            evs[dt][fo] += fsh
        if extra:
            for (dt, fo, fsh) in extra:
                evs.setdefault(dt, {}).setdefault(fo, 0.0)
                evs[dt][fo] += fsh
        out = []
        for dt in range(1, upto):
            if owner >= 0:
                ships += p["prod"]
            if dt in evs:
                groups = sorted(evs[dt].items(), key=lambda kv: -kv[1])
                if len(groups) >= 2:
                    atk_o, atk_s = groups[0]
                    atk_s -= groups[1][1]
                else:
                    atk_o, atk_s = groups[0]
                if atk_s > 0:
                    if atk_o == owner:
                        ships += atk_s
                    elif atk_s > ships:
                        owner, ships = atk_o, atk_s - ships
                    else:
                        ships -= atk_s
            out.append((owner, ships))
        return out  # index dt-1 -> state after turn dt

    mine = [p for p in P.values() if p["owner"] == player]
    if not mine:
        return []
    others = [p for p in P.values() if p["owner"] != player]
    remaining = EPISODE_STEPS - step

    tl = {pid: timeline(pid) for pid in P}

    front_d = {}
    for m in mine:
        front_d[m["id"]] = min((math.hypot(m["x"] - o["x"], m["y"] - o["y"]) for o in others),
                               default=0.0)

    # surplus: max ships we can send while my planet never flips in its timeline
    avail = {}
    deficit = {}  # pid -> (deficit ships, flip turn)
    for m in mine:
        line = tl[m["id"]]
        flip = next((i + 1 for i, (o, s) in enumerate(line) if o != player), None)
        if flip is not None:
            need_extra = 1.0
            for i, (o, s) in enumerate(line):
                if o != player:
                    need_extra = max(need_extra, s + 1)
            deficit[m["id"]] = (need_extra, flip)
            avail[m["id"]] = 0
            continue
        # safe to send: garrison now minus worst future dip caused by known attacks
        min_future = min((s for (o, s) in line), default=m["ships"])
        safe = min(m["ships"], min_future) - max(1.0, 0.05 * m["ships"])
        avail[m["id"]] = max(0, int(safe))

    moves = []

    # ---- phase 0: rescue threatened planets (exact deficit, before flip turn) ----
    for pid, (df, flip) in sorted(deficit.items(), key=lambda kv: kv[1][1]):
        tgt = P[pid]
        helpers = sorted([m for m in mine if m["id"] != pid and avail[m["id"]] >= 3],
                         key=lambda m: math.hypot(m["x"] - tgt["x"], m["y"] - tgt["y"]))
        needed = df
        for h in helpers:
            if needed <= 0:
                break
            contrib = min(avail[h["id"]], int(math.ceil(needed)))
            if contrib < 3:
                continue
            ax, ay = tgt["x"], tgt["y"]; eta = 1.0
            for _ in range(3):
                dist = math.hypot(ax - h["x"], ay - h["y"])
                eta = max(1.0, dist / _speed(contrib))
                ax, ay = planet_pos(tgt, eta)
            if eta > flip + 2 or _seg_hits_sun(h["x"], h["y"], ax, ay):
                continue  # help would land after the flip
            ang = math.atan2(ay - h["y"], ax - h["x"])
            moves.append([h["id"], ang, contrib])
            avail[h["id"]] -= contrib
            needed -= contrib

    # ---- phase 1: focus-drip at neutrals (need from exact timeline) ----
    need = {}
    for t in others:
        line = tl[t["id"]]
        if t["owner"] == -1:
            # ships at a representative future arrival (use ~current; neutrals static unless contested)
            fin_owner, fin_ships = line[min(len(line) - 1, 15)]
            if fin_owner == player:
                continue  # already being captured by my inbound
            need[t["id"]] = fin_ships + 1

    for src in sorted(mine, key=lambda m: front_d[m["id"]]):
        sur = avail[src["id"]]
        if sur < 3:
            continue
        best = None; best_val = 0.0; best_aim = None
        for t in others:
            if t["owner"] != -1 or t["id"] not in need:
                continue
            rem_need = need[t["id"]]
            if rem_need <= 0:
                continue
            ships_guess = max(3, min(sur, int(math.ceil(rem_need))))
            ax, ay = t["x"], t["y"]; eta = 1.0
            for _ in range(3):
                dist = math.hypot(ax - src["x"], ay - src["y"])
                eta = max(1.0, dist / _speed(ships_guess))
                ax, ay = planet_pos(t, eta)
            if _seg_hits_sun(src["x"], src["y"], ax, ay) or eta > remaining:
                continue
            pay = t["prod"] * max(0.0, remaining - eta)
            if t["id"] in comet_ids:
                pay *= 0.35
            if pay < rem_need:
                continue
            val = (t["prod"] + 0.3) / (rem_need * (1.0 + 0.05 * eta))
            if val > best_val:
                best_val, best, best_aim = val, t, (ax, ay)
        if best is None:
            continue
        send = min(sur, int(math.ceil(need[best["id"]])))
        if send < 3:
            continue
        ang = math.atan2(best_aim[1] - src["y"], best_aim[0] - src["x"])
        moves.append([src["id"], ang, send])
        avail[src["id"]] -= send
        need[best["id"]] -= send

    # ---- phase 2: decisive packages vs enemy planets (timeline-based requirement) ----
    enemy_t = [t for t in others if t["owner"] != -1]
    enemy_t.sort(key=lambda t: -(t["prod"] + 0.3) / (t["ships"] + 10.0))
    for t in enemy_t:
        srcs = sorted([m for m in mine if avail[m["id"]] >= 3],
                      key=lambda m: math.hypot(m["x"] - t["x"], m["y"] - t["y"]))
        if not srcs:
            continue
        d0 = math.hypot(srcs[0]["x"] - t["x"], srcs[0]["y"] - t["y"])
        eta0 = max(1.0, d0 / _speed(max(1, int(t["ships"]))))
        line = tl[t["id"]]
        idx = min(len(line) - 1, int(eta0))
        own_at, ships_at = line[idx]
        if own_at == player:
            continue
        req = ships_at + t["prod"] * 2 + 4  # margin for eta error
        plan = []; covered = 0.0; worst_eta = eta0
        for s in srcs:
            if covered >= req:
                break
            contrib = min(avail[s["id"]], int(math.ceil(req - covered)))
            if contrib < 3:
                continue
            ax, ay = t["x"], t["y"]; eta = 1.0
            for _ in range(3):
                dist = math.hypot(ax - s["x"], ay - s["y"])
                eta = max(1.0, dist / _speed(contrib))
                ax, ay = planet_pos(t, eta)
            if _seg_hits_sun(s["x"], s["y"], ax, ay) or eta > remaining:
                continue
            plan.append((s, contrib, ax, ay))
            covered += contrib
            worst_eta = max(worst_eta, eta)
        idx2 = min(len(line) - 1, int(worst_eta))
        own2, ships2 = line[idx2]
        if own2 == player:
            continue
        req2 = ships2 + t["prod"] * 2 + 4
        if covered < req2:
            continue
        for s, contrib, ax, ay in plan:
            ang = math.atan2(ay - s["y"], ax - s["x"])
            moves.append([s["id"], ang, contrib])
            avail[s["id"]] -= contrib

    # ---- phase 3: rear -> front funneling ----
    for s in mine:
        left = avail[s["id"]]
        if left < 20 or len(mine) < 2:
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
            ax, ay = planet_pos(dest, eta)
        if _seg_hits_sun(s["x"], s["y"], ax, ay) or eta > remaining:
            continue
        ang = math.atan2(ay - s["y"], ax - s["x"])
        moves.append([s["id"], ang, left])
        avail[s["id"]] = 0

    return moves
