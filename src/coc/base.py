"""
Procedural Town Hall 15 base generation.

Produces bases that look like real war bases: Town Hall in a central
compartment, high-value defenses (X-Bow / Inferno / Scattershot / Monolith /
Eagle) in the core, ordinary defenses in a middle ring, collectors and storages
pushed to the outside, walls forming compartments.

Every building is a real object with a footprint, hitpoints, range and targeting
rules, not a single grid cell. That matters: a 3x3 Scattershot and a 2x2 Tesla
behave differently, and the old code collapsed adjacent same-id cells into one
merged "building" via flood fill, which silently fused separate turrets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import config as C
from . import layout as L
from .defenders import Trap as TrapSite


# Category ids used in the building grid / observation channels
CAT_EMPTY = 0
CAT_TOWN_HALL = 1
CAT_AIR_DEFENSE = 2
CAT_HIGH_DEFENSE = 3
CAT_DEFENSE = 4
CAT_NON_DEFENSE = 5
CAT_WALL = 6


@dataclass
class Building:
    uid: int
    name: str
    cat: int
    x: int                    # top-left tile
    y: int
    w: int
    h: int
    hp: float
    max_hp: float
    dps: float = 0.0
    rng: float = 0.0
    min_range: float = 0.0
    hits_ground: bool = True
    splash: bool = False
    tier: str = "normal"
    is_defense: bool = False
    is_dead: bool = False
    # inferno / eagle state
    lock_time: float = 0.0    # seconds this building has held its target

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def distance_to(self, px: float, py: float) -> float:
        """Distance from the building's EDGE, which is how Clash measures range."""
        dx = max(self.x - px, 0.0, px - (self.x + self.w))
        dy = max(self.y - py, 0.0, py - (self.y + self.h))
        return math.hypot(dx, dy)


def _fits(grid: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    if x < 0 or y < 0 or x + w > C.GRID_SIZE or y + h > C.GRID_SIZE:
        return False
    return bool(np.all(grid[y:y + h, x:x + w] == CAT_EMPTY))


def _stamp(grid: np.ndarray, b: Building) -> None:
    grid[b.y:b.y + b.h, b.x:b.x + b.w] = b.cat


def _try_place(grid: np.ndarray, w: int, h: int,
               lo: int, hi: int, tries: int = 400,
               rng: Optional[random.Random] = None) -> Optional[Tuple[int, int]]:
    """Random placement inside the square [lo, hi)."""
    r = rng or random
    for _ in range(tries):
        x = r.randint(lo, max(lo, hi - w))
        y = r.randint(lo, max(lo, hi - h))
        if _fits(grid, x, y, w, h):
            return x, y
    return None


def _try_place_ring(grid: np.ndarray, w: int, h: int,
                    inner: int, outer: int, tries: int = 400,
                    rng: Optional[random.Random] = None) -> Optional[Tuple[int, int]]:
    """Random placement inside the square [outer_lo, outer_hi) but OUTSIDE the
    inner square -- i.e. in a ring."""
    r = rng or random
    lo_o, hi_o = outer, C.GRID_SIZE - outer
    lo_i, hi_i = inner, C.GRID_SIZE - inner
    for _ in range(tries):
        x = r.randint(lo_o, max(lo_o, hi_o - w))
        y = r.randint(lo_o, max(lo_o, hi_o - h))
        inside_inner = (x >= lo_i and x + w <= hi_i and y >= lo_i and y + h <= hi_i)
        if inside_inner:
            continue
        if _fits(grid, x, y, w, h):
            return x, y
    return None


def generate_base(defense_frac: float = 1.0,
                  seed: Optional[int] = None,
                  traps: bool = False,
                  cc: bool = False,
                  hero: bool = False,
                  archetype: Optional[str] = None):
    """Build a Town Hall 15 war base the way a real one is built: walls first.

    A base is a skeleton of wall compartments, and then buildings assigned into
    those compartments BY ROLE. That ordering is the whole difference between a
    layout that looks real and one that just has the right building census.

    The rules encoded here, each from base-design guides or pro tournament
    layouts:

      * 10-16 compartments, interiors 4-6 tiles, max 1-3 defenses in any one --
        more than that puts too much value behind a single wall break
      * SPLIT CORE VALUE: Town Hall, Clan Castle and Eagle Artillery go in
        DIFFERENT compartments, never stacked into one easy push
      * Air Defenses in a diamond, deep, one per compartment, 8-12 tiles apart
        so one Lightning chain cannot take two
      * Scattershots as ISLANDS -- clear tiles around them so troops path past
      * Infernos separated from each other and from the Eagle, so one Freeze
        cannot catch two
      * Monolith directly behind the Town Hall (the pro-layout convention)
      * X-Bows ringing the core behind another defense
      * storages hugging the Air Defenses to soak Lightning splash, and in front
        of the Town Hall to absorb the first hits
      * collectors, army camps and labs OUTSIDE the walls as buffers, plus
        high-hitpoint buildings in the far corners to burn the three-minute clock
      * no OPEN compartments: every one is flood-fill checked and sealed

    defense_frac thins the defensive roster for the curriculum. archetype picks
    the family (box / diamond / ring / askew) or leaves it random.
    """
    r = random.Random(seed)
    grid = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.int16)
    buildings: List[Building] = []
    uid = 0

    arch = archetype or r.choices(L.ARCHETYPES, weights=L.ARCHETYPE_WEIGHTS)[0]
    walls, cells = L.build_skeleton(r, arch)
    L.seal(walls, cells)

    def add(name, cat, x, y, w, h, hp, **kw) -> Optional[Building]:
        nonlocal uid
        if not _fits(grid, x, y, w, h):
            return None
        b = Building(uid=uid, name=name, cat=cat, x=x, y=y, w=w, h=h,
                     hp=float(hp), max_hp=float(hp), **kw)
        _stamp(grid, b)
        buildings.append(b)
        uid += 1
        return b

    def place_in(cell: L.Cell, w: int, h: int, tries: int = 60,
                 pad: int = 0) -> Optional[Tuple[int, int]]:
        for _ in range(tries):
            x = r.randint(cell.x0 + pad, max(cell.x0 + pad, cell.x1 - w - pad))
            y = r.randint(cell.y0 + pad, max(cell.y0 + pad, cell.y1 - h - pad))
            if _fits(grid, x, y, w, h) and not walls[y:y + h, x:x + w].any():
                return x, y
        return None

    def nearest_cell(px: float, py: float, free_only: bool = True) -> L.Cell:
        cand = [c for c in cells if (c.slots < 3 or not free_only)]
        return min(cand or cells,
                   key=lambda c: math.hypot(c.cx - px, c.cy - py))

    by_ring = sorted(cells, key=lambda c: (c.ring, c.dist_to_centre()))

    # ---- Town Hall: its own compartment, offset on anti-3-star layouts ----
    th_cell = by_ring[0] if arch != "askew" else r.choice(by_ring[1:3])
    pos = place_in(th_cell, 4, 4) or place_in(by_ring[0], 4, 4)
    if pos is None:
        pos = (20, 20)
    th = add("Town Hall", CAT_TOWN_HALL, pos[0], pos[1], 4, 4, C.TH_HP,
             dps=C.TH_DPS, rng=C.TH_RANGE, hits_ground=True, splash=True,
             tier="high", is_defense=True)
    th_cell.slots += 2
    thx, thy = (th.cx, th.cy) if th else (22.0, 22.0)

    # ---- roster, thinned by defense_frac -------------------------------
    roster: Dict[str, int] = {}
    for spec in C.DEFENSES:
        keep = spec.count * (defense_frac if spec.tier != "high"
                             else 0.5 + 0.5 * defense_frac)
        n = int(keep) + (1 if r.random() < (keep - int(keep)) else 0)
        if spec.name in ("Monolith", "Eagle Artillery", "Scattershot",
                         "Air Defense", "Air Sweeper") and defense_frac >= 0.5:
            n = max(n, 1 if spec.name != "Air Defense" else min(4, spec.count))
        roster[spec.name] = max(0, min(spec.count, n))
    specs = {s.name: s for s in C.DEFENSES}

    def put(name: str, cell: L.Cell, pad: int = 0) -> Optional[Building]:
        """Place a defense in `cell`, falling back to the nearest compartments
        with room. A base must end up with its full roster -- a Scattershot
        that failed to fit is a missing Scattershot, not a design choice."""
        sp = specs[name]
        order = [cell] + sorted((c for c in cells if c is not cell),
                                key=lambda c: (c.slots,
                                               math.hypot(c.cx - cell.cx,
                                                          c.cy - cell.cy)))
        pos = None
        for cand in order:
            pos = place_in(cand, sp.size[0], sp.size[1], pad=pad)
            if pos is None and pad:
                pos = place_in(cand, sp.size[0], sp.size[1])
            if pos is not None:
                cell = cand
                break
        if pos is None:
            return None
        cell.slots += 1
        return add(name, CAT_AIR_DEFENSE if name in ("Air Defense", "Air Sweeper")
                   else (CAT_HIGH_DEFENSE if sp.tier == "high" else CAT_DEFENSE),
                   pos[0], pos[1], sp.size[0], sp.size[1], sp.hp,
                   dps=sp.dps, rng=sp.rng, min_range=sp.min_range,
                   hits_ground=sp.hits_ground, splash=sp.splash,
                   tier=sp.tier, is_defense=True)

    # ---- Air Defenses: diamond, deep, one per compartment ---------------
    ad_spots, symmetric = L.air_defense_spots(r, cells, arch)
    ad_cells: List[L.Cell] = []
    placed_ads: List[Tuple[float, float]] = []
    for k in range(roster.get("Air Defense", 0)):
        tx, ty = ad_spots[k % 4]
        cell = min((c for c in cells if c not in ad_cells),
                   key=lambda c: math.hypot(c.cx - tx, c.cy - ty), default=None)
        if cell is None:
            break
        b = put("Air Defense", cell)
        if b is not None:
            ad_cells.append(cell)
            placed_ads.append((b.cx, b.cy))

    # ---- Eagle: central, but never sharing with the Town Hall -----------
    if roster.get("Eagle Artillery"):
        for c in by_ring[:6]:
            if c is not th_cell and put("Eagle Artillery", c):
                break

    # ---- Monolith: directly behind the Town Hall ------------------------
    if roster.get("Monolith"):
        ang = math.atan2(thy - 22.0, thx - 22.0)
        bx, by = thx + 5 * math.cos(ang), thy + 5 * math.sin(ang)
        put("Monolith", nearest_cell(bx, by))

    # ---- Clan Castle and Hero Altar --------------------------------------
    # Placed HERE, before the filler defenses, not after. A TH15 base is packed
    # solid by the time the Cannons and Archer Towers are down -- measured, one
    # free 3x3 spot in the whole interior -- so a Clan Castle placed last simply
    # fails, and about half the generated bases came out with no Clan Castle at
    # all. That is not a cosmetic bug: the Castle holds the defending troops,
    # and a recogniser trained on those renders never learns what one looks
    # like. Core buildings get their space first, like a real builder does it.
    cc_pos = None
    if cc:
        for c in by_ring[1:8]:
            if c is not th_cell:
                pos = place_in(c, 3, 3)
                if pos:
                    add("Clan Castle", CAT_NON_DEFENSE, pos[0], pos[1], 3, 3, C.CC_HP)
                    cc_pos = (pos[0] + 1.5, pos[1] + 1.5)
                    break
    altar_pos = None
    if hero:
        for c in by_ring[1:9]:
            pos = place_in(c, 3, 3)
            if pos:
                add("Hero Altar", CAT_NON_DEFENSE, pos[0], pos[1], 3, 3, 1000)
                altar_pos = (pos[0] + 1.5, pos[1] + 1.5)
                break

    # ---- Scattershots: islands, opposing sides --------------------------
    for k in range(roster.get("Scattershot", 0)):
        a = r.uniform(0, 2 * math.pi) + k * math.pi
        put("Scattershot", nearest_cell(22 + 8 * math.cos(a), 22 + 8 * math.sin(a)),
            pad=1)

    # ---- Infernos: separated from each other ----------------------------
    inf_cells: List[L.Cell] = []
    for k in range(roster.get("Inferno Tower", 0)):
        cand = [c for c in by_ring[:9] if c not in inf_cells]
        if not cand:
            break
        c = r.choice(cand)
        if put("Inferno Tower", c):
            inf_cells.append(c)

    # ---- X-Bows ring the core -------------------------------------------
    for k in range(roster.get("X-Bow", 0)):
        a = 2 * math.pi * k / max(1, roster.get("X-Bow", 1)) + r.uniform(-.3, .3)
        put("X-Bow", nearest_cell(22 + 7 * math.cos(a), 22 + 7 * math.sin(a)))

    # ---- Spell Towers: one on the Town Hall (Invisibility is the meta) ---
    if roster.get("Spell Tower"):
        put("Spell Tower", th_cell)
        if roster["Spell Tower"] > 1:
            put("Spell Tower", r.choice(by_ring[1:6]))

    # ---- Air Sweepers: outward, 90-180 degrees apart ---------------------
    for k in range(roster.get("Air Sweeper", 0)):
        a = r.uniform(0, 2 * math.pi) + k * math.pi * r.choice([0.5, 1.0])
        put("Air Sweeper", nearest_cell(22 + 9 * math.cos(a), 22 + 9 * math.sin(a)))

    # ---- everything else fills remaining compartment slots ---------------
    rest = ["Archer Tower", "Cannon", "Wizard Tower", "Hidden Tesla",
            "Mortar", "Bomb Tower"]
    queue: List[str] = []
    for name in rest:
        queue += [name] * roster.get(name, 0)
    r.shuffle(queue)
    for name in queue:
        cand = [c for c in cells if c.slots < 3]
        if not cand:
            cand = cells
        put(name, r.choice(cand))

    # ---- anything the compartments could not take goes in anyway ---------
    # A base is defined by its roster. If the wall skeleton ran out of room, put
    # the remainder in the free space rather than shipping a base that is
    # quietly missing four Cannons.
    #
    # This used to be 120 random guesses inside [6, 38). On a TH15 layout that
    # is nowhere near enough -- measured, the interior has a handful of free
    # 3x3 spots by this point, so random sampling found one about half the time
    # and the generator shipped bases averaging 6.6 Archer Towers instead of 8
    # and 5.7 Cannons instead of 7. Nearly five missing defenses per base, which
    # made every base easier than a real one and taught the recogniser a roster
    # that does not exist. So: scan, do not guess. Prefer inside the walls and
    # near the middle; fall back to the outer ring, which is where a real
    # builder puts the overflow anyway.
    def _scan_defense(name: str) -> bool:
        sp = specs[name]
        w, h = sp.size
        best = None
        for y in range(2, C.GRID_SIZE - h - 1):
            for x in range(2, C.GRID_SIZE - w - 1):
                if not _fits(grid, x, y, w, h):
                    continue
                walled = bool(walls[y:y + h, x:x + w].any())
                if walled:
                    continue
                d = math.hypot(x + w / 2 - 22, y + h / 2 - 22)
                # inside the wall footprint is worth about 10 tiles of centrality
                key = d + (0.0 if 8 <= x <= 33 and 8 <= y <= 33 else 10.0)
                if best is None or key < best[0]:
                    best = (key, x, y)
        if best is None:
            return False
        _, x, y = best
        return add(name,
                   CAT_AIR_DEFENSE if name in ("Air Defense", "Air Sweeper")
                   else (CAT_HIGH_DEFENSE if sp.tier == "high" else CAT_DEFENSE),
                   x, y, w, h, sp.hp,
                   dps=sp.dps, rng=sp.rng, min_range=sp.min_range,
                   hits_ground=sp.hits_ground, splash=sp.splash,
                   tier=sp.tier, is_defense=True) is not None

    for name in rest + ["Air Defense", "X-Bow", "Inferno Tower", "Scattershot",
                        "Spell Tower", "Air Sweeper", "Monolith",
                        "Eagle Artillery"]:
        have = sum(1 for b in buildings if b.name == name)
        for _ in range(max(0, roster.get(name, 0) - have)):
            if not _scan_defense(name):
                break

    # ---- last-resort placement for the two core buildings ----------------
    # Exhaustive scan, nearest-to-centre first, rather than random tries: on a
    # base with one free 3x3 spot left, 400 random guesses find it half the time
    # and a scan finds it always.
    def _scan_place(name: str, hp: float):
        best = None
        for y in range(4, C.GRID_SIZE - 7):
            for x in range(4, C.GRID_SIZE - 7):
                if _fits(grid, x, y, 3, 3) and not walls[y:y + 3, x:x + 3].any():
                    d = math.hypot(x + 1.5 - 22, y + 1.5 - 22)
                    if best is None or d < best[0]:
                        best = (d, x, y)
        if best is None:
            return None
        _, x, y = best
        add(name, CAT_NON_DEFENSE, x, y, 3, 3, hp)
        return (x + 1.5, y + 1.5)

    if cc and cc_pos is None:
        cc_pos = _scan_place("Clan Castle", C.CC_HP)
    if hero and altar_pos is None:
        altar_pos = _scan_place("Hero Altar", 1000.0)

    # ---- storages hug the Air Defenses to soak Lightning splash ---------
    for (ax, ay) in placed_ads:
        for _ in range(2):
            for _try in range(25):
                a = r.uniform(0, 2 * math.pi)
                x = int(ax + r.uniform(3, 5) * math.cos(a))
                y = int(ay + r.uniform(3, 5) * math.sin(a))
                if 0 <= x < C.GRID_SIZE - 3 and 0 <= y < C.GRID_SIZE - 3 \
                        and _fits(grid, x, y, 3, 3) and not walls[y:y+3, x:x+3].any():
                    add("Storage", CAT_NON_DEFENSE, x, y, 3, 3, C.NON_DEFENSE_HP * 2)
                    break

    # ---- walls become real buildings ------------------------------------
    wy, wx = np.nonzero(walls)
    n_walls = 0
    idx = list(range(len(wy)))
    r.shuffle(idx)
    for i in idx:
        if n_walls >= C.WALL_SEGMENTS:
            break
        x, y = int(wx[i]), int(wy[i])
        if grid[y, x] == CAT_EMPTY:
            add("Wall", CAT_WALL, x, y, 1, 1, C.WALL_HP)
            n_walls += 1

    # ---- outer layer: buffers, and high-hitpoint bulk in the corners -----
    # "positioned in corners and edges that require troops to walk long
    # distances to reach" -- clock burn, and it denies deployment tiles.
    corners = [(3, 3), (C.GRID_SIZE - 8, 3), (3, C.GRID_SIZE - 8),
               (C.GRID_SIZE - 8, C.GRID_SIZE - 8)]
    for (cxx, cyy) in corners:
        for _ in range(2):
            for _try in range(30):
                x = cxx + r.randint(-2, 4)
                y = cyy + r.randint(-2, 4)
                if 0 <= x < C.GRID_SIZE - 4 and 0 <= y < C.GRID_SIZE - 4 \
                        and _fits(grid, x, y, 4, 4):
                    add("Army Camp", CAT_NON_DEFENSE, x, y, 4, 4,
                        C.NON_DEFENSE_HP)
                    break
    placed_nd = sum(1 for b in buildings if b.cat == CAT_NON_DEFENSE)
    for _ in range(max(0, C.NON_DEFENSE_COUNT - placed_nd)):
        w = h = r.choice([2, 3, 3, 4])
        for _try in range(60):
            x = r.randint(0, C.GRID_SIZE - w)
            y = r.randint(0, C.GRID_SIZE - h)
            if math.hypot(x - 22, y - 22) < 11:
                continue
            if _fits(grid, x, y, w, h):
                add("Collector", CAT_NON_DEFENSE, x, y, w, h, C.NON_DEFENSE_HP)
                break

    # ---- traps, by convention -------------------------------------------
    trap_list: List[TrapSite] = []
    if traps:
        free = [(x, y) for y in range(4, C.GRID_SIZE - 4)
                for x in range(4, C.GRID_SIZE - 4) if grid[y, x] == CAT_EMPTY]
        r.shuffle(free)
        used: Set[Tuple[int, int]] = set()

        def take_near(px: float, py: float, lo: float, hi: float):
            best = None
            bd = 1e9
            for (x, y) in free:
                if (x, y) in used:
                    continue
                d = math.hypot(x - px, y - py)
                if lo <= d <= hi and d < bd:
                    bd, best = d, (x, y)
            if best:
                used.add(best)
            return best

        # Giant Bombs: deep, between defenses, where troop packs group up
        defs = [b for b in buildings if b.is_defense and b.cat != CAT_TOWN_HALL]
        spec = {n: (t, d, rad) for n, _c, _s, t, d, rad in C.TRAPS}
        for _ in range(7):
            if len(defs) < 2:
                break
            a, b2 = r.sample(defs, 2)
            p = take_near((a.cx + b2.cx) / 2, (a.cy + b2.cy) / 2, 0, 6)
            if p:
                t, d, rad = spec["Giant Bomb"]
                trap_list.append(TrapSite("Giant Bomb", p[0], p[1], 2, 2, t, d, rad))
        # Springs paired with the Giant Bombs, on the same paths
        for _ in range(9):
            p = take_near(22, 22, 5, 14)
            if p:
                t, d, rad = spec["Spring Trap"]
                trap_list.append(TrapSite("Spring Trap", p[0], p[1], 1, 1, t, d, rad))
        for _ in range(8):
            p = take_near(22, 22, 12, 20)
            if p:
                t, d, rad = spec["Bomb"]
                trap_list.append(TrapSite("Bomb", p[0], p[1], 1, 1, t, d, rad))
        p = take_near(22, 22, 0, 7)          # Tornado near the core
        if p:
            t, d, rad = spec["Tornado Trap"]
            trap_list.append(TrapSite("Tornado Trap", p[0], p[1], 1, 1, t, d, rad))
        for _ in range(C.SKELETON_TRAPS):
            p = take_near(22, 22, 8, 16)
            if p:
                trap_list.append(TrapSite("Skeleton Trap", p[0], p[1], 1, 1,
                                          5.0, 0.0, 0.0))

    return grid, buildings, trap_list, cc_pos, altar_pos


def destruction_percent(buildings: List[Building]) -> float:
    """Real Clash destruction % counts BUILDINGS, not wall segments."""
    scoring = [b for b in buildings if b.cat != CAT_WALL]
    if not scoring:
        return 0.0
    dead = sum(1 for b in scoring if b.is_dead)
    return dead / len(scoring)


def stars(buildings: List[Building], th_destroyed: bool) -> int:
    """1 star for 50%, 1 for the Town Hall, 1 for 100%."""
    pct = destruction_percent(buildings)
    s = 0
    if pct >= 0.50:
        s += 1
    if th_destroyed:
        s += 1
    if pct >= 0.999:
        s += 1
    return s
