"""
The attacking army: Dragons, Baby Dragons, Grand Warden, Archer Queen.

The Royal Champion charge is the opening phase, not the attack. This module is
the rest of it.

The three mechanics that decide whether a mass-dragon attack triples:

TARGETING. Dragons and Baby Dragons have NO preferred target -- they fly at the
nearest BUILDING, collectors included. That single rule is the entire reason
funnelling exists: one uncleared hut three tiles off-axis peels the whole stack
sideways and it rings the base instead of penetrating it.
    https://clashofclans.fandom.com/wiki/Dragon

THE BABY DRAGON TANTRUM. A Baby Dragon with no allied air unit within 4.5 tiles
gets +100% damage and +50% attack speed -- 310 DPS instead of 155, which is
nearly a full Dragon for half the housing. So the two funnel Baby Dragons must
be dropped far from the stack AND far from each other. Drop them together and
you have thrown away both the funnel and the damage.
    https://clashofclans.fandom.com/wiki/Baby_Dragon

THE AIR SWEEPER. 120-degree arc, eight possible facings, locked before the
battle starts, zero damage. It pushes air units back four tiles and mutes them
for 1.2 seconds. It does not kill your army, it kills your clock -- and the
RC-charge dragon list carries no Rage and no Haste to punch through it. Which
is why the only answer is to come in on a side it is not facing.
    https://clashofclans.fandom.com/wiki/Air_Sweeper
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import config as C


@dataclass
class Unit:
    """An attacking unit. The Royal Champion is handled separately because she
    has a completely different targeting rule (defenses only)."""
    kind: str
    hp: float
    max_hp: float
    dps: float
    rng: float
    speed: float                  # tiles / second
    x: float
    y: float
    flying: bool = True
    splash: float = 0.0
    is_dead: bool = False
    mute_timer: float = 0.0       # Air Sweeper knockback lockout
    invuln_timer: float = 0.0     # Eternal Tome
    aura_bonus: float = 0.0       # Grand Warden Life Aura, applied at spawn
    defenses_only: bool = False   # Stone Slammer ignores non-defenses
    target_uid: Optional[int] = None

    def distance_to(self, px: float, py: float) -> float:
        return math.hypot(self.x - px, self.y - py)


SPECS = {"Dragon": C.DRAGON, "Baby Dragon": C.BABY_DRAGON,
         "Grand Warden": C.WARDEN, "Archer Queen": C.QUEEN,
         "Dragon Duke": C.DRAGON_DUKE, "Stone Slammer": C.STONE_SLAMMER}


def make_unit(kind: str, x: float, y: float) -> Unit:
    spec = SPECS[kind]
    hp = float(spec["hp"])
    dps = float(spec["dps"])
    if kind == "Archer Queen":                # Giant Arrow passive bonuses
        hp += C.QUEEN_HP_BONUS
        dps += C.QUEEN_DPS_BONUS
    return Unit(kind=kind, hp=hp, max_hp=hp, dps=dps, rng=float(spec["rng"]),
                speed=spec["speed"] / 8.0, x=float(x), y=float(y),
                flying=(kind != "Archer Queen"),
                splash=float(spec.get("splash", 0.0)),
                defenses_only=(kind == "Stone Slammer"))


# ----------------------------------------------------------------------
# Air Sweeper
# ----------------------------------------------------------------------
@dataclass
class Sweeper:
    """Wraps an Air Sweeper building with its facing and cooldown."""
    x: float
    y: float
    facing_deg: float             # one of 8 directions, 45 degrees apart
    cooldown: float = 0.0

    def covers(self, px: float, py: float) -> bool:
        d = math.hypot(px - self.x, py - self.y)
        if d > C.SWEEPER_RANGE or d < C.SWEEPER_MIN_RANGE:
            return False
        ang = math.degrees(math.atan2(py - self.y, px - self.x))
        diff = abs((ang - self.facing_deg + 180.0) % 360.0 - 180.0)
        return diff <= C.SWEEPER_ARC_DEG / 2.0

    def blast(self, units: List[Unit]) -> int:
        """Push every flying unit inside the cone. Returns how many were hit."""
        hit = 0
        for u in units:
            if u.is_dead or not u.flying:
                continue
            if not self.covers(u.x, u.y):
                continue
            ang = math.atan2(u.y - self.y, u.x - self.x)
            u.x = float(np.clip(u.x + math.cos(ang) * C.SWEEPER_PUSH,
                                0, C.GRID_SIZE - 1))
            u.y = float(np.clip(u.y + math.sin(ang) * C.SWEEPER_PUSH,
                                0, C.GRID_SIZE - 1))
            u.mute_timer = C.SWEEPER_MUTE
            hit += 1
        self.cooldown = C.SWEEPER_COOLDOWN
        return hit


def assign_facings(sweeper_positions: List[Tuple[float, float]],
                   rng: random.Random) -> List[float]:
    """Give each Air Sweeper one of the eight legal facings.

    A competent base designer points them at offset quadrants so that no single
    side of the base is completely safe -- the attacker's realistic goal is to
    find the side covered by ONE sweeper rather than two. That is modelled by
    pushing the second sweeper roughly opposite the first.
    """
    step = 360.0 / C.SWEEPER_FACINGS
    out: List[float] = []
    base = rng.randrange(C.SWEEPER_FACINGS) * step
    for i in range(len(sweeper_positions)):
        offset = (i * (C.SWEEPER_FACINGS // 2) + rng.choice([-1, 0, 0, 1]))
        out.append(((base + offset * step) % 360.0))
    return out


def sweeper_coverage_map(sweepers: List[Sweeper]) -> np.ndarray:
    """A 44x44 map of how many Air Sweeper cones cover each tile.

    This goes straight into the observation. It is the map the agent needs in
    order to answer the question the whole air phase turns on: which side do I
    bring the dragons in from?
    """
    cov = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
    yy, xx = np.indices((C.GRID_SIZE, C.GRID_SIZE))
    for s in sweepers:
        dx = xx - s.x
        dy = yy - s.y
        d = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx))
        diff = np.abs((ang - s.facing_deg + 180.0) % 360.0 - 180.0)
        cov += ((d <= C.SWEEPER_RANGE) & (d >= C.SWEEPER_MIN_RANGE)
                & (diff <= C.SWEEPER_ARC_DEG / 2.0)).astype(np.float32)
    return cov


# ----------------------------------------------------------------------
# Unit behaviour
# ----------------------------------------------------------------------
def is_enraged(u: Unit, units: List[Unit]) -> Tuple[bool, float, float]:
    """Baby Dragon tantrum and Dragon Duke Royal Rampage.

    Same idea, different radii and different owners: the buff fires only while
    NO friendly air unit is nearby. Baby Dragon 4.5 tiles, Dragon Duke 6.0.
    Both are why those units are dropped on their own flank -- park either one
    inside the dragon stack and the buff simply never turns on.

    Returns (enraged, damage multiplier, speed multiplier).
    """
    if u.kind == "Baby Dragon":
        radius, dmg, spd = C.TANTRUM_RADIUS, C.TANTRUM_DAMAGE_MULT, C.TANTRUM_SPEED_MULT
    elif u.kind == "Dragon Duke":
        radius, dmg, spd = (C.DUKE_RAMPAGE_RADIUS, C.DUKE_RAMPAGE_DAMAGE,
                            C.DUKE_RAMPAGE_SPEED)
    else:
        return False, 1.0, 1.0
    for o in units:
        if o is u or o.is_dead or not o.flying:
            continue
        if u.distance_to(o.x, o.y) <= radius:
            return False, 1.0, 1.0
    return True, dmg, spd


def arrow_value_map(deploy_mask, buildings, grid_size: int) -> np.ndarray:
    """For every legal Queen deploy cell, how many Air Defenses a Giant Arrow
    fired from there would pierce.

    This is the whole "line up the arrow" problem, solved exactly and handed to
    the agent as an observation channel. For each candidate tile it finds the
    NEAREST BUILDING (which is what the Queen will target, and therefore what
    aims the arrow), casts the ray through it, and counts the Air Defenses whose
    centre falls within the arrow's 1-tile hit radius of that line.

    Vectorised, because doing it per-tile in Python for 484 tiles x ~350
    buildings every reset is far too slow.
    """
    g = grid_size
    out = np.zeros((C.ACTION_GRID, C.ACTION_GRID), dtype=np.float32)
    alive = [b for b in buildings if not b.is_dead and b.cat != 6]
    if not alive:
        return out
    bx = np.array([b.cx for b in alive], dtype=np.float32)
    by = np.array([b.cy for b in alive], dtype=np.float32)
    ads = [b for b in alive if b.name in ("Air Defense", "Air Sweeper")]
    if not ads:
        return out
    ax = np.array([b.cx for b in ads], dtype=np.float32)
    ay = np.array([b.cy for b in ads], dtype=np.float32)

    cells = np.flatnonzero(deploy_mask)
    if len(cells) == 0:
        return out
    cy_i, cx_i = np.divmod(cells, C.ACTION_GRID)
    px = (cx_i * C.ACTION_STRIDE + C.ACTION_STRIDE / 2).astype(np.float32)
    py = (cy_i * C.ACTION_STRIDE + C.ACTION_STRIDE / 2).astype(np.float32)

    # nearest building to each candidate tile == what the Queen will aim at
    d2 = (px[:, None] - bx[None, :]) ** 2 + (py[:, None] - by[None, :]) ** 2
    nearest = np.argmin(d2, axis=1)
    tx, ty = bx[nearest], by[nearest]

    # unit vector from the Queen through her target
    dx, dy = tx - px, ty - py
    norm = np.hypot(dx, dy)
    norm[norm < 1e-6] = 1e-6
    dx, dy = dx / norm, dy / norm

    # perpendicular distance from each Air Defense to each ray, forward only
    rx = ax[None, :] - px[:, None]
    ry = ay[None, :] - py[:, None]
    along = rx * dx[:, None] + ry * dy[:, None]
    perp = np.abs(rx * dy[:, None] - ry * dx[:, None])
    hits = ((perp <= C.GIANT_ARROW_WIDTH) & (along > 0)
            & (along <= C.GIANT_ARROW_RANGE)).sum(axis=1)

    out[cy_i, cx_i] = hits.astype(np.float32)
    return out
