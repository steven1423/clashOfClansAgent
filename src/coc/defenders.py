"""
Traps, defending Clan Castle troops, and defending heroes.

These are the two failure modes real players complain about most, and neither
existed in the simulation before.

TRAPS. Critically, an Invisibility Spell does NOT protect against them --
Fandom is explicit: "Traps are not affected by this spell. An invisible troop
walking over a trap will trigger as normal." So a perfect cloaking rhythm still
walks her onto a Giant Bomb.
    https://clashofclans.fandom.com/wiki/Invisibility_Spell

CLAN CASTLE. The Royal Champion "will leave her previously targeted defense and
instead engage with the troops and/or heroes", then resume on the NEAREST
defense -- not her original target. That is the real cost: not the damage, but
the tempo. Every second she spends killing an Ice Golem is a second of
invisibility burned for zero progress. And once deployed, CC troops have no
leash at all: they chase across the whole map.
    https://clashofclans.fandom.com/wiki/Royal_Champion
    https://clashofclans.fandom.com/wiki/Clan_Castle

The Headhunter deserves special mention: it deals FOUR TIMES damage to heroes
(500 DPS at TH15 max) and poisons on hit. It is the purpose-built counter to
exactly this attack.
    https://clashofclans.fandom.com/wiki/Headhunter
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import config as C


# ----------------------------------------------------------------------
# Traps
# ----------------------------------------------------------------------
@dataclass
class Trap:
    name: str
    x: int
    y: int
    w: int
    h: int
    trigger: float
    damage: float
    radius: float
    fired: bool = False

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def distance_to(self, px: float, py: float) -> float:
        return math.hypot(self.cx - px, self.cy - py)


# ----------------------------------------------------------------------
# Defending units (Clan Castle troops and defending heroes)
# ----------------------------------------------------------------------
@dataclass
class Defender:
    name: str
    hp: float
    max_hp: float
    dps: float
    rng: float
    speed: float                 # tiles / second
    x: float
    y: float
    is_dead: bool = False
    # specials
    hero_multiplier: float = 1.0     # Headhunter hits heroes 4x
    poison_on_hit: bool = False      # Headhunter
    freeze_on_death: float = 0.0     # Ice Golem
    freeze_radius: float = 0.0
    leash: float = 0.0               # 0 = chases forever (CC troops)
    home: Optional[tuple] = None     # defending heroes patrol their altar

    def distance_to(self, px: float, py: float) -> float:
        return math.hypot(self.x - px, self.y - py)

    def effective_dps(self) -> float:
        return self.dps * self.hero_multiplier

    def step(self, hero_pos, dt: float, hero_visible: bool) -> None:
        """Chase the hero. While she is invisible they cannot see her, so they
        hold position -- invisibility breaks aggro as well as damage."""
        if self.is_dead or not hero_visible:
            return
        if self.leash > 0 and self.home is not None:
            if math.hypot(self.home[0] - hero_pos[0], self.home[1] - hero_pos[1]) > self.leash:
                # lured too far: a defending hero retreats to its patrol zone
                ang = math.atan2(self.home[1] - self.y, self.home[0] - self.x)
                if math.hypot(self.home[0] - self.x, self.home[1] - self.y) > 0.5:
                    self.x += math.cos(ang) * self.speed * dt
                    self.y += math.sin(ang) * self.speed * dt
                return
        d = self.distance_to(*hero_pos)
        if d > self.rng:
            ang = math.atan2(hero_pos[1] - self.y, hero_pos[0] - self.x)
            self.x += math.cos(ang) * self.speed * dt
            self.y += math.sin(ang) * self.speed * dt


def make_cc_troops(x: float, y: float, comp: Optional[List[str]] = None) -> List[Defender]:
    """Build a defending Clan Castle.

    Default is a realistic anti-hero composition for TH15: two Headhunters
    (the dedicated hero counter, 4x damage) plus an Ice Golem (tempo counter --
    high HP to soak her time, and a death freeze). ~36 of the 50 housing space.
    """
    comp = comp or C.CC_COMPOSITION
    out: List[Defender] = []
    for name in comp:
        spec = C.CC_TROOPS[name]
        out.append(Defender(
            name=name, hp=spec["hp"], max_hp=spec["hp"], dps=spec["dps"],
            rng=spec["range"], speed=spec["speed"] / 8.0,
            x=x + random.uniform(-1.5, 1.5), y=y + random.uniform(-1.5, 1.5),
            hero_multiplier=spec.get("hero_mult", 1.0),
            poison_on_hit=spec.get("poison", False),
            freeze_on_death=spec.get("freeze", 0.0),
            freeze_radius=spec.get("freeze_radius", 0.0),
        ))
    return out


def make_defending_queen(x: float, y: float) -> Defender:
    """Defending Archer Queen, level 90. Unlike CC troops she patrols: pull her
    far enough from her altar and she retreats and re-engages inside her zone.
        https://clashofclans.fandom.com/wiki/Archer_Queen
    """
    s = C.DEFENDING_QUEEN
    return Defender(name="Archer Queen", hp=s["hp"], max_hp=s["hp"], dps=s["dps"],
                    rng=s["range"], speed=s["speed"] / 8.0, x=x, y=y,
                    leash=s["patrol"], home=(x, y))
