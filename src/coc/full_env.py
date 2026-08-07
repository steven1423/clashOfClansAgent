"""
The FULL attack: Royal Champion charge, then mass dragons.

This is the whole three-minute battle, not just the charge. The agent decides:

  1. WHERE TO DEPLOY THE ROYAL CHAMPION
  2. WHEN AND WHERE TO CAST EACH INVISIBILITY SPELL   (her life support)
  3. WHEN TO FIRE SEEKING SHIELD
  4. WHEN TO STOP CHARGING AND COMMIT THE AIR ARMY
  5. WHERE TO DROP EACH BABY DRAGON                    (the funnel)
  6. WHERE TO DROP EACH OF THE FOURTEEN DRAGONS        (the side and the shape)
  7. WHERE TO DROP THE WARDEN AND THE QUEEN
  8. WHEN TO FIRE THE ETERNAL TOME

The two things that make this a real strategy problem rather than a shooting
gallery:

AIR DEFENSES ARE THE POINT. They cannot scratch the Royal Champion -- she is a
ground unit -- and they are 540 DPS each against the fourteen Dragons behind
her. The charge exists to delete them. "Harmless to her, lethal to the army she
is clearing for" is the whole thesis, and it is why the reward for killing
something is not the same as the threat it poses.

THE AIR SWEEPERS DECIDE WHICH SIDE YOU COME IN FROM. Two of them, each with a
120-degree cone locked to one of eight facings before the battle starts. They
do no damage at all -- they push the stack back four tiles and mute it for 1.2
seconds, every 5 seconds. Against a 3-minute clock with no Rage and no Haste in
this army, coming in through a cone is how a triple becomes a two-star. The
coverage map is in the observation so the agent can actually see it.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .army import (Sweeper, Unit, arrow_value_map, assign_facings,
                   is_enraged, make_unit, sweeper_coverage_map)
from .base import (Building, CAT_AIR_DEFENSE, CAT_DEFENSE, CAT_EMPTY,
                   CAT_HIGH_DEFENSE, CAT_NON_DEFENSE, CAT_TOWN_HALL, CAT_WALL,
                   destruction_percent, generate_base, stars)
from .defenders import Defender, make_cc_troops, make_defending_queen

# ----------------------------------------------------------------------
# Action space
# ----------------------------------------------------------------------
A_WAIT = 0
A_SHIELD = 1          # Royal Champion: Seeking Shield
A_TOME = 2            # Grand Warden: Eternal Tome
A_ARROW = 3           # Archer Queen: GIANT ARROW
A_DUKE = 4            # Dragon Duke: self-heal
A_SPELL = 5           # 3 .. 3+G^2-1   cast Invisibility on a tile
A_DEPLOY = A_SPELL + C.ACTION_GRID * C.ACTION_GRID
N_ACTIONS_FULL = A_DEPLOY + C.ACTION_GRID * C.ACTION_GRID

# Observation
N_CH = 15
N_SC = 12


class FullAttackEnv:
    def __init__(self, defense_frac: float = 1.0, max_spells: int = 11,
                 seed: Optional[int] = None, record: bool = False,
                 traps: bool = True, cc: bool = True, hero: bool = True):
        self.defense_frac = defense_frac
        self.max_spells = max_spells
        self.use_traps, self.use_cc, self.use_hero = traps, cc, hero
        self.rng = random.Random(seed)
        self.record = record
        self._yy, self._xx = np.indices((C.GRID_SIZE, C.GRID_SIZE))
        self._yy = self._yy.astype(np.float32)
        self._xx = self._xx.astype(np.float32)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, **kw):
        for k in ("defense_frac", "max_spells"):
            if kw.get(k) is not None:
                setattr(self, k, kw[k])
        for k, a in (("traps", "use_traps"), ("cc", "use_cc"), ("hero", "use_hero")):
            if kw.get(k) is not None:
                setattr(self, a, kw[k])

        (self.grid, self.buildings, self.traps,
         self.cc_pos, self.altar_pos) = generate_base(
            self.defense_frac, seed=self.rng.randrange(1 << 30),
            traps=self.use_traps, cc=self.use_cc, hero=self.use_hero)
        self.town_hall = next(b for b in self.buildings if b.cat == CAT_TOWN_HALL)
        self.by_uid = {b.uid: b for b in self.buildings}

        # Air Sweepers get their facings here. They are locked for the battle.
        sw = [b for b in self.buildings if b.name == "Air Sweeper"]
        facings = assign_facings([(b.cx, b.cy) for b in sw], self.rng)
        self.sweepers = [Sweeper(b.cx, b.cy, f) for b, f in zip(sw, facings)]
        self.sweeper_buildings = sw
        self._sweeper_cov = sweeper_coverage_map(self.sweepers)

        # hidden anti-air traps
        self.air_traps: List[Dict] = []
        if self.use_traps:
            free = [(x, y) for y in range(5, C.GRID_SIZE - 5)
                    for x in range(5, C.GRID_SIZE - 5) if self.grid[y, x] == CAT_EMPTY]
            self.rng.shuffle(free)
            it = iter(free)
            for spec, n in ((C.SEEKING_AIR_MINE, C.SEEKING_AIR_MINE["count"]),
                            (C.AIR_BOMB, C.AIR_BOMB["count"])):
                for _ in range(n):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        break
                    self.air_traps.append(dict(x=x, y=y, dmg=spec["damage"],
                                               trig=spec["trigger"],
                                               rad=spec.get("radius", 0.0),
                                               fired=False))

        # Royal Champion
        self.rc_hp = float(C.RC_MAX_HP)
        self.rc_pos = [C.GRID_SIZE / 2.0, C.GRID_SIZE / 2.0]
        self.rc_deployed = False
        self.rc_dead = False
        self.spells_left = int(self.max_spells)
        self.shield_left = C.SHIELD_USES
        self.tome_left = 1
        self.tome_timer = 0.0
        self.arrow_left = 1
        self.duke_heal_left = 1
        self.fox_invis = 0.0
        self.fox_cd = 0.0

        # army
        self.units: List[Unit] = []
        self.queue: List[str] = list(C.DEPLOY_ORDER)
        self.units_lost = 0

        self.defenders: List[Defender] = []
        self.cc_released = self.hero_released = False
        self.freeze_timer = self.hh_poison = 0.0
        self.tower_poison = 0.0
        self.tower_cd: Dict[int, float] = {}

        self.steps = 0
        self.time = 0.0
        self.th_destroyed = False
        self.giga_active = False
        self.poison_timer = 0.0
        self.invis = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        self.eagle_active = False
        self._dirty = True
        self._static: Optional[np.ndarray] = None
        self._ground_threat: Optional[np.ndarray] = None
        self._air_threat: Optional[np.ndarray] = None
        self.prev_destruction = 0.0
        self.frames: List[Dict] = []
        self.stats = dict(key_kills=0, defense_kills=0, ad_killed=0,
                          traps_hit=0, units_lost=0, spells_used=0,
                          sweeper_hits=0, rc_kills=0, arrow_ads=0,
                          arrow_hits=0)
        self._arrow_map = arrow_value_map(self._deploy_mask(),
                                          self.buildings, C.GRID_SIZE)
        return self._obs()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def legal_actions(self) -> np.ndarray:
        m = np.zeros(N_ACTIONS_FULL, dtype=bool)
        if not self.rc_deployed:
            m[A_DEPLOY:] = self._deploy_mask()
            return m
        m[A_WAIT] = True
        if self.shield_left > 0 and not self.rc_dead:
            m[A_SHIELD] = True
        if self.tome_left > 0 and any(u.kind == "Grand Warden" and not u.is_dead
                                      for u in self.units):
            m[A_TOME] = True
        if self.arrow_left > 0 and any(u.kind == "Archer Queen" and not u.is_dead
                                       for u in self.units):
            m[A_ARROW] = True
        if self.duke_heal_left > 0 and any(u.kind == "Dragon Duke" and not u.is_dead
                                           for u in self.units):
            m[A_DUKE] = True
        if self.spells_left > 0 and not self.rc_dead:
            m[A_SPELL:A_DEPLOY] = True
        if self.queue:
            m[A_DEPLOY:] = self._deploy_mask()
        return m

    def _deploy_mask(self) -> np.ndarray:
        occ = self.grid != CAT_EMPTY
        blocked = occ.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                blocked |= np.roll(np.roll(occ, dy, axis=0), dx, axis=1)
        s = C.ACTION_STRIDE
        legal = ~blocked.reshape(C.ACTION_GRID, s, C.ACTION_GRID, s).any(axis=(1, 3))
        if not legal.any():
            legal[0, :] = legal[-1, :] = legal[:, 0] = legal[:, -1] = True
        return legal.reshape(-1)

    @staticmethod
    def _tile(idx: int) -> Tuple[int, int]:
        ay, ax = divmod(idx, C.ACTION_GRID)
        s = C.ACTION_STRIDE
        return ax * s + s // 2, ay * s + s // 2

    # ------------------------------------------------------------------
    def step(self, action: int):
        info: Dict = {}
        reward = 0.0

        if not self.rc_deployed:
            x, y = self._tile(max(0, action - A_DEPLOY))
            self.rc_pos = [float(x), float(y)]
            self.rc_deployed = True
            if self.record:
                self._record()
            return self._obs(), 0.0, False, info

        self.steps += 1
        self.time += C.DT
        reward += C.R_PER_STEP

        # ---- 1. action -------------------------------------------------
        if action == A_SHIELD and self.shield_left > 0:
            self.shield_left -= 1
            reward += self._seeking_shield()
        elif action == A_TOME and self.tome_left > 0:
            self.tome_left -= 1
            self.tome_timer = C.TOME_DURATION
        elif action == A_ARROW and self.arrow_left > 0:
            self.arrow_left -= 1
            reward += self._giant_arrow()
        elif action == A_DUKE and self.duke_heal_left > 0:
            self.duke_heal_left -= 1
            for u in self.units:
                if u.kind == "Dragon Duke" and not u.is_dead:
                    u.hp = min(u.max_hp, u.hp + C.DUKE_HEAL_ABILITY)
        elif A_SPELL <= action < A_DEPLOY and self.spells_left > 0:
            x, y = self._tile(action - A_SPELL)
            mask = ((self._xx - x) ** 2 + (self._yy - y) ** 2) <= C.SPELL_RADIUS ** 2
            np.maximum(self.invis, np.where(mask, C.SPELL_DURATION, 0.0), out=self.invis)
            self.spells_left -= 1
            self.stats["spells_used"] += 1
            reward += C.R_SPELL_CAST
        elif action >= A_DEPLOY and self.queue:
            x, y = self._tile(action - A_DEPLOY)
            kind = self.queue.pop(0)
            u = make_unit(kind, x, y)
            # Grand Warden Life Aura is applied to everything already out
            if kind == "Grand Warden":
                for o in self.units:
                    if not o.is_dead and o.distance_to(x, y) <= C.WARDEN_AURA_RADIUS:
                        o.max_hp *= (1 + C.WARDEN_AURA_HP_BONUS)
                        o.hp *= (1 + C.WARDEN_AURA_HP_BONUS)
            self.units.append(u)

        # ---- 2. timers -------------------------------------------------
        np.subtract(self.invis, C.DT, out=self.invis)
        np.clip(self.invis, 0.0, None, out=self.invis)
        self.tome_timer = max(0.0, self.tome_timer - C.DT)
        self.freeze_timer = max(0.0, self.freeze_timer - C.DT)
        self.hh_poison = max(0.0, self.hh_poison - C.DT)
        self.tower_poison = max(0.0, self.tower_poison - C.DT)
        for u in self.units:
            u.mute_timer = max(0.0, u.mute_timer - C.DT)
            if u.kind == "Dragon Duke" and not u.is_dead:
                # Fire Heart: his only sustain, because Healers cannot target a
                # flying melee unit
                u.hp = min(u.max_hp, u.hp + C.DUKE_REGEN * C.DT)

        # ---- 3. Royal Champion ----------------------------------------
        if not self.rc_dead:
            reward += self._run_rc()

        # ---- 4. the army ----------------------------------------------
        reward += self._run_army()

        # ---- 5. defenses shoot back -----------------------------------
        reward += self._defenses_fire()

        # ---- 6. air sweepers ------------------------------------------
        for s in self.sweepers:
            b = next((bb for bb in self.sweeper_buildings
                      if abs(bb.cx - s.x) < 0.01 and abs(bb.cy - s.y) < 0.01), None)
            if b is not None and b.is_dead:
                continue
            s.cooldown = max(0.0, s.cooldown - C.DT)
            if s.cooldown <= 0:
                n = s.blast(self.units)
                self.stats["sweeper_hits"] += n

        # ---- 7. anti-air traps ----------------------------------------
        reward += self._air_traps()

        # ---- 8. destruction shaping -----------------------------------
        pct = destruction_percent(self.buildings)
        reward += (pct - self.prev_destruction) * 100.0 * C.R_DESTRUCTION_PER_PCT
        self.prev_destruction = pct
        if pct >= C.TH_ACTIVATE_DESTRUCTION:
            self.giga_active = True

        # ---- 9. termination -------------------------------------------
        alive_units = [u for u in self.units if not u.is_dead]
        army_spent = (not self.queue) and (not alive_units)
        done = False
        if pct >= 0.999:
            done = True
            info["end"] = "3-star"
        elif self.steps >= C.MAX_STEPS:
            done = True
            info["end"] = "timeout"
        elif army_spent and self.rc_dead:
            done = True
            info["end"] = "army wiped"

        if done:
            st = stars(self.buildings, self.th_destroyed)
            if pct >= 0.50:
                reward += C.R_STAR_1
            if self.th_destroyed:
                reward += C.R_STAR_2
            if pct >= 0.999:
                reward += C.R_STAR_3
            info.update(stars=st, destruction=pct, th=self.th_destroyed,
                        steps=self.steps, **self.stats)

        if self.record:
            self._record()
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    # Royal Champion phase
    # ------------------------------------------------------------------
    def _run_rc(self) -> float:
        reward = 0.0
        rx, ry = self.rc_pos
        tx = int(np.clip(rx, 0, C.GRID_SIZE - 1))
        ty = int(np.clip(ry, 0, C.GRID_SIZE - 1))

        self.fox_invis = max(0.0, self.fox_invis - C.DT)
        self.fox_cd = max(0.0, self.fox_cd - C.DT)
        if (C.SPIRIT_FOX and self.fox_invis <= 0 and self.fox_cd <= 0
                and self.invis[ty, tx] <= 0 and self._gt()[ty, tx] > 0):
            self.fox_invis = C.FOX_INVIS_DURATION
            self.fox_cd = C.FOX_COOLDOWN + C.FOX_INVIS_DURATION
        invisible = self.invis[ty, tx] > 0 or self.fox_invis > 0

        if self.rc_hp < C.RC_MAX_HP:
            self.rc_hp = min(C.RC_MAX_HP, self.rc_hp + C.ELECTRO_HEAL_PER_SEC * C.DT)

        # Electro Boots aura
        aura = C.ELECTRO_DPS * C.DT
        for b in self.buildings:
            if not b.is_dead and b.distance_to(rx, ry) <= C.ELECTRO_RADIUS:
                reward += self._hit_building(b, aura, by_rc=True)

        # nearest DEFENSE, skipping anything under invisibility
        tgt = self._nearest_defense(rx, ry)
        if tgt is not None:
            if tgt.distance_to(rx, ry) <= C.RC_RANGE:
                reward += self._hit_building(tgt, C.RC_DPS * C.DT * self._rc_atk(),
                                             by_rc=True)
            else:
                self._move_rc(tgt.cx, tgt.cy)

        if not invisible:
            reward += self._rc_incoming(rx, ry)
        else:
            for b in self.buildings:
                b.lock_time = 0.0

        if self.rc_hp <= 0:
            self.rc_dead = True
            reward += C.R_DEATH
        return reward

    def _rc_atk(self) -> float:
        m = 1.0
        if self.tower_poison > 0:
            m *= C.SPELL_TOWER_SLOW_ATTACK
        if self.poison_timer > 0:
            m *= C.TH_POISON_SLOW
        return m

    def _move_rc(self, tx: float, ty: float) -> None:
        rx, ry = self.rc_pos
        ang = math.atan2(ty - ry, tx - rx)
        m = 1.0
        if self.tower_poison > 0:
            m *= C.SPELL_TOWER_SLOW_MOVE
        if self.poison_timer > 0:
            m *= C.TH_POISON_SLOW
        sp = C.RC_SPEED * C.DT * m
        self.rc_pos[0] = float(np.clip(rx + math.cos(ang) * sp, 0, C.GRID_SIZE - 1))
        self.rc_pos[1] = float(np.clip(ry + math.sin(ang) * sp, 0, C.GRID_SIZE - 1))

    def _nearest_defense(self, rx: float, ry: float) -> Optional[Building]:
        best = None
        bd = 1e9
        any_b = None
        ad = 1e9
        for b in self.buildings:
            if b.is_dead or b.cat == CAT_WALL:
                continue
            bx = int(np.clip(b.cx, 0, C.GRID_SIZE - 1))
            by = int(np.clip(b.cy, 0, C.GRID_SIZE - 1))
            if self.invis[by, bx] > 0:
                continue
            d = b.distance_to(rx, ry)
            if b.is_defense and d < bd:
                bd, best = d, b
            if d < ad:
                ad, any_b = d, b
        return best or any_b

    def _seeking_shield(self) -> float:
        rx, ry = self.rc_pos
        defs = sorted([b for b in self.buildings if not b.is_dead and b.is_defense],
                      key=lambda b: b.distance_to(rx, ry))
        others = sorted([b for b in self.buildings
                         if not b.is_dead and not b.is_defense and b.cat != CAT_WALL],
                        key=lambda b: b.distance_to(rx, ry))
        r = 0.0
        for b in (defs + others)[:C.SHIELD_TARGETS]:
            r += self._hit_building(b, C.SHIELD_DAMAGE, by_rc=True)
        self.rc_hp = min(C.RC_MAX_HP, self.rc_hp + C.SHIELD_HEAL)
        return r

    def _rc_incoming(self, rx: float, ry: float) -> float:
        total = 0.0
        for b in self.buildings:
            if (b.is_dead or not b.is_defense or not b.hits_ground or b.dps <= 0
                    or (b.name == "Eagle Artillery" and not self.eagle_active)):
                continue
            d = b.distance_to(rx, ry)
            if d > b.rng or d < b.min_range:
                b.lock_time = 0.0
                continue
            if b.name == "Inferno Tower":
                b.lock_time += C.DT
                dps = C.INFERNO_RAMP[0][1]
                for t, v in C.INFERNO_RAMP:
                    if b.lock_time >= t:
                        dps = v
                total += dps * C.DT
            elif b.name == "Monolith":
                total += (b.dps + (C.MONOLITH_PCT_DAMAGE * C.RC_MAX_HP) / 1.5) * C.DT
            elif b.cat == CAT_TOWN_HALL:
                if self.giga_active:
                    total += (b.dps / 4.0) * C.DT
            else:
                total += b.dps * C.DT
        if total > 0:
            self.rc_hp -= total
            return C.R_DAMAGE_TAKEN * total
        return 0.0

    # ------------------------------------------------------------------
    # Army phase
    # ------------------------------------------------------------------
    def _run_army(self) -> float:
        reward = 0.0
        alive = [u for u in self.units if not u.is_dead]
        for u in alive:
            if u.mute_timer > 0:
                continue
            # Dragons and Baby Dragons go for the NEAREST BUILDING -- no
            # preference at all. That is why funnelling exists.
            b = self.by_uid.get(u.target_uid)
            if b is None or b.is_dead:
                # the Stone Slammer targets DEFENSES ONLY until none remain
                b = (self._nearest_defense_any(u.x, u.y) if u.defenses_only
                     else self._nearest_building(u.x, u.y))
                u.target_uid = None if b is None else b.uid
            if b is None:
                continue
            _, mult, spd_mult = is_enraged(u, alive)
            if b.distance_to(u.x, u.y) <= u.rng:
                reward += self._hit_building(b, u.dps * C.DT * mult)
                if u.splash > 0:
                    for o in self.buildings:
                        if o is b or o.is_dead:
                            continue
                        if o.distance_to(b.cx, b.cy) <= u.splash + 1.0:
                            reward += self._hit_building(o, u.dps * C.DT * mult * 0.35)
            else:
                sp = u.speed * C.DT * spd_mult
                ang = math.atan2(b.cy - u.y, b.cx - u.x)
                u.x = float(np.clip(u.x + math.cos(ang) * sp, 0, C.GRID_SIZE - 1))
                u.y = float(np.clip(u.y + math.sin(ang) * sp, 0, C.GRID_SIZE - 1))
        return reward

    def _giant_arrow(self) -> float:
        """Fire the Giant Arrow.

        It travels the straight line that passes through the Queen and her
        CURRENT TARGET, pierces everything on the way with no falloff, and does
        double damage to Air Defenses -- 3,000, against 1,750 hitpoints. So
        every Air Defense on that line dies at once.

        The agent does not aim the arrow. It aims the QUEEN, by choosing where
        to drop her: her nearest building becomes the gunsight, and whatever is
        collinear behind it dies.
        """
        q = next((u for u in self.units
                  if u.kind == "Archer Queen" and not u.is_dead), None)
        if q is None:
            return 0.0
        tgt = self.by_uid.get(q.target_uid) or self._nearest_building(q.x, q.y)
        if tgt is None:
            return 0.0
        dx, dy = tgt.cx - q.x, tgt.cy - q.y
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return 0.0
        dx, dy = dx / n, dy / n

        reward = 0.0
        ads = 0
        hits = 0
        for b in self.buildings:
            if b.is_dead:
                continue
            rx, ry = b.cx - q.x, b.cy - q.y
            along = rx * dx + ry * dy
            perp = abs(rx * dy - ry * dx)
            if perp > C.GIANT_ARROW_WIDTH or along <= 0 or along > C.GIANT_ARROW_RANGE:
                continue
            dmg = C.GIANT_ARROW_DAMAGE
            if b.name in ("Air Defense", "Air Sweeper"):
                dmg *= C.GIANT_ARROW_AD_MULT
                ads += 1
            hits += 1
            reward += self._hit_building(b, dmg)
        self.stats["arrow_ads"] += ads
        self.stats["arrow_hits"] += hits
        return reward

    def _nearest_building(self, x: float, y: float) -> Optional[Building]:
        best, bd = None, 1e9
        for b in self.buildings:
            if b.is_dead or b.cat == CAT_WALL:
                continue
            d = b.distance_to(x, y)
            if d < bd:
                bd, best = d, b
        return best

    def _nearest_defense_any(self, x: float, y: float) -> Optional[Building]:
        best, bd, fb, fd = None, 1e9, None, 1e9
        for b in self.buildings:
            if b.is_dead or b.cat == CAT_WALL:
                continue
            d = b.distance_to(x, y)
            if b.is_defense and d < bd:
                bd, best = d, b
            if d < fd:
                fd, fb = d, b
        return best or fb

    def _defenses_fire(self) -> float:
        """Every air-capable defense shoots the nearest flying unit in range.

        The Air Defense is the reason the Champion went in first: 540 DPS each,
        and four of them will delete the stack in seconds if they are still up.
        """
        reward = 0.0
        alive = [u for u in self.units if not u.is_dead]
        if not alive:
            return 0.0
        for b in self.buildings:
            if b.is_dead or not b.is_defense or b.dps <= 0:
                continue
            if b.name in ("Air Sweeper",):
                continue
            if b.name == "Eagle Artillery" and not self.eagle_active:
                continue
            if b.cat == CAT_TOWN_HALL and not self.giga_active:
                continue
            # ground-only defenses cannot touch flying units
            if b.name in ("Cannon", "Mortar", "Bomb Tower"):
                continue
            tgt, bd = None, 1e9
            for u in alive:
                d = b.distance_to(u.x, u.y)
                if d <= b.rng and d >= b.min_range and d < bd:
                    bd, tgt = d, u
            if tgt is None:
                continue
            dps = b.dps
            if b.name == "Inferno Tower":
                b.lock_time += C.DT
                dps = C.INFERNO_RAMP[0][1]
                for t, v in C.INFERNO_RAMP:
                    if b.lock_time >= t:
                        dps = v
            elif b.name == "Monolith":
                dps = b.dps + (C.MONOLITH_PCT_DAMAGE * tgt.max_hp) / 1.5
            elif b.cat == CAT_TOWN_HALL:
                dps = b.dps / 4.0
            reward += self._hurt_unit(tgt, dps * C.DT)
        return reward

    def _air_traps(self) -> float:
        reward = 0.0
        for t in self.air_traps:
            if t["fired"]:
                continue
            for u in self.units:
                if u.is_dead or not u.flying:
                    continue
                if math.hypot(u.x - t["x"], u.y - t["y"]) <= t["trig"]:
                    t["fired"] = True
                    self.stats["traps_hit"] += 1
                    if t["rad"] > 0:
                        for o in self.units:
                            if (not o.is_dead and o.flying
                                    and math.hypot(o.x - t["x"], o.y - t["y"]) <= t["rad"]):
                                reward += self._hurt_unit(o, t["dmg"])
                    else:
                        reward += self._hurt_unit(u, t["dmg"])
                    break
        return reward

    def _hurt_unit(self, u: Unit, dmg: float) -> float:
        # Eternal Tome blocks DAMAGE only -- not sweeper knockback, not slows
        if self.tome_timer > 0:
            w = next((x for x in self.units
                      if x.kind == "Grand Warden" and not x.is_dead), None)
            if w is not None and u.distance_to(w.x, w.y) <= C.TOME_RADIUS:
                return 0.0
        if u.kind == "Dragon Duke":
            en, _, _ = is_enraged(u, [x for x in self.units if not x.is_dead])
            if en:
                dmg *= (1.0 - C.DUKE_TRAP_REDUCTION)
        u.hp -= dmg
        if u.hp <= 0:
            u.is_dead = True
            self.stats["units_lost"] += 1
            r = C.R_UNIT_LOST
            if u.kind == "Dragon Duke":          # Fire Heart death explosion
                for b in self.buildings:
                    if not b.is_dead and b.distance_to(u.x, u.y) <= 4.0:
                        r += self._hit_building(b, C.DUKE_DEATH_EXPLOSION)
            elif u.kind == "Stone Slammer":
                for b in self.buildings:
                    if not b.is_dead and b.distance_to(u.x, u.y) <= 3.0:
                        r += self._hit_building(b, C.STONE_SLAMMER["death_damage"])
            return r
        return 0.0

    # ------------------------------------------------------------------
    def _hit_building(self, b: Building, dmg: float, by_rc: bool = False) -> float:
        if b.is_dead or dmg <= 0:
            return 0.0
        if b.cat == CAT_TOWN_HALL:
            self.giga_active = True
        dmg = min(dmg, b.hp)
        b.hp -= dmg
        val = max(0.25, C.target_value(b.name, b.is_defense))
        reward = C.R_DAMAGE_DEALT * dmg * val
        if b.hp <= 0:
            b.is_dead = True
            b.hp = 0.0
            self.grid[b.y:b.y + b.h, b.x:b.x + b.w] = CAT_EMPTY
            self._dirty = True
            reward += C.target_value(b.name, b.is_defense)
            if b.is_defense:
                self.stats["defense_kills"] += 1
            if b.name in ("Air Defense", "Air Sweeper"):
                self.stats["ad_killed"] += 1
            if C.target_value(b.name, b.is_defense) >= 2.0:
                self.stats["key_kills"] += 1
            if by_rc:
                self.stats["rc_kills"] += 1
            if b.cat == CAT_TOWN_HALL:
                self.th_destroyed = True
                self.poison_timer = C.TH_POISON_DURATION
            if b.name in ("Air Defense", "Air Sweeper"):
                self._arrow_map = None
            if b.name == "Air Sweeper":
                self._sweeper_cov = sweeper_coverage_map(
                    [s for s, bb in zip(self.sweepers, self.sweeper_buildings)
                     if not bb.is_dead])
        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _gt(self) -> np.ndarray:
        """Ground threat -- what can hurt the Royal Champion."""
        if self._ground_threat is not None and not self._dirty:
            return self._ground_threat
        self._rebuild_threat()
        return self._ground_threat

    def _at(self) -> np.ndarray:
        """AIR threat -- what can hurt the Dragons. Completely different map:
        Air Defenses dominate it and Cannons are absent from it."""
        if self._air_threat is not None and not self._dirty:
            return self._air_threat
        self._rebuild_threat()
        return self._air_threat

    def _rebuild_threat(self) -> None:
        g = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        a = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        for b in self.buildings:
            if b.is_dead or not b.is_defense or b.dps <= 0:
                continue
            if b.name == "Eagle Artillery" and not self.eagle_active:
                continue
            dx = np.maximum.reduce([b.x - self._xx, np.zeros_like(self._xx),
                                    self._xx - (b.x + b.w)])
            dy = np.maximum.reduce([b.y - self._yy, np.zeros_like(self._yy),
                                    self._yy - (b.y + b.h)])
            d = np.hypot(dx, dy)
            band = (d <= b.rng) & (d >= b.min_range)
            dps = 2300.0 if b.name == "Inferno Tower" else b.dps
            if b.hits_ground and b.name not in ("Air Defense", "Air Sweeper"):
                g += np.where(band, dps, 0.0)
            if b.name not in ("Cannon", "Mortar", "Bomb Tower", "Air Sweeper"):
                a += np.where(band, dps, 0.0)
        self._ground_threat, self._air_threat = g, a
        self._dirty = False

    def _static_ch(self) -> np.ndarray:
        if self._static is not None and not self._dirty:
            return self._static
        ch = np.zeros((8, C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        m = {CAT_TOWN_HALL: 0, CAT_AIR_DEFENSE: 1, CAT_HIGH_DEFENSE: 2,
             CAT_DEFENSE: 3, CAT_NON_DEFENSE: 4, CAT_WALL: 5}
        for b in self.buildings:
            if b.is_dead:
                continue
            c = m.get(b.cat)
            if c is None:
                continue
            sl = (slice(b.y, b.y + b.h), slice(b.x, b.x + b.w))
            ch[(c,) + sl] = 1.0
            ch[(6,) + sl] = b.hp / b.max_hp
            ch[(7,) + sl] = min(1.0, C.target_value(b.name, b.is_defense)
                                / C.TARGET_VALUE_NORM)
        self._static = ch
        return ch

    def _obs(self):
        sp = np.zeros((N_CH, C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        st = self._static_ch()
        sp[0:7] = st[0:7]
        sp[13] = st[7]                                           # target priority
        sp[7] = np.clip(self._gt() / C.THREAT_NORM, 0, 1)        # threat to the RC
        sp[8] = np.clip(self._at() / C.THREAT_NORM, 0, 1)        # threat to dragons
        sp[9] = self.invis / C.SPELL_DURATION
        if self.rc_deployed and not self.rc_dead:
            d2 = (self._xx - self.rc_pos[0]) ** 2 + (self._yy - self.rc_pos[1]) ** 2
            sp[10] = np.exp(-d2 / 4.0)
        for u in self.units:                                     # friendly air
            if u.is_dead:
                continue
            xi = int(np.clip(u.x, 0, C.GRID_SIZE - 1))
            yi = int(np.clip(u.y, 0, C.GRID_SIZE - 1))
            sp[11, yi, xi] += 0.25
        np.clip(sp[11], 0, 1, out=sp[11])
        # THE map for choosing which side to bring the dragons in from
        sp[12] = np.clip(self._sweeper_cov / 2.0, 0, 1)

        # GIANT ARROW VALUE MAP: for every legal deploy cell, how many Air
        # Defenses an arrow fired from there would pierce. This turns "line the
        # Queen up with the Air Defenses" from something the agent has to infer
        # from raw geometry into something it can simply read off the map.
        if getattr(self, "_arrow_map", None) is None:
            self._arrow_map = arrow_value_map(self._deploy_mask(),
                                              self.buildings, C.GRID_SIZE)
        am = np.repeat(np.repeat(self._arrow_map, C.ACTION_STRIDE, axis=0),
                       C.ACTION_STRIDE, axis=1)
        sp[14] = np.clip(am / 3.0, 0, 1)

        n_units = len(C.DEPLOY_ORDER)
        alive = sum(1 for u in self.units if not u.is_dead)
        sc = np.array([
            self.rc_hp / C.RC_MAX_HP,
            self.spells_left / max(1, self.max_spells),
            float(self.shield_left > 0),
            float(self.tome_left > 0),
            self.steps / C.MAX_STEPS,
            float(self.th_destroyed),
            destruction_percent(self.buildings),
            sum(1 for b in self.buildings if b.is_defense and not b.is_dead)
            / max(1, sum(1 for b in self.buildings if b.is_defense)),
            1.0 - len(self.queue) / n_units,
            alive / n_units,
            float(not self.rc_deployed),
            float(self.rc_dead),
        ], dtype=np.float32)
        return sp, sc

    def _record(self) -> None:
        self.frames.append(dict(
            grid=self.grid.copy(), invis=self.invis.copy(),
            pos=tuple(self.rc_pos), hp=self.rc_hp, spells=self.spells_left,
            th=self.th_destroyed, rc_dead=self.rc_dead,
            units=[(u.x, u.y, u.kind) for u in self.units if not u.is_dead],
            dest=destruction_percent(self.buildings)))
