"""
The Royal Champion walk environment.

What the agent actually controls
--------------------------------
1. WHERE TO DEPLOY HER  (the first action of every episode)
2. WHEN AND WHERE TO CAST INVISIBILITY  (8-10 spells, 4 tile radius, 4.25s)
3. WHEN TO POP SEEKING SHIELD  (once, 4 targets, big heal)

Everything else -- her pathing, her target choice, her attacks -- is the game's
hard-coded hero AI, exactly as in the real game.

The mechanics that were wrong before, and are right now
-------------------------------------------------------
* She targets the NEAREST DEFENSE, not "Town Hall then Air Defenses". The Town
  Hall is a defense (Giga Inferno), so deploying her near it means she walks
  straight at it. This is the whole reason a Town Hall snipe is possible.
* Air Defenses and Air Sweepers CANNOT hit her. She is a ground unit.
* The Eagle Artillery never activates: it needs 200 housing space deployed and
  a solo hero plus spells is ~65.
* Inferno Towers ramp up (100 -> 230 -> 2300 DPS) and RESET when they lose
  their target. Turning invisible resets them. This is why timing matters.
* Range is measured to the building's EDGE, not its centre.
* Mortars (4), Scattershots (3) and the Eagle (7) have minimum ranges -- there
  are safe pockets right up against them.
* An invisible BUILDING is skipped by her targeting but still shoots.
* Killing something with the Electro Boots aura pays the same reward as killing
  it with her spear. (The old code computed the reward and threw it away.)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .defenders import (Defender, Trap, make_cc_troops,
                        make_defending_queen)
from .base import (Building, CAT_AIR_DEFENSE, CAT_DEFENSE, CAT_EMPTY,
                   CAT_HIGH_DEFENSE, CAT_NON_DEFENSE, CAT_TOWN_HALL, CAT_WALL,
                   destruction_percent, generate_base, stars)


class RCWalkEnv:
    """Single Royal Champion attacking a TH15 base with Invisibility Spells."""

    def __init__(self, defense_frac: float = 1.0, max_spells: int = C.MAX_SPELLS,
                 seed: Optional[int] = None, record: bool = False,
                 traps: bool = False, cc: bool = False, hero: bool = False):
        self.defense_frac = defense_frac
        self.max_spells = max_spells
        self.use_traps = traps
        self.use_cc = cc
        self.use_hero = hero
        self.rng = random.Random(seed)
        self.record = record
        self._tile_idx = np.indices((C.GRID_SIZE, C.GRID_SIZE))   # (2, H, W) -> y, x
        self._yy = self._tile_idx[0].astype(np.float32)
        self._xx = self._tile_idx[1].astype(np.float32)
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def reset(self, defense_frac: Optional[float] = None,
              max_spells: Optional[int] = None, traps: Optional[bool] = None,
              cc: Optional[bool] = None, hero: Optional[bool] = None):
        if defense_frac is not None:
            self.defense_frac = defense_frac
        if max_spells is not None:
            self.max_spells = max_spells
        if traps is not None:
            self.use_traps = traps
        if cc is not None:
            self.use_cc = cc
        if hero is not None:
            self.use_hero = hero

        (self.grid, self.buildings, self.traps,
         self.cc_pos, self.altar_pos) = generate_base(
            self.defense_frac, seed=self.rng.randrange(1 << 30),
            traps=self.use_traps, cc=self.use_cc, hero=self.use_hero)
        self.defenders: List[Defender] = []
        self.cc_released = False
        self.hero_released = False
        self.freeze_timer = 0.0
        self.hh_poison = 0.0
        self.town_hall = next(b for b in self.buildings if b.cat == CAT_TOWN_HALL)

        self.rc_hp = float(C.RC_MAX_HP)
        self.spells_left = int(self.max_spells)
        self.ability_left = C.SHIELD_USES
        self.steps = 0
        self.time = 0.0
        self.th_destroyed = False
        self.poison_timer = 0.0
        self.giga_active = False        # Giga Inferno sleeps until provoked
        self.tower_poison = 0.0         # Spell Tower poison remaining
        self.tower_cd = {}              # per-Spell-Tower re-arm timer
        self.fox_invis = 0.0            # Spirit Fox cloak remaining
        self.fox_cd = 0.0
        self.deployed = False
        self.rc_pos = [C.GRID_SIZE / 2.0, C.GRID_SIZE / 2.0]   # placeholder
        self.invis = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        self.current_target: Optional[Building] = None
        self.slow_timer = 0.0

        # Eagle Artillery activation: housing deployed = hero + spells.
        housing = C.RC_HOUSING + self.max_spells * C.SPELL_HOUSING_COST
        self.eagle_active = housing >= C.EAGLE_ACTIVATION_HOUSING

        self._dirty_static = True
        self._dirty_threat = True
        self._static_cache: Optional[np.ndarray] = None
        self._threat_cache: Optional[np.ndarray] = None

        self.frames: List[Dict] = []
        self.stats = dict(kills=0, high_kills=0, spells_used=0, damage_dealt=0.0,
                          damage_taken=0.0, ability_used=0, traps_hit=0,
                          defenders_killed=0, key_kills=0, defense_kills=0)
        return self._obs()

    # ------------------------------------------------------------------
    # Action space helpers
    # ------------------------------------------------------------------
    def legal_actions(self) -> np.ndarray:
        """Boolean mask over the action space.

        Masking beats penalising. The old code fired a -10 'empty click' penalty
        when the agent cast with no spells left; now that action simply cannot
        be selected, so no capacity is wasted learning not to do it.
        """
        mask = np.zeros(C.N_ACTIONS, dtype=bool)
        if not self.deployed:
            # deploy phase: only tile actions, and only perimeter tiles
            mask[C.ACTION_TILE_OFFSET:] = self._deploy_mask()
            return mask
        mask[C.ACTION_WAIT] = True
        if self.ability_left > 0:
            mask[C.ACTION_ABILITY] = True
        if self.spells_left > 0:
            mask[C.ACTION_TILE_OFFSET:] = True
        return mask

    def _deploy_mask(self) -> np.ndarray:
        """Real rule: you may deploy on any tile not occupied by a building.
        On a tight war base that means the border -- but if the layout leaves a
        gap she can be dropped straight into it, which is exactly the
        'deploy her next to the Town Hall' idea.

        This is the second half of the fix. Before, she spawned on a random map
        edge and the agent had no say in it.

        Clash also enforces a one-tile buffer: you may deploy on the *second*
        tile away from any structure, walls included. So the occupancy mask is
        dilated by one tile before it is inverted.
        """
        occupied = self.grid != CAT_EMPTY
        blocked = occupied.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                blocked |= np.roll(np.roll(occupied, dy, axis=0), dx, axis=1)
        s = C.ACTION_STRIDE
        blocks = blocked.reshape(C.ACTION_GRID, s, C.ACTION_GRID, s)
        legal = ~blocks.any(axis=(1, 3))
        if not legal.any():           # degenerate layout: fall back to the border
            legal[0, :] = legal[-1, :] = legal[:, 0] = legal[:, -1] = True
        return legal.reshape(-1)

    @staticmethod
    def action_to_tile(action: int) -> Tuple[int, int]:
        idx = action - C.ACTION_TILE_OFFSET
        ay, ax = divmod(idx, C.ACTION_GRID)
        # centre of the coarse cell
        x = ax * C.ACTION_STRIDE + C.ACTION_STRIDE // 2
        y = ay * C.ACTION_STRIDE + C.ACTION_STRIDE // 2
        return x, y

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int):
        info: Dict = {}
        reward = 0.0

        # ---------- deployment phase ----------
        if not self.deployed:
            x, y = self.action_to_tile(action) if action >= C.ACTION_TILE_OFFSET else (2, 2)
            x = int(np.clip(x, 0, C.GRID_SIZE - 1))
            y = int(np.clip(y, 0, C.GRID_SIZE - 1))
            self.rc_pos = [float(x), float(y)]
            self.deployed = True
            if self.record:
                self._record()
            return self._obs(), 0.0, False, info

        self.steps += 1
        self.time += C.DT
        reward += C.R_PER_STEP

        # ---------- 1. agent action ----------
        if action == C.ACTION_ABILITY and self.ability_left > 0:
            reward += C.R_ABILITY_USE + self._seeking_shield()
            self.ability_left -= 1
            self.stats["ability_used"] += 1
        elif action >= C.ACTION_TILE_OFFSET and self.spells_left > 0:
            tx, ty = self.action_to_tile(action)
            self._cast_invisibility(tx, ty)
            self.spells_left -= 1
            self.stats["spells_used"] += 1
            reward += C.R_SPELL_CAST

        # ---------- 2. spell timers ----------
        self.freeze_timer = max(0.0, self.freeze_timer - C.DT)
        self.hh_poison = max(0.0, self.hh_poison - C.DT)
        np.subtract(self.invis, C.DT, out=self.invis)
        np.clip(self.invis, 0.0, None, out=self.invis)

        rx, ry = self.rc_pos
        tile_x = int(np.clip(rx, 0, C.GRID_SIZE - 1))
        tile_y = int(np.clip(ry, 0, C.GRID_SIZE - 1))

        # Spirit Fox: cloaks her on its own cooldown. It only covers part of
        # the cycle, so the spells fill the gaps rather than replacing it.
        self.fox_invis = max(0.0, self.fox_invis - C.DT)
        self.fox_cd = max(0.0, self.fox_cd - C.DT)
        if (C.SPIRIT_FOX and self.fox_invis <= 0 and self.fox_cd <= 0
                and self.invis[tile_y, tile_x] <= 0
                and self._threat_channel()[tile_y, tile_x] > 0):
            self.fox_invis = C.FOX_INVIS_DURATION
            self.fox_cd = C.FOX_COOLDOWN + C.FOX_INVIS_DURATION

        hero_invisible = (self.invis[tile_y, tile_x] > 0.0) or (self.fox_invis > 0.0)

        # ---------- 3. passive self-heal ----------
        if self.rc_hp < C.RC_MAX_HP:
            self.rc_hp = min(C.RC_MAX_HP, self.rc_hp + C.ELECTRO_HEAL_PER_SEC * C.DT)

        # ---------- 4. Electro Boots aura ----------
        # Damages EVERYTHING in radius, and kills here are credited properly.
        aura_dmg = C.ELECTRO_DPS * C.DT
        for b in self.buildings:
            if b.is_dead:
                continue
            if b.distance_to(rx, ry) <= C.ELECTRO_RADIUS:
                reward += self._damage_building(b, aura_dmg)

        for d in self.defenders:
            if not d.is_dead and d.distance_to(rx, ry) <= C.ELECTRO_RADIUS:
                reward += self._damage_defender(d, aura_dmg)

        # ---------- 4b. traps, Clan Castle, defending hero ----------
        reward += self._trigger_traps(rx, ry)
        self._release_defenders(rx, ry)
        reward += self._run_defenders(rx, ry, hero_invisible)

        # ---------- 5. hero targeting + attack/move ----------
        # She ABANDONS her building target for Clan Castle troops, defending
        # heroes or Skeleton Trap skeletons, then resumes on the NEAREST defense
        # -- not the one she was on. That tempo loss is the real cost of a CC.
        live_def = [d for d in self.defenders if not d.is_dead]
        if live_def and not hero_invisible:
            # aggro: troops outrank every building
            nearest = min(live_def, key=lambda d: d.distance_to(rx, ry))
            self.current_target = None
            if nearest.distance_to(rx, ry) <= C.RC_RANGE:
                reward += self._damage_defender(
                    nearest, C.RC_DPS * C.DT * self._attack_mult())
            else:
                self._move_toward(nearest.x, nearest.y)
        else:
            target = self._best_target()
            self.current_target = target
            if target is not None:
                dist = target.distance_to(rx, ry)
                if dist <= C.RC_RANGE:
                    reward += self._damage_building(
                        target, C.RC_DPS * C.DT * self._attack_mult())
                else:
                    self._move_toward(target.cx, target.cy)
            # no target at all means the base is cleared; handled below

        # ---------- 6. incoming damage ----------
        if not hero_invisible:
            reward += self._apply_incoming_damage(rx, ry)
        else:
            # invisibility breaks every Inferno lock -- the key timing mechanic
            for b in self.buildings:
                b.lock_time = 0.0

        # ---------- 6b. Spell Tower poison ----------
        self.tower_poison = max(0.0, self.tower_poison - C.DT)
        for b in self.buildings:
            if b.is_dead or b.name != "Spell Tower":
                continue
            cd = self.tower_cd.get(b.uid, 0.0)
            if cd > 0:
                self.tower_cd[b.uid] = cd - C.DT
                continue
            if b.distance_to(rx, ry) <= C.SPELL_TOWER_TRIGGER and not hero_invisible:
                self.tower_poison = C.SPELL_TOWER_DURATION
                self.tower_cd[b.uid] = C.SPELL_TOWER_DURATION + C.SPELL_TOWER_DELAY
        if self.tower_poison > 0 and not hero_invisible:
            dmg = C.SPELL_TOWER_DPS * C.DT
            self.rc_hp -= dmg
            self.stats["damage_taken"] += dmg
            reward += C.R_DAMAGE_TAKEN * dmg

        # ---------- 7. town hall poison ----------
        if self.poison_timer > 0:
            self.poison_timer -= C.DT
            self.slow_timer = max(0.0, self.slow_timer - C.DT)
            if not hero_invisible:
                d = self.town_hall.distance_to(*self.rc_pos)
                if d <= C.TH_DEATH_RADIUS:
                    dmg = C.TH_POISON_DPS * C.DT
                    self.rc_hp -= dmg
                    self.stats["damage_taken"] += dmg
                    reward += C.R_DAMAGE_TAKEN * dmg
                    self.slow_timer = 1.0

        # ---------- 8. termination ----------
        done = False
        alive_any = any(not b.is_dead for b in self.buildings)
        pct = destruction_percent(self.buildings)
        if pct >= C.TH_ACTIVATE_DESTRUCTION:
            self.giga_active = True

        if self.rc_hp <= 0:
            self.rc_hp = 0.0
            reward += C.R_DEATH
            done = True
            info["end"] = "died"
        elif not alive_any:
            done = True
            info["end"] = "cleared"
        elif self.steps >= C.MAX_STEPS:
            reward += C.R_TIMEOUT
            done = True
            info["end"] = "timeout"

        if done:
            # star bonuses paid once, at the end
            if pct >= 0.50:
                reward += C.R_STAR_50_PCT
            if pct >= 0.999:
                reward += C.R_STAR_100_PCT
            info.update(
                th_destroyed=self.th_destroyed,
                destruction=pct,
                stars=stars(self.buildings, self.th_destroyed),
                steps=self.steps,
                game_time=self.time,
                hp_left=self.rc_hp / C.RC_MAX_HP,
                spells_used=self.stats["spells_used"],
                **{k: v for k, v in self.stats.items() if k != "spells_used"},
            )

        if self.record:
            self._record()
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    # Mechanics
    # ------------------------------------------------------------------
    def _best_target(self) -> Optional[Building]:
        """The real rule: nearest DEFENSE (the Town Hall counts). Only when no
        defense is left does she attack anything else. Buildings hidden under an
        Invisibility Spell are skipped entirely."""
        best_def = None
        best_def_d = 1e9
        best_any = None
        best_any_d = 1e9
        rx, ry = self.rc_pos
        for b in self.buildings:
            if b.is_dead or b.cat == CAT_WALL:
                continue
            bx = int(np.clip(b.cx, 0, C.GRID_SIZE - 1))
            by = int(np.clip(b.cy, 0, C.GRID_SIZE - 1))
            if self.invis[by, bx] > 0.0:
                continue                      # invisible -> she ignores it
            d = b.distance_to(rx, ry)
            if b.is_defense:
                if d < best_def_d:
                    best_def_d, best_def = d, b
            if d < best_any_d:
                best_any_d, best_any = d, b
        return best_def if best_def is not None else best_any

    def _move_mult(self) -> float:
        if self.freeze_timer > 0:      # Ice Golem death freeze
            return 0.0
        m = 1.0
        if self.slow_timer > 0:
            m *= C.TH_POISON_SLOW
        if self.tower_poison > 0:
            m *= C.SPELL_TOWER_SLOW_MOVE
        if self.hh_poison > 0:
            m *= C.HEADHUNTER_SLOW_MOVE
        return m

    def _attack_mult(self) -> float:
        if self.freeze_timer > 0:
            return 0.0
        m = 1.0
        if self.slow_timer > 0:
            m *= C.TH_POISON_SLOW
        if self.tower_poison > 0:
            m *= C.SPELL_TOWER_SLOW_ATTACK
        if self.hh_poison > 0:
            m *= C.HEADHUNTER_SLOW_ATTACK
        return m

    def _move_toward(self, tx: float, ty: float) -> None:
        rx, ry = self.rc_pos
        ang = math.atan2(ty - ry, tx - rx)
        speed = C.RC_SPEED * C.DT * self._move_mult()
        self.rc_pos[0] = float(np.clip(rx + math.cos(ang) * speed, 0, C.GRID_SIZE - 1))
        self.rc_pos[1] = float(np.clip(ry + math.sin(ang) * speed, 0, C.GRID_SIZE - 1))

    def _damage_building(self, b: Building, dmg: float) -> float:
        """Apply damage and return the reward it earns. Used by BOTH the spear
        and the Electro aura -- the old code only credited the spear."""
        if b.is_dead or dmg <= 0:
            return 0.0
        dmg = min(dmg, b.hp)
        b.hp -= dmg
        self.stats["damage_dealt"] += dmg
        if b.cat == CAT_TOWN_HALL:
            self.giga_active = True      # touching it wakes the Giga Inferno
        # Dense shaping is weighted by what the target is WORTH, not by what it
        # threatens. Chipping an Air Defense pays; chipping a Cannon barely does.
        mult = max(0.25, C.target_value(b.name, b.is_defense))
        reward = C.R_DAMAGE_DEALT * dmg * mult
        if b.hp <= 0:
            reward += self._kill_building(b)
        return reward

    def _kill_building(self, b: Building) -> float:
        b.is_dead = True
        b.hp = 0.0
        self.grid[b.y:b.y + b.h, b.x:b.x + b.w] = CAT_EMPTY   # full footprint
        self._dirty_static = True
        if b.is_defense:
            self._dirty_threat = True
        self.stats["kills"] += 1

        if b.cat == CAT_TOWN_HALL:
            self.th_destroyed = True
            self.poison_timer = C.TH_POISON_DURATION
            reward = C.R_TH_DESTROYED
            # poison bomb
            if b.distance_to(*self.rc_pos) <= C.TH_DEATH_RADIUS:
                self.rc_hp -= C.TH_DEATH_DAMAGE
                self.stats["damage_taken"] += C.TH_DEATH_DAMAGE
                reward += C.R_DAMAGE_TAKEN * C.TH_DEATH_DAMAGE
            return reward
        val = C.target_value(b.name, b.is_defense)
        if val >= 2.0:
            self.stats["key_kills"] += 1
        if b.is_defense:
            self.stats["defense_kills"] += 1
        return val

    # ------------------------------------------------------------------
    # Traps and defenders
    # ------------------------------------------------------------------
    def _trigger_traps(self, rx: float, ry: float) -> float:
        """Traps fire on proximity REGARDLESS of invisibility.

        This is the one thing a perfect casting rhythm cannot protect against,
        and it is why the threat map in the observation is not the whole story:
        traps are hidden, exactly as they are to a player who has not scouted.
        """
        reward = 0.0
        for t in self.traps:
            if t.fired or t.distance_to(rx, ry) > t.trigger:
                continue
            t.fired = True
            self.stats["traps_hit"] += 1
            if t.name == "Skeleton Trap":
                # spawns skeletons, which pull her off her target -- the Electro
                # Boots aura is what makes them a nuisance instead of a disaster
                for _ in range(C.SKELETON_COUNT):
                    self.defenders.append(Defender(
                        name="Skeleton", hp=C.SKELETON_HP, max_hp=C.SKELETON_HP,
                        dps=C.SKELETON_DPS, rng=0.5, speed=3.0,
                        x=t.cx + random.uniform(-1, 1),
                        y=t.cy + random.uniform(-1, 1)))
                continue
            self.rc_hp -= t.damage
            self.stats["damage_taken"] += t.damage
            reward += C.R_DAMAGE_TAKEN * t.damage
        return reward

    def _release_defenders(self, rx: float, ry: float) -> None:
        if (self.use_cc and not self.cc_released and self.cc_pos is not None
                and math.hypot(self.cc_pos[0] - rx, self.cc_pos[1] - ry)
                <= C.CC_TRIGGER_RADIUS):
            self.cc_released = True
            self.defenders.extend(make_cc_troops(*self.cc_pos))
        if (self.use_hero and not self.hero_released and self.altar_pos is not None
                and math.hypot(self.altar_pos[0] - rx, self.altar_pos[1] - ry)
                <= C.DEFENDING_QUEEN["patrol"]):
            self.hero_released = True
            self.defenders.append(make_defending_queen(*self.altar_pos))

    def _run_defenders(self, rx: float, ry: float, hero_invisible: bool) -> float:
        """Move the defenders and apply their damage.

        While she is invisible they cannot see her, so they neither chase nor
        shoot -- invisibility breaks aggro as well as damage. That is what makes
        cloaking a viable answer to a Clan Castle, and why the timing matters.
        """
        reward = 0.0
        total = 0.0
        for d in self.defenders:
            if d.is_dead:
                continue
            d.step((rx, ry), C.DT, not hero_invisible)
            if hero_invisible:
                continue
            if d.distance_to(rx, ry) <= d.rng:
                total += d.effective_dps() * C.DT
                if d.poison_on_hit:
                    self.hh_poison = C.HEADHUNTER_POISON_DURATION
        if total > 0:
            self.rc_hp -= total
            self.stats["damage_taken"] += total
            reward += C.R_DAMAGE_TAKEN * total
        return reward

    def _damage_defender(self, d: Defender, dmg: float) -> float:
        if d.is_dead or dmg <= 0:
            return 0.0
        dmg = min(dmg, d.hp)
        d.hp -= dmg
        self.stats["damage_dealt"] += dmg
        reward = C.R_DAMAGE_DEALT * dmg
        if d.hp <= 0:
            d.is_dead = True
            self.stats["defenders_killed"] += 1
            reward += C.R_KILL_DEFENDER
            if d.freeze_on_death > 0:      # Ice Golem
                if d.distance_to(*self.rc_pos) <= d.freeze_radius:
                    self.freeze_timer = d.freeze_on_death
        return reward

    def _cast_invisibility(self, cx: int, cy: int) -> None:
        mask = ((self._xx - cx) ** 2 + (self._yy - cy) ** 2) <= C.SPELL_RADIUS ** 2
        np.maximum(self.invis, np.where(mask, C.SPELL_DURATION, 0.0), out=self.invis)

    def _seeking_shield(self) -> float:
        """Seeks up to 4 targets, prioritising defenses, regardless of distance,
        then heals her."""
        rx, ry = self.rc_pos
        defs = [b for b in self.buildings if not b.is_dead and b.is_defense]
        others = [b for b in self.buildings
                  if not b.is_dead and not b.is_defense and b.cat != CAT_WALL]
        defs.sort(key=lambda b: b.distance_to(rx, ry))
        others.sort(key=lambda b: b.distance_to(rx, ry))
        troops = [d for d in self.defenders if not d.is_dead]
        troops.sort(key=lambda d: d.distance_to(rx, ry))
        picks = (defs + troops + others)[:C.SHIELD_TARGETS]
        reward = 0.0
        for b in picks:
            if isinstance(b, Defender):
                reward += self._damage_defender(b, C.SHIELD_DAMAGE)
            else:
                reward += self._damage_building(b, C.SHIELD_DAMAGE)
        self.rc_hp = min(C.RC_MAX_HP, self.rc_hp + C.SHIELD_HEAL)
        return reward

    def _apply_incoming_damage(self, rx: float, ry: float) -> float:
        total = 0.0
        for b in self.buildings:
            if b.is_dead or not b.is_defense or not b.hits_ground or b.dps <= 0:
                continue
            if b.name == "Eagle Artillery" and not self.eagle_active:
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
                # 262.5 per shot + 12% of the target's MAX hp, every 1.5s
                pct_dps = (C.MONOLITH_PCT_DAMAGE * C.RC_MAX_HP) / 1.5
                total += (b.dps + pct_dps) * C.DT
            elif b.cat == CAT_TOWN_HALL:
                if not self.giga_active:
                    continue
                # multi-target: its 300 DPS is split across up to 4 slots, so a
                # lone hero never eats the full output. This is exactly why solo
                # hero attacks on a Town Hall are viable at all.
                total += (b.dps / 4.0) * C.DT
            else:
                total += b.dps * C.DT
        if total > 0:
            self.rc_hp -= total
            self.stats["damage_taken"] += total
            return C.R_DAMAGE_TAKEN * total
        return 0.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _static_channels(self) -> np.ndarray:
        """Channels 0-6: footprints + hp fraction. Rebuilt only when something
        dies, not every tick."""
        if not self._dirty_static and self._static_cache is not None:
            return self._static_cache
        ch = np.zeros((8, C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        cat_to_ch = {CAT_TOWN_HALL: 0, CAT_AIR_DEFENSE: 1, CAT_HIGH_DEFENSE: 2,
                     CAT_DEFENSE: 3, CAT_NON_DEFENSE: 4, CAT_WALL: 5}
        for b in self.buildings:
            if b.is_dead:
                continue
            c = cat_to_ch.get(b.cat)
            if c is None:
                continue
            ch[c, b.y:b.y + b.h, b.x:b.x + b.w] = 1.0
            ch[6, b.y:b.y + b.h, b.x:b.x + b.w] = b.hp / b.max_hp
            ch[7, b.y:b.y + b.h, b.x:b.x + b.w] = min(
                1.0, C.target_value(b.name, b.is_defense) / C.TARGET_VALUE_NORM)
        self._static_cache = ch
        self._dirty_static = False
        return ch

    def _threat_channel(self) -> np.ndarray:
        """Channel 7: total incoming DPS that reaches each tile. Only counts
        defenses that can actually hit a GROUND unit -- so Air Defenses do not
        show up as danger, because they are not."""
        if not self._dirty_threat and self._threat_cache is not None:
            return self._threat_cache
        t = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        for b in self.buildings:
            if b.is_dead or not b.is_defense or not b.hits_ground or b.dps <= 0:
                continue
            if b.name == "Eagle Artillery" and not self.eagle_active:
                continue
            dx = np.maximum.reduce([b.x - self._xx, np.zeros_like(self._xx),
                                    self._xx - (b.x + b.w)])
            dy = np.maximum.reduce([b.y - self._yy, np.zeros_like(self._yy),
                                    self._yy - (b.y + b.h)])
            dist = np.hypot(dx, dy)
            dps = 2300.0 if b.name == "Inferno Tower" else b.dps
            t += np.where((dist <= b.rng) & (dist >= b.min_range), dps, 0.0)
        self._threat_cache = t
        self._dirty_threat = False
        return t

    def _obs(self) -> Tuple[np.ndarray, np.ndarray]:
        spatial = np.zeros((C.N_SPATIAL_CHANNELS, C.GRID_SIZE, C.GRID_SIZE),
                           dtype=np.float32)
        st = self._static_channels()
        spatial[0:7] = st[0:7]
        spatial[10] = st[7]              # target priority
        spatial[7] = np.clip(self._threat_channel() / C.THREAT_NORM, 0.0, 1.0)
        spatial[8] = self.invis / C.SPELL_DURATION

        # channel 9: HERO POSITION. This is the channel whose absence made the
        # old agent's task unsolvable -- it was being asked where to protect her
        # without being told where she was.
        if self.deployed:
            d2 = (self._xx - self.rc_pos[0]) ** 2 + (self._yy - self.rc_pos[1]) ** 2
            spatial[9] = np.exp(-d2 / 4.0).astype(np.float32)

        n_def = sum(1 for b in self.buildings if b.is_defense)
        alive_def = sum(1 for b in self.buildings if b.is_defense and not b.is_dead)
        scalars = np.array([
            self.rc_hp / C.RC_MAX_HP,
            self.spells_left / max(1, self.max_spells),
            float(self.ability_left > 0),
            self.steps / C.MAX_STEPS,
            float(self.th_destroyed),
            destruction_percent(self.buildings),
            alive_def / max(1, n_def),
            float(not self.deployed),
        ], dtype=np.float32)
        return spatial, scalars

    # ------------------------------------------------------------------
    def _record(self) -> None:
        self.frames.append(dict(
            grid=self.grid.copy(),
            invis=self.invis.copy(),
            pos=tuple(self.rc_pos),
            hp=self.rc_hp,
            spells=self.spells_left,
            th=self.th_destroyed,
            target=None if self.current_target is None else
                   (self.current_target.cx, self.current_target.cy),
        ))
