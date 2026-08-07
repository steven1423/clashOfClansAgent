"""
Hand-written reference policies.

These serve three jobs at once:

  * baselines to measure the trained agent against (evaluate.py)
  * demonstrations to seed the replay buffer with (train.py)
  * the guided-exploration teacher during early training

`policy_scripted_human` is a deliberately simple heuristic -- deploy nearest the
Town Hall, stay invisible whenever standing in a threatened tile, pop the shield
when low. It encodes the actual Royal Champion charge that guides describe
("cast one Invisibility Spell roughly every four seconds while she pushes"), and
nothing cleverer. If the DQN cannot beat it, the DQN is not adding value.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import config as C


def deploy_nearest_th(env) -> int:
    """Pick the legal deployment cell closest to the Town Hall.

    The user's insight, and it is correct: the point of the attack is the Town
    Hall, so the walk should start on the Town Hall's side of the base. Because
    she targets the nearest DEFENSE and the Town Hall is a defense, starting
    close to it is what puts it at the front of her queue.
    """
    mask = env.legal_actions()[C.ACTION_TILE_OFFSET:]
    th = env.town_hall
    best, bd = C.ACTION_TILE_OFFSET, 1e9
    for i in np.flatnonzero(mask):
        x, y = env.action_to_tile(i + C.ACTION_TILE_OFFSET)
        d = math.hypot(x - th.cx, y - th.cy)
        if d < bd:
            bd, best = d, i + C.ACTION_TILE_OFFSET
    return best


def policy_random(env, t: int) -> int:
    return int(np.random.choice(np.flatnonzero(env.legal_actions())))


def policy_never_cast(env, t: int) -> int:
    if t == -1:
        return deploy_nearest_th(env)
    return C.ACTION_WAIT


def policy_scripted_human(env, t: int) -> int:
    if t == -1:
        return deploy_nearest_th(env)
    x = int(np.clip(env.rc_pos[0], 0, C.GRID_SIZE - 1))
    y = int(np.clip(env.rc_pos[1], 0, C.GRID_SIZE - 1))
    if env.ability_left > 0 and env.rc_hp < 0.35 * C.RC_MAX_HP:
        return C.ACTION_ABILITY
    in_threat = env._threat_channel()[y, x] > 0
    covered = env.invis[y, x]
    if env.spells_left > 0 and in_threat and covered <= C.DT * 1.5:
        ax, ay = x // C.ACTION_STRIDE, y // C.ACTION_STRIDE
        return C.ACTION_TILE_OFFSET + ay * C.ACTION_GRID + ax
    return C.ACTION_WAIT


BASELINES = {
    "random": policy_random,
    "never-cast": policy_never_cast,
    "scripted-human": policy_scripted_human,
}


# ======================================================================
# Scripted baseline for the FULL attack: RC charge -> funnel -> dragons
#
# This encodes what the guides actually say to do, and nothing cleverer:
#
#   * deploy the Champion near the Town Hall side
#   * chain Invisibility on her roughly every 4 seconds while she is exposed
#   * Seeking Shield when she drops below 35%
#   * when the spells run out (or she dies, or the 1:20 deadline hits),
#     commit the air army ON THE SIDE THE AIR SWEEPERS ARE NOT FACING
#   * two Baby Dragons on the flanks, far apart so both keep their tantrum
#   * fourteen Dragons in a wide line, not a stack
#   * Warden behind the middle of the line
#   * Eternal Tome when the stack is deepest in air threat
# ======================================================================
from .full_env import (A_ARROW, A_DEPLOY, A_DUKE, A_SHIELD, A_SPELL,
                       A_TOME, A_WAIT)


def _cell_of(x, y):
    return (y // C.ACTION_STRIDE) * C.ACTION_GRID + (x // C.ACTION_STRIDE)


def choose_air_side(env):
    """Pick the deployment arc for the dragons.

    Scores every legal deploy cell by the Air Sweeper coverage it would fly
    through on its way to the centre, and returns the cells sorted best-first.
    This is the decision the whole air phase turns on: two sweepers, 120-degree
    cones, and no Rage or Haste in this army to punch through one.
    """
    mask = env.legal_actions()[A_DEPLOY:]
    cov = env._sweeper_cov
    air = env._at()
    cands = []
    for i in np.flatnonzero(mask):
        ay, ax = divmod(i, C.ACTION_GRID)
        x = ax * C.ACTION_STRIDE + 1
        y = ay * C.ACTION_STRIDE + 1
        # sample the flight path from here to the centre
        score = 0.0
        for t in np.linspace(0.0, 1.0, 8):
            px = int(np.clip(x + (22 - x) * t, 0, C.GRID_SIZE - 1))
            py = int(np.clip(y + (22 - y) * t, 0, C.GRID_SIZE - 1))
            score += cov[py, px] * 3.0 + air[py, px] / 600.0
        cands.append((score, i))
    cands.sort()
    return [i for _, i in cands]


def policy_full_attack(env, t: int) -> int:
    # ---- deployment of the Royal Champion -----------------------------
    if not env.rc_deployed:
        return A_DEPLOY + deploy_nearest_th_full(env)

    rx = int(np.clip(env.rc_pos[0], 0, C.GRID_SIZE - 1))
    ry = int(np.clip(env.rc_pos[1], 0, C.GRID_SIZE - 1))

    charge_over = (env.rc_dead or env.spells_left == 0
                   or env.time >= C.DRAGON_DEADLINE_S)

    # ---- phase 1: the charge ------------------------------------------
    if not charge_over:
        if env.shield_left > 0 and env.rc_hp < 0.35 * C.RC_MAX_HP:
            return A_SHIELD
        if (env.spells_left > 0 and env._gt()[ry, rx] > 0
                and env.invis[ry, rx] <= C.DT * 1.5):
            return A_SPELL + _cell_of(rx, ry)
        return A_WAIT

    # ---- phase 2: the Giant Arrow Air Defense snipe --------------------
    # Fire the instant she lands, before she walks or re-targets. The arrow
    # flies along the line from the Queen through her CURRENT target, so any
    # movement ruins the alignment you just paid a deploy slot for.
    if env.arrow_left > 0 and any(u.kind == "Archer Queen" and not u.is_dead
                                  for u in env.units):
        return A_ARROW

    # ---- phase 3: commit the army -------------------------------------
    if env.queue:
        if not hasattr(env, "_side_cache") or env._side_cache is None:
            env._side_cache = choose_air_side(env)
        order = env._side_cache
        kind = env.queue[0]
        if not order:
            return A_WAIT
        if kind == "Archer Queen":
            # Drop her where the arrow lines up through the most Air Defenses.
            # The environment already solved that geometry -- read it off.
            am = env._arrow_map
            if am is None:
                from .army import arrow_value_map
                am = arrow_value_map(env._deploy_mask(), env.buildings, C.GRID_SIZE)
                env._arrow_map = am
            flat = am.reshape(-1)
            legal = env.legal_actions()[A_DEPLOY:]
            scored = np.where(legal, flat, -1.0)
            if scored.max() > 0:
                return A_DEPLOY + int(np.argmax(scored))
            return A_DEPLOY + order[0]
        if kind == "Dragon Duke":
            # solo, far from the stack, or Royal Rampage never turns on
            return A_DEPLOY + order[max(0, len(order) - 2)]
        if kind == "Baby Dragon":
            # flanks: take from opposite ends of the sorted safe arc so the two
            # Baby Dragons are far apart and BOTH keep their tantrum buff
            idx = order[0] if len(env.units) == 0 else order[min(len(order) - 1,
                                                                len(order) // 3)]
            return A_DEPLOY + idx
        if kind == "Dragon":
            n = sum(1 for u in env.units if u.kind == "Dragon")
            return A_DEPLOY + order[min(len(order) - 1, 2 + (n * 2) % 12)]
        return A_DEPLOY + order[min(len(order) - 1, 6)]

    # ---- phase 4: hero abilities --------------------------------------
    if env.duke_heal_left > 0:
        d = next((u for u in env.units
                  if u.kind == "Dragon Duke" and not u.is_dead), None)
        if d is not None and d.hp < 0.4 * d.max_hp:
            return A_DUKE
    if env.tome_left > 0:
        alive = [u for u in env.units if not u.is_dead]
        if alive:
            at = env._at()
            hot = sum(1 for u in alive
                      if at[int(np.clip(u.y, 0, 43)), int(np.clip(u.x, 0, 43))] > 900)
            if hot >= max(2, len(alive) // 3):
                return A_TOME
    if env.spells_left > 0 and not env.rc_dead and env._gt()[ry, rx] > 0:
        return A_SPELL + _cell_of(rx, ry)
    return A_WAIT


def deploy_nearest_th_full(env) -> int:
    """Deploy cell index (not action) nearest the Town Hall."""
    mask = env.legal_actions()[A_DEPLOY:]
    th = env.town_hall
    best, bd = 0, 1e9
    for i in np.flatnonzero(mask):
        ay, ax = divmod(i, C.ACTION_GRID)
        x = ax * C.ACTION_STRIDE + 1
        y = ay * C.ACTION_STRIDE + 1
        d = math.hypot(x - th.cx, y - th.cy)
        if d < bd:
            bd, best = d, i
    return best
