# ======================================================================
#  ROYAL CHAMPION WALK -- Clash of Clans RL attack agent  (single cell)
#
#  Auto-generated from src/coc/ by scripts/build_single_cell.py -- do not edit
#  this file by hand, edit the package and rebuild.
#
#  Paste this whole thing into one Jupyter cell and run it, then:
#
#     trainer = Trainer(TrainConfig(), out_dir="runs/rc")
#     trainer.load()          # resumes from the latest checkpoint if there is one
#     trainer.train()         # Ctrl-C is safe: resume picks up exactly where it stopped
#
#  To see how it plays:
#     compare(model_path="runs/rc/latest.pt", episodes=100)
# ======================================================================
from __future__ import annotations

import csv, glob, json, math, os, random, statistics, time
from collections import deque
from dataclasses import dataclass, field, asdict, replace
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import animation



# ====================================================================
# ---- config.py ---------------------------------------------------
"""
Game constants and training configuration for the Royal Champion walk agent.

Every game number here is sourced from the Clash of Clans Fandom wiki / clasher.us
at TOWN HALL 15 MAX LEVEL. Sources are cited inline. If a value is a deliberate
simplification of the real game it is marked  # SIMPLIFIED.

The single most important correction over the previous version:

    The Royal Champion targets the NEAREST DEFENSE. Not "Town Hall first, then
    Air Defenses". The Town Hall counts as a defense at TH12+ because of its
    Giga Inferno, so she can walk straight to it if it is the nearest defense.
    Only when zero defenses remain does she attack other buildings.

That one rule is what makes a Town Hall snipe possible, and therefore what makes
this task learnable at all.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import math

# --------------------------------------------------------------------------
# Map / simulation
# --------------------------------------------------------------------------
GRID_SIZE = 44          # playable village area is 44x44 tiles
DT = 0.5                # seconds of game time per environment tick
BATTLE_SECONDS = 180    # a real attack is 3 minutes
MAX_STEPS = int(BATTLE_SECONDS / DT)          # 360 ticks

# The agent picks WHERE to deploy her (this was random before). Real deployment
# happens on the map border, outside the base.
DEPLOY_MARGIN = 1       # deploy tiles are this far in from the map edge

# Spell placement resolution. The Invisibility Spell has a 4-tile radius, so
# choosing a target tile to 1-tile precision is wasted resolution -- and it
# quadruples the action space for nothing. Stride 2 gives a 22x22 = 484 tile
# action grid instead of 44x44 = 1936.
ACTION_STRIDE = 2
ACTION_GRID = GRID_SIZE // ACTION_STRIDE      # 22

# Action layout:  0 = wait, 1 = Seeking Shield, 2.. = cast invisibility on tile
ACTION_WAIT = 0
ACTION_ABILITY = 1
ACTION_TILE_OFFSET = 2
N_ACTIONS = ACTION_TILE_OFFSET + ACTION_GRID * ACTION_GRID     # 486


# --------------------------------------------------------------------------
# Royal Champion  (level 40 = TH15 max)
#   https://clashofclans.fandom.com/wiki/Royal_Champion
#   https://www.clasher.us/clash-of-clans/unit/Royal_Champion
# --------------------------------------------------------------------------
RC_BASE_HP = 3910       # level 40
RC_BASE_DPS = 530       # level 40 (636 damage per hit / 1.2s attack speed)
RC_RANGE = 3.0          # tiles
RC_ATTACK_SPEED = 1.2   # seconds between hits
# In-game movement speed 24. The wiki's conversion is roughly speed/8 tiles per
# second:  https://clashofclans.fandom.com/wiki/Troop_Movement_Speed
RC_SPEED = 24.0 / 8.0   # = 3.0 tiles/second
RC_HOUSING = 25         # heroes cost 25 housing space (matters for the Eagle)

# She jumps walls like a Hog Rider, so walls never block her path.
RC_JUMPS_WALLS = True


# --------------------------------------------------------------------------
# Seeking Shield  (her ability)
#   Throws a shield that seeks up to FOUR targets, prioritising defenses,
#   REGARDLESS OF DISTANCE, and heals her.
#   https://clashofclans.fandom.com/wiki/Seeking_Shield
# --------------------------------------------------------------------------
SHIELD_TARGETS = 4
SHIELD_DAMAGE = 1860    # per target, ability level 4
SHIELD_HEAL = 2600      # ability level 4
SHIELD_USES = 1         # one activation per battle


# --------------------------------------------------------------------------
# Electro Boots  (equipment)
#   Passive damage aura + bonus hitpoints + passive self-heal.
#   https://clashofclans.fandom.com/wiki/Electro_Boots
#   Values below are level 18, a realistic TH15 boot level.
# --------------------------------------------------------------------------
ELECTRO_RADIUS = 5.0
ELECTRO_DPS = 177
ELECTRO_HP_BONUS = 2400
ELECTRO_HEAL_PER_SEC = 36

RC_MAX_HP = RC_BASE_HP + ELECTRO_HP_BONUS     # 6310
RC_DPS = RC_BASE_DPS


# --------------------------------------------------------------------------
# Invisibility Spell
#   4 tile radius, 4.25s at level 3. Makes friendly units UNTARGETABLE, and
#   makes enemy buildings untargetable BY YOUR TROOPS (so she walks past them)
#   -- but a defense that is itself invisible STILL FIRES at visible targets.
#   https://clashofclans.fandom.com/wiki/Invisibility_Spell
# --------------------------------------------------------------------------
SPELL_RADIUS = 4.0
SPELL_DURATION = 4.25
SPELL_HOUSING = 1

# TH15 spell capacity is 11 (Spell Factory lvl 8 gives 10, Dark Spell Factory
# adds 1 dark slot). Invisibility is an elixir spell, so the honest maximum in
# an all-invisibility army is 10.
#   https://clashofclans.fandom.com/wiki/Spell_Factory
MAX_SPELLS_CAP = 10
# A realistic mixed army carries 2-4; guides recommend 2 at TH13-15. We train
# toward the all-in charge army, which is what this agent models.
MAX_SPELLS = 8


# --------------------------------------------------------------------------
# Town Hall 15 / Giga Inferno
#   https://clashofclans.fandom.com/wiki/Town_Hall/Giga_Inferno_(TH15)
# --------------------------------------------------------------------------
TH_HP = 9600
TH_DPS = 300            # Giga Inferno, multi-target (up to 4)
TH_RANGE = 10.0
TH_DEATH_DAMAGE = 1000  # poison bomb on destruction
TH_DEATH_RADIUS = 4.5
TH_POISON_DPS = 180
TH_POISON_DURATION = 12.0
TH_POISON_SLOW = 0.5    # 50% slower movement and attack rate


# --------------------------------------------------------------------------
# Giga Inferno activation
#   The TH15 Giga Inferno is DORMANT until the Town Hall takes direct damage,
#   or until 51% of the base is destroyed. This is why every RC-charge guide
#   says to path her deep and AVOID an early Town Hall takedown -- touching it
#   switches on a 300 DPS multi-target weapon and arms the death poison.
#   https://theriagames.com/guide/clash-of-clans-giga-inferno-th15/
#   She still treats the Town Hall as a defense for targeting purposes.
# --------------------------------------------------------------------------
TH_ACTIVATE_DESTRUCTION = 0.51


# --------------------------------------------------------------------------
# Spell Tower (Poison mode)
#   Low damage, but -35% movement and -25% attack rate for 12 seconds. The
#   damage is irrelevant against her HP pool; the SLOW is the problem, because
#   it breaks the ~4 second casting rhythm -- she covers less ground per spell
#   and a 10-spell walk quietly becomes a 14-spell walk.
#   https://clashofclans.fandom.com/wiki/Spell_Tower
# --------------------------------------------------------------------------
SPELL_TOWER_DPS = 60.0
SPELL_TOWER_TRIGGER = 9.0
SPELL_TOWER_DURATION = 12.0
SPELL_TOWER_DELAY = 1.2
SPELL_TOWER_SLOW_MOVE = 0.65       # -35%
SPELL_TOWER_SLOW_ATTACK = 0.75     # -25%


# --------------------------------------------------------------------------
# Spirit Fox (pet)
#   The standard pairing with Electro Boots: it cloaks her periodically, and
#   because the boots' self-heal is passive, every second of invisibility is
#   also a second of healing with zero incoming damage. That loop -- cloak,
#   regen, emerge topped up -- is the actual engine of the RC charge, and it is
#   why the charge goes so much deeper than the old healer-based RC walk.
#   The fox alone leaves a ~2 second naked window each cycle; the spells exist
#   to cover that window, not to replace it.
#   https://clashofclans.fandom.com/wiki/Spirit_Fox
# --------------------------------------------------------------------------
SPIRIT_FOX = True
FOX_INVIS_DURATION = 4.0
FOX_COOLDOWN = 6.0


# --------------------------------------------------------------------------
# Defenses at TH15 max level.
#
# Fields: (count, size, hp, dps, range, min_range, hits_ground, splash)
#
# CRITICAL, and wrong in the old version: AIR DEFENSE AND AIR SWEEPER CANNOT
# TOUCH A GROUND UNIT. The Royal Champion is a ground troop. Air Defenses deal
# her exactly zero damage. They are still defenses (she will target them, they
# count for destruction %) but there is no survival reason to prioritise them.
# The old +2500 "kill an Air Defense" reward was the single most misleading
# incentive in the reward function -- it is a Queen-walk-with-Healers idea that
# does not apply to a solo Champion.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DefenseSpec:
    name: str
    count: int
    size: Tuple[int, int]
    hp: int
    dps: float
    rng: float
    min_range: float = 0.0
    hits_ground: bool = True
    splash: bool = False
    tier: str = "normal"          # "high" = priority target, used for rewards


DEFENSES: List[DefenseSpec] = [
    # name              n  size    hp    dps   rng  min  ground  splash tier
    DefenseSpec("Cannon",        7, (3, 3), 2250, 160.0,  9.0, 0.0, True,  False, "normal"),
    DefenseSpec("Archer Tower",  8, (3, 3), 1800, 145.0, 10.0, 0.0, True,  False, "normal"),
    DefenseSpec("Mortar",        4, (3, 3), 2150,  54.0, 11.0, 4.0, True,  True,  "normal"),
    DefenseSpec("Wizard Tower",  5, (3, 3), 3000,  95.0,  7.0, 0.0, True,  True,  "normal"),
    DefenseSpec("Hidden Tesla",  5, (2, 2), 1450, 160.0,  7.0, 0.0, True,  False, "normal"),
    DefenseSpec("Bomb Tower",    2, (3, 3), 2500,  94.0,  6.0, 0.0, True,  True,  "normal"),
    DefenseSpec("X-Bow",         4, (3, 3), 4400, 205.0, 11.5, 0.0, True,  False, "high"),
    DefenseSpec("Inferno Tower", 3, (2, 2), 4000, 100.0,  9.0, 0.0, True,  False, "high"),
    DefenseSpec("Scattershot",   2, (3, 3), 5100, 175.0, 10.0, 3.0, True,  True,  "high"),
    DefenseSpec("Monolith",      1, (3, 3), 5050, 175.0, 11.0, 0.0, True,  False, "high"),
    DefenseSpec("Eagle Artillery", 1, (4, 4), 5900, 142.5, 50.0, 7.0, True, True, "high"),
    DefenseSpec("Spell Tower",   2, (2, 2), 3100,   0.0,  9.0, 0.0, True,  False, "normal"),
    # Air-only. Zero threat to a ground hero.
    DefenseSpec("Air Defense",   4, (3, 3), 1750, 540.0, 10.0, 0.0, False, False, "normal"),
    DefenseSpec("Air Sweeper",   2, (2, 2), 1050,   0.0, 15.0, 1.0, False, False, "normal"),
]

# Inferno Tower single-target ramp: 100 DPS, ->230 after 1.5s, ->2300 after 5.25s.
# The ramp RESETS when it loses its target -- which is exactly what happens when
# she turns invisible. This is the mechanic that makes spell timing matter.
INFERNO_RAMP = [(0.0, 100.0), (1.5, 230.0), (5.25, 2300.0)]

# Monolith deals 12% of the target's MAX hp as bonus damage per shot (lvl 2).
MONOLITH_PCT_DAMAGE = 0.12

# Eagle Artillery stays dormant until 200 housing space has been deployed.
# A solo Royal Champion (25) plus 8 invisibility spells (5 each = 40) is 65.
# So in this attack THE EAGLE NEVER WAKES UP. That is not a bug, it is the
# actual rule, and it is a large part of why a solo Champion charge is viable.
#   https://clashofclans.fandom.com/wiki/Eagle_Artillery
EAGLE_ACTIVATION_HOUSING = 200
SPELL_HOUSING_COST = 5      # spells count 5 housing each toward the Eagle

NON_DEFENSE_COUNT = 45      # collectors, storages, army buildings, etc.
NON_DEFENSE_HP = 1200
WALL_SEGMENTS = 250
WALL_HP = 11000


# --------------------------------------------------------------------------
# Traps at TH15 max.
#
# Only the ones that can touch a GROUND HERO are listed. Air Bombs and Seeking
# Air Mines are air-only and irrelevant to her.
#
# The rule that matters most: INVISIBILITY DOES NOT STOP TRAPS. Fandom:
# "Traps are not affected by this spell. An invisible troop walking over a trap
# will trigger as normal." A flawless casting rhythm still walks her onto a
# Giant Bomb -- which is why the threat map is not the whole picture.
#
# Spring Trap note: heroes used to be immune. The October 2025 rework changed
# that -- it now always deals damage, halved against heroes, and heroes are
# never ejected off the map.
#   https://clashofclans.fandom.com/wiki/Giant_Bomb
#   https://supercell.com/en/games/clashofclans/blog/news/upcoming-changes-to-spring-trap-th-weapon-level-removal/
# --------------------------------------------------------------------------
TRAPS = [
    # name,           count, size,   trigger, damage, radius
    ("Bomb",              8, (1, 1), 1.5,      155.0, 3.0),
    ("Giant Bomb",        7, (2, 2), 2.0,      425.0, 4.0),
    ("Spring Trap",       9, (1, 1), 1.0,      525.0, 0.5),   # 1050 halved vs heroes
    ("Tornado Trap",      1, (1, 1), 3.0,       56.0, 3.0),   # 8 dps x 7s
]
# Skeleton Trap is handled separately: it spawns units rather than dealing
# damage, and the skeletons pull her off her target (the same aggro rule as
# Clan Castle troops). The Electro Boots aura shreds them almost instantly,
# which is one of the quieter reasons the boots are the equipment of choice.
SKELETON_TRAPS = 2
SKELETON_COUNT = 5
SKELETON_HP = 30.0
SKELETON_DPS = 25.0


# --------------------------------------------------------------------------
# Defending Clan Castle (level 11 at TH15: 5,400 hp, 3x3, 50 housing)
#   Trigger radius 13 tiles. Once out, the troops have NO leash and will chase
#   across the entire map.
#   https://clashofclans.fandom.com/wiki/Clan_Castle
# --------------------------------------------------------------------------
CC_HP = 5400
CC_TRIGGER_RADIUS = 13.0

CC_TROOPS = {
    # Headhunter: the purpose-built hero counter. 125 base DPS but x4 against
    # heroes = 500, plus a poison that cuts her movement 44% and attack rate 65%.
    "Headhunter":  dict(hp=440,  dps=125.0, range=3.0, speed=32, housing=6,
                        hero_mult=4.0, poison=True),
    # Ice Golem: doesn't kill her, costs her TIME, and freezes on death.
    "Ice Golem":   dict(hp=3900, dps=48.0,  range=1.0, speed=12, housing=15,
                        freeze=3.5, freeze_radius=5.5),
    "Super Minion": dict(hp=1800, dps=360.0, range=4.0, speed=16, housing=12),
    "Archer":      dict(hp=304,  dps=50.0,  range=3.5, speed=24, housing=1),
}
# ~36 of 50 housing. A realistic anti-Royal-Champion castle.
CC_COMPOSITION = ["Headhunter", "Headhunter", "Ice Golem"]

HEADHUNTER_POISON_DURATION = 3.0
HEADHUNTER_SLOW_MOVE = 0.56      # -44%
HEADHUNTER_SLOW_ATTACK = 0.35    # -65%


# --------------------------------------------------------------------------
# Defending Archer Queen (level 90). She patrols around her altar rather than
# chasing across the map, and retreats if lured too far.
#   https://clashofclans.fandom.com/wiki/Archer_Queen
# --------------------------------------------------------------------------
DEFENDING_QUEEN = dict(hp=3096, dps=748.0, range=5.0, speed=24, patrol=12.0)


# --------------------------------------------------------------------------
# WHAT THE CHARGE IS FOR -- target value profiles
#
# The single most important thing to get right, and the easiest to get wrong.
#
# A Royal Champion charge is almost never run solo. It is the opening phase of
# an air attack, and its job is to delete the defenses that will shred the
# dragons coming in behind her. So the value of killing something is NOT the
# same as the threat it poses to HER:
#
#   * Air Defenses and Air Sweepers cannot touch her at all -- she is a ground
#     unit -- and they are nonetheless among the most valuable kills on the map,
#     because they are the reason the charge is happening.
#   * A Cannon can genuinely hurt her and is worth almost nothing to kill.
#
# She targets the nearest defense and the agent cannot pick her target directly.
# What it CAN do is steer her: choose the deploy tile so the chain of nearest
# defenses starts where you want, and drop Invisibility on buildings you want
# her to SKIP so she walks past them to the next one. That is the real skill in
# an RC charge, and it is the thing a fixed heuristic cannot do.
#
# Values are in the same units as the rest of the reward function.
# --------------------------------------------------------------------------
TARGET_PROFILES = {
    # Default: clearing a path for a dragon / e-drag / Root Rider army.
    "air_support": {
        "Town Hall":       15.0,
        "Air Defense":      6.0,
        "Monolith":         5.0,
        "Scattershot":      4.0,
        "Eagle Artillery":  4.0,
        "Air Sweeper":      3.0,
        "Inferno Tower":    2.0,
        "X-Bow":            1.0,
        "Wizard Tower":     0.6,
        "Spell Tower":      0.5,
        "Archer Tower":     0.3,
        "Hidden Tesla":     0.3,
        "Bomb Tower":       0.3,
        "Cannon":           0.2,
        "Mortar":           0.2,
    },
    # A lone Champion going for the Town Hall and nothing else.
    "solo_snipe": {
        "Town Hall":       20.0,
        "Monolith":         2.0,
        "Inferno Tower":    1.5,
        "Scattershot":      1.5,
        "Eagle Artillery":  1.0,
        "X-Bow":            1.0,
        "Air Defense":      0.2,
        "Air Sweeper":      0.1,
    },
    # Opening for a ground army: kill what deletes high-HP ground troops.
    "ground_support": {
        "Town Hall":       12.0,
        "Monolith":         8.0,
        "Inferno Tower":    6.0,
        "Scattershot":      5.0,
        "Eagle Artillery":  4.0,
        "X-Bow":            2.0,
        "Wizard Tower":     1.0,
        "Bomb Tower":       0.8,
        "Air Defense":      0.1,
        "Air Sweeper":      0.1,
    },
}
TARGET_PROFILE = "air_support"

# Anything not named in the profile
TARGET_VALUE_DEFAULT = 0.2
TARGET_VALUE_TRASH = 0.05        # collectors, storages, army buildings
# Normaliser for the priority observation channel
TARGET_VALUE_NORM = 15.0


def target_value(name: str, is_defense: bool, profile: str = None) -> float:
    """Reward for destroying a building under the active profile."""
    prof = TARGET_PROFILES[profile or TARGET_PROFILE]
    if name in prof:
        return prof[name]
    return TARGET_VALUE_DEFAULT if is_defense else TARGET_VALUE_TRASH


# --------------------------------------------------------------------------
# Rewards
#
# Rewritten from scratch. Old scale was -2000..+5000 with MSE loss, which makes
# the TD targets enormous and the gradients unstable. Everything here is in
# "points" of roughly unit scale, so a good episode returns about +20 and a bad
# one about -3.
#
# The other change is philosophical. In the old reward, dying cost -2000 and so
# dominated everything. But in real Clash of Clans a Royal Champion who destroys
# the Town Hall and then dies HAS SUCCEEDED -- that is a star. Death is now a
# small penalty, and the Town Hall is the prize.
# --------------------------------------------------------------------------
R_TH_DESTROYED = 15.0        # the star. this is the objective.
R_STAR_50_PCT = 5.0          # 50% destruction is also a star
R_STAR_100_PCT = 10.0        # 100% is the third star
R_DEATH = -2.0               # modest: dying after the job is done is fine
R_TIMEOUT = -2.0
R_PER_STEP = -0.01           # mild pressure to be quick

R_DAMAGE_DEALT = 1.0 / 4000.0    # dense shaping: progress signal every tick
R_DAMAGE_TH_MULT = 3.0           # chipping the Town Hall is worth more than chipping a hut
R_DAMAGE_TAKEN = -1.0 / 4000.0
R_KILL_HIGH = 1.0            # X-Bow / Inferno / Scattershot / Monolith / Eagle
R_KILL_NORMAL = 0.4
R_KILL_TRASH = 0.05
R_KILL_DEFENDER = 0.3        # clearing CC troops / skeletons is progress, not waste
R_SPELL_CAST = -0.05         # spells are a scarce resource, not free
R_ABILITY_USE = -0.05

# Curriculum stages. Each stage keeps the real TH15 stats but scales how much
# base she has to survive. `defense_frac` is the fraction of the full defense
# roster placed; `spells` is how many Invisibility Spells she carries.
#
# The point of the curriculum is that the agent must SEE WINS to learn. Stage 0
# is deliberately easy enough that even a mediocre policy sometimes snipes the
# Town Hall, which puts the +15 into the replay buffer. Everything after that is
# bootstrapping.
@dataclass
class Stage:
    name: str
    defense_frac: float
    spells: int
    promote_win_rate: float      # win rate over the eval window to move up
    traps: bool = False          # Bombs / Giant Bombs / Spring / Tornado / Skeletons
    cc: bool = False             # defending Clan Castle troops
    hero: bool = False           # defending Archer Queen


# Curriculum, calibrated by actually measuring the scripted baseline on every
# combination. Town Hall kill rate for `scripted-human`:
#
#     spells ->     10      8      6      4      3
#     frac 0.25    100%    95%    88%    52%    50%
#     frac 0.40     88%    80%    50%    20%    10%
#     frac 0.55     78%    68%    42%     8%     8%
#     frac 0.75     68%    50%    18%     0%     0%
#     frac 1.00     40%    12%     2%     0%     0%
#
# Stage 0 is easy enough that demonstrations are almost all successful, which is
# what gives the agent something to bootstrap from. Stage 4 is the realistic
# target: a full TH15 base against a dedicated 10-Invisibility charge army, the
# composition real guides actually run. Stages 5 and 6 are where the scripted
# heuristic falls apart and a learned policy has to earn its keep.
CURRICULUM: List[Stage] = [
    Stage("s0-easy",       0.25, 10, 0.70),
    Stage("s1-light",      0.40, 10, 0.60),
    Stage("s2-traps",      0.60, 10, 0.50, traps=True),
    Stage("s3-traps",      0.80, 10, 0.45, traps=True),
    Stage("s4-full",       1.00, 10, 0.40, traps=True),
    Stage("s5-cc",         1.00, 10, 0.30, traps=True, cc=True),
    Stage("s6-everything", 1.00, 10, 0.00, traps=True, cc=True, hero=True),
]


# ==========================================================================
# THE FULL ATTACK:  RC charge -> funnel -> mass dragons
#
# This is what the agent actually plays now. A Royal Champion charge is not an
# attack, it is the OPENING PHASE of one: she goes in alone under a chain of
# Invisibility Spells and deletes the core anti-air, and then fourteen Dragons
# fly into a base that can no longer shoot them down.
#
# Two consequences reshape the whole problem:
#
#   1. AIR DEFENSES NOW MATTER ENORMOUSLY. They cannot touch the Champion --
#      she is ground -- but they are 540 DPS each against the army behind her.
#      "Harmless to her, lethal to them" is the entire point of the charge.
#
#   2. THREE STARS BECOMES REACHABLE. A solo Champion structurally could not
#      pass 50% destruction because she walks past every collector. Dragons
#      target the NEAREST BUILDING, so they clear everything.
#
# Army: the canonical Blueprint CoC list -- 14 Dragons, 2 Baby Dragons,
# 11 Invisibility Spells, no Rage, no Freeze. All spell budget is the
# Champion's life support.
#   https://blueprintcoc.com/blogs/town-hall-15/best-th15-attack-strategies
#   https://www.youtube.com/watch?v=5C4IwCq1sv0  (Blueprint CoC)
# ==========================================================================

ARMY_CAMP_CAPACITY = 320        # 4 x level 12 at TH15
SPELL_CAPACITY = 11
HERO_SLOTS = 4                  # Hero Hall: only 4 of 6 heroes may be deployed

# ---- Dragon (level 10 = TH15 max) ----
#   https://clashofclans.fandom.com/wiki/Dragon
DRAGON = dict(hp=4900, dps=370.0, rng=3.0, speed=16, splash=0.3, housing=20)
DRAGON_COUNT = 14

# ---- Baby Dragon (level 9 = TH15 max) ----
#   Tantrum: +100% damage and +50% attack speed when NO allied air unit is
#   within 4.5 tiles. This is why funnel Baby Dragons must be dropped far from
#   the stack AND far from each other -- an isolated one does 310 DPS, which is
#   nearly a full Dragon for half the housing.
#   https://clashofclans.fandom.com/wiki/Baby_Dragon
BABY_DRAGON = dict(hp=2000, dps=155.0, rng=2.75, speed=20, splash=0.3, housing=10)
BABY_DRAGON_COUNT = 2
TANTRUM_RADIUS = 4.5
TANTRUM_DAMAGE_MULT = 2.0
TANTRUM_SPEED_MULT = 1.5

# ---- Grand Warden (level 65 = TH15 max), AIR MODE ----
#   Air mode is mandatory for a dragon army: in ground mode he falls behind the
#   flying stack within seconds and dies alone outside his own aura. The cost is
#   that Air Defenses and Seeking Air Mines can now hit him.
#   Eternal Tome grants INVULNERABILITY -- damage only. It does not stop Air
#   Sweeper knockback or slows.
#   https://clashofclans.fandom.com/wiki/Grand_Warden
#   https://clashofclans.fandom.com/wiki/Eternal_Tome
WARDEN = dict(hp=2329, dps=297.0, rng=7.0, speed=16, housing=0)
WARDEN_AURA_RADIUS = 8.0        # sources disagree 7-10; tunable
WARDEN_AURA_HP_BONUS = 0.35     # Life Aura, fraction of max hp
TOME_DURATION = 9.0             # Eternal Tome 15-18
TOME_RADIUS = 8.0

# ---- Archer Queen (level 90 = TH15 max), attacking ----
QUEEN = dict(hp=3096, dps=748.0, rng=5.0, speed=24, housing=0)

# ---- Air Sweeper: the mechanic that decides which side you attack from ----
#   120 degree arc, EIGHT discrete facings set by the defender before the
#   battle and LOCKED once it starts. Zero damage -- it only pushes air units
#   back 4 tiles and mutes them for 1.2s. It beats you with the clock.
#   Two of them at TH15, usually pointed at offset quadrants so no side is
#   fully safe: the goal is to eat ONE sweeper instead of two.
#   https://clashofclans.fandom.com/wiki/Air_Sweeper
SWEEPER_ARC_DEG = 120.0
SWEEPER_RANGE = 15.0
SWEEPER_MIN_RANGE = 1.0
SWEEPER_PUSH = 4.0              # level 7
SWEEPER_COOLDOWN = 5.0
SWEEPER_MUTE = 1.2
SWEEPER_FACINGS = 8             # 45 degrees apart

# ---- Anti-air traps ----
#   https://coc.guide/trap/megaairtrap
SEEKING_AIR_MINE = dict(count=8, damage=2700.0, trigger=4.0)
AIR_BOMB = dict(count=7, damage=350.0, trigger=4.0, radius=3.0)

# ---- Deployment / phases ----
# Order matters and matches how the attack is actually executed:
#   Queen FIRST after the charge, so the Giant Arrow deletes the Air Defenses
#   BEFORE the dragons commit; then the funnel; then the stack; the Duke goes
#   solo down the far flank to keep Royal Rampage alive; the Warden goes in
#   behind the stack so it overtakes him and he ends up inside it.
DEPLOY_ORDER = (["Archer Queen"]
                + ["Baby Dragon"] * BABY_DRAGON_COUNT
                + ["Stone Slammer"]
                + ["Dragon"] * DRAGON_COUNT
                + ["Dragon", "Dragon", "Baby Dragon"]     # Clan Castle
                + ["Dragon Duke", "Grand Warden"])
# Hard deadline every guide quotes: the dragons must be committed by 1:40 left,
# i.e. 1:20 elapsed. A charge still limping at that point should be abandoned.
DRAGON_DEADLINE_S = 80.0

# ---- Rewards for the full attack ----
# Now that dragons clear everything, destruction percentage is a real objective
# and the three stars are genuinely reachable.
R_STAR_1 = 8.0                  # 50% destruction
R_STAR_2 = 12.0                 # Town Hall
R_STAR_3 = 25.0                 # 100%
R_DESTRUCTION_PER_PCT = 0.4     # dense: every percent of the base is worth this
R_UNIT_LOST = -0.3              # losing dragons is losing damage output


# ==========================================================================
# GIANT ARROW  (Archer Queen equipment)  -- the Air Defense snipe
#
# Reworked 26 May 2026. The Fandom wiki is STALE on this; these are the
# official numbers:
#     base damage 1,500 at level 18  (max level available at TH15)
#     "Extra Damage vs Air Defence: 1x -> 2x"   ->  3,000 against an Air Defense
# A max TH15 Air Defense has 1,750 hp, so ONE ARROW KILLS IT with 171% headroom
# -- and the arrow pierces with NO falloff, so it kills every Air Defense on the
# line.
#   https://supercell.com/en/games/clashofclans/blog/news/state-of-gameplay-part-2/
#   https://clashofclans.fandom.com/wiki/Giant_Arrow
#
# THE MECHANIC THAT MAKES THIS A PLACEMENT PROBLEM:
#
#   "The direction that the Giant Arrow travels depends on where the Archer
#    Queen's CURRENT TARGET is, relative to her location at the moment the
#    ability is used; the Giant Arrow will travel a straight line that passes
#    through these two locations."
#
# You do not aim the arrow. You aim the QUEEN. She will shoot at the nearest
# building to wherever she lands, and the arrow flies along that line. So the
# problem is: find a deploy tile where the nearest building to it AND two or
# more Air Defenses are COLLINEAR. The outer building is your gunsight.
#
# That is a clean geometric optimisation, so the environment computes it for
# every legal deploy tile and hands the agent the answer as an observation
# channel: "if you drop the Queen here and fire, this many Air Defenses die."
# ==========================================================================
GIANT_ARROW_DAMAGE = 1500.0
GIANT_ARROW_AD_MULT = 2.0       # 3,000 vs an Air Defense: a one-shot kill
GIANT_ARROW_WIDTH = 1.0         # hit radius either side of the line, in tiles
GIANT_ARROW_RANGE = 75.0        # crosses the whole map from any position
QUEEN_HP_BONUS = 581            # Giant Arrow lvl 18 passive
QUEEN_DPS_BONUS = 132


# ==========================================================================
# STONE SLAMMER  (siege machine, level 5 = TH15 max)
#   Flying siege that targets DEFENSES ONLY, and carries the Clan Castle
#   troops in -- they pop out when it dies or is detonated.
#   https://clashofclans.fandom.com/wiki/Stone_Slammer
# ==========================================================================
STONE_SLAMMER = dict(hp=6800, dps=750.0, rng=2.0, speed=16, splash=3.0,
                     death_damage=500.0, housing=0)


# ==========================================================================
# DRAGON DUKE  (Hero Hall 9 = TH15, max level 10)
#   Flying melee hero. His Royal Rampage passive fires when NO friendly air
#   unit is within 6 tiles: +100% damage and +50% attack speed, roughly 3x
#   effective DPS. So he is sent SOLO down the opposite flank -- put him in the
#   dragon stack and you have thrown the passive away.
#
#   Trap-damage reduction was nerfed twice (50% -> 40% -> 20% on 9 July 2026).
#   Most guides online still quote 50%; 20% is current.
#
#   Fire Heart is effectively mandatory because HEALERS CANNOT TARGET a flying
#   melee unit -- it is his only sustain. +5,600 hp, +45 dps, 150 hp/sec regen,
#   and a 3,000 damage explosion when he dies.
#   https://clashofclans.fandom.com/wiki/Dragon_Duke
# ==========================================================================
DRAGON_DUKE = dict(hp=9775 + 5600, dps=340.0 + 45.0, rng=1.25, speed=20,
                   splash=0.0, housing=0)
DUKE_RAMPAGE_RADIUS = 6.0
DUKE_RAMPAGE_DAMAGE = 2.0
DUKE_RAMPAGE_SPEED = 1.5
DUKE_TRAP_REDUCTION = 0.20
DUKE_REGEN = 150.0
DUKE_HEAL_ABILITY = 4000.0
DUKE_DEATH_EXPLOSION = 3000.0


# ==========================================================================
# CLAN CASTLE (attacking) -- level 11 at TH15: 50 troop / 3 spell / 1 siege
#   Dragons are 20 housing each, so two fit with ten spare (one Baby Dragon).
#   Invisibility is 1 housing space and IS allowed in the Castle, so three more
#   spells ride along -- 11 + 3 = 14 total, which matches the "14 Invisibility
#   Spells" armies the top creators run.
# ==========================================================================
CC_TROOP_HOUSING = 50
CC_SPELL_HOUSING = 3
CC_ATTACK_TROOPS = ["Dragon", "Dragon", "Baby Dragon"]     # 20 + 20 + 10 = 50
CC_ATTACK_SPELLS = 3                                        # Invisibility
TOTAL_SPELLS = SPELL_CAPACITY + CC_ATTACK_SPELLS            # 14


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------
# Spatial channels, all GRID_SIZE x GRID_SIZE:
#   0 Town Hall footprint
#   1 Air Defense footprint          (harmless to her, but still a defense)
#   2 High-value defense footprint   (X-Bow/Inferno/Scattershot/Monolith/Eagle)
#   3 Other defense footprint
#   4 Non-defense building footprint
#   5 Wall footprint
#   6 Remaining HP fraction of whatever occupies the tile
#   7 Threat map: incoming DPS that reaches this tile, normalised
#   8 Invisibility seconds remaining, normalised
#   9 HERO POSITION  <-- this did not exist before, and its absence was fatal
#  10 TARGET PRIORITY: how much each tile's building is worth killing under
#     the active profile. This is what lets the agent see that an Air Defense
#     is a prize and a Cannon is not, even though the Cannon is the one
#     shooting her.
N_SPATIAL_CHANNELS = 11
# Scalars concatenated into the fully-connected trunk:
#   hp fraction, spells fraction, ability ready, time fraction,
#   town hall destroyed, destruction fraction, defenses remaining fraction,
#   awaiting-deploy flag
N_SCALARS = 8

THREAT_NORM = 1500.0     # divisor for the threat channel


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # DQN
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 64
    memory_size: int = 50_000    # uint8 obs -> ~1.9 GB. lower this if RAM is tight
    learn_start: int = 2_000          # transitions before the first update
    train_every: int = 4              # gradient step every N env steps
    target_update_steps: int = 2_000  # hard target sync, in env steps
    grad_clip: float = 10.0
    double_dqn: bool = True
    dueling: bool = True
    huber_delta: float = 1.0
    n_step: int = 3                   # multi-step returns

    # exploration -- decayed on STEPS, and stored in the checkpoint, so a resume
    # does not silently jump back to 40% random like the old code did
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000

    # GUIDED EXPLORATION.
    # Destroying the Town Hall is a sparse event: uniformly random play almost
    # never does it, so a from-scratch DQN has nothing to bootstrap from. Two
    # standard fixes, both cheap:
    #   1. seed the replay buffer with scripted-policy episodes, so successful
    #      trajectories are in memory from step one (DQfD-style demonstrations)
    #   2. when exploring, sometimes take the scripted action instead of a
    #      uniformly random one, decaying to pure epsilon-greedy
    # The scripted policy is a plain heuristic, not another network -- it is the
    # same `scripted-human` baseline the agent is later measured against, so
    # beating it still means something.
    demo_episodes: int = 400
    bc_steps: int = 3_000        # behaviour-cloning warm start on the demo buffer
    bc_lambda_start: float = 1.0  # weight of the supervised demo loss during RL
    bc_anneal_steps: int = 40_000 # ...annealed to zero over this many updates
    teacher_start: float = 0.35
    teacher_end: float = 0.0
    teacher_decay_steps: int = 60_000

    # model width
    ch1: int = 24
    ch2: int = 48
    ch3: int = 48
    head_ch: int = 24
    global_dim: int = 128
    global_broadcast: int = 32

    # run control
    max_episodes: int = 100_000
    max_hours: float = 10.0
    eval_every: int = 250             # episodes
    eval_episodes: int = 40
    checkpoint_every: int = 250
    log_every: int = 10
    seed: int = 0
    device: str = "auto"
    curriculum: bool = True
    start_stage: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


def gpu_preset() -> TrainConfig:
    """Wider network and bigger batches when CUDA is available."""
    c = TrainConfig()
    c.batch_size = 256
    c.train_every = 2
    c.ch1, c.ch2, c.ch3, c.head_ch = 32, 64, 64, 32
    c.memory_size = 120_000
    return c


# ----------------------------------------------------------------------
# In the package the config lives in its own module and is referenced as `C.X`.
# Flattened into one cell there is no module, so bind a namespace with the same
# name over everything defined above. Keeps the code below byte-identical to
# the package version.
# ----------------------------------------------------------------------
import types as _types
C = _types.SimpleNamespace(**{_k: _v for _k, _v in list(globals().items())
                              if not _k.startswith("__")})



# ====================================================================
# ---- defenders.py ------------------------------------------------
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


import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np



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


# `base.py` imports the Trap dataclass under an alias; flattened there is no
# module boundary, so bind the alias by hand.
TrapSite = Trap


# ====================================================================
# ---- army.py -----------------------------------------------------
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


import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np



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


# ====================================================================
# ---- layout.py ---------------------------------------------------
"""
Real war-base layout generation.

The previous generator scattered buildings into concentric rings and packed the
leftovers with walls. It produced something with the right *census* -- correct
building counts, correct footprints -- and completely the wrong *structure*.
Real bases are not built by placing buildings and then adding walls. They are
built walls first: a skeleton of compartments, and then buildings assigned into
compartments by role.

This module builds the skeleton. The rules below come from base-design guides
and pro tournament layouts; sources are cited inline.

Why not just use real bases directly? Because Clash's "copy base" links are not
decodable. The payload is 24 bytes -- an owner id, a layout slot, and a
signature -- so the link is a pointer the game resolves against Supercell's
servers, not an encoding of the layout. It cannot be otherwise: a TH15 village
is ~470 placed objects on a 44x44 grid, which needs 646+ bytes for coordinates
alone. Base links also expire when the owner edits that slot, which a
self-contained encoding would not do.
    https://github.com/nschmeller/clash-bases

So instead: encode the rules.
"""


import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np



# ----------------------------------------------------------------------
# Archetypes
#
#   box        compact rectangular core, symmetric compartments. Very common
#              at TH15/16.
#   diamond    45-degree rotated outline, symmetric. Also very common, and the
#              one that produces collinear Air Defenses (see below).
#   ring       continuous annulus of compartments around a central core.
#   askew      deliberately asymmetric, offset compartments, no obvious entry.
#              This is what "anti-3-star" layouts look like.
# ----------------------------------------------------------------------
ARCHETYPES = ("box", "diamond", "ring", "askew")
ARCHETYPE_WEIGHTS = (0.30, 0.25, 0.15, 0.30)


@dataclass
class Cell:
    """One wall compartment."""
    x0: int
    y0: int
    x1: int          # exclusive
    y1: int
    ring: int = 0    # 0 = core, higher = further out
    slots: int = 0   # defenses placed so far

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h

    def dist_to_centre(self) -> float:
        return math.hypot(self.cx - 22.0, self.cy - 22.0)


# ----------------------------------------------------------------------
def _cuts(lo: int, hi: int, n: int, r: random.Random) -> List[int]:
    """Divide [lo, hi] into n compartment bands.

    Interior cells land at 4-6 tiles, which is what the 325-wall budget
    actually affords and what a 3x3 defense plus one tile of clearance needs.
        https://www.cocbaselinks.online/guides/base-compartments-guide
    """
    span = hi - lo
    base = span / n
    out = [lo]
    for i in range(1, n):
        pos = lo + int(round(base * i)) + r.choice([-1, 0, 0, 1])
        out.append(max(out[-1] + 5, min(hi - 5 * (n - i), pos)))
    out.append(hi)
    return out


def build_skeleton(r: random.Random, archetype: str) -> Tuple[np.ndarray, List[Cell]]:
    """Lay the walls first, then hand back the compartments they enclose.

    Returns a boolean wall mask and the list of compartment interiors.

    Real bases use irregular, offset compartments rather than a clean grid --
    "irregular shapes outperform symmetric squares and rectangles" and "offset
    compartments with angled walls prevent Wall Breakers from opening clean
    paths" -- so after laying the grid we merge a fraction of adjacent cells to
    vary their size, and jitter the division lines.
    """
    walls = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=bool)

    half = r.randint(15, 17)
    lo, hi = 22 - half, 22 + half
    nx = r.randint(4, 5)
    ny = r.randint(4, 5)
    if archetype == "ring":
        nx = ny = 5
    xs = _cuts(lo, hi, nx, r)
    ys = _cuts(lo, hi, ny, r)

    # every division line becomes a wall
    for x in xs:
        walls[ys[0]:ys[-1] + 1, x] = True
    for y in ys:
        walls[y, xs[0]:xs[-1] + 1] = True

    cells: List[Cell] = []
    for j in range(ny):
        for i in range(nx):
            cells.append(Cell(xs[i] + 1, ys[j] + 1, xs[i + 1], ys[j + 1]))

    # Merge some neighbours by knocking out the wall between them. This is what
    # turns a tidy grid into the varied compartment sizes a real base has.
    merges = r.randint(3, 6)
    grid_of = {(i, j): cells[j * nx + i] for j in range(ny) for i in range(nx)}
    merged: Set[Tuple[int, int]] = set()
    for _ in range(merges):
        i, j = r.randrange(nx), r.randrange(ny)
        if (i, j) in merged:
            continue
        horiz = r.random() < 0.5
        ni, nj = (i + 1, j) if horiz else (i, j + 1)
        if ni >= nx or nj >= ny or (ni, nj) in merged:
            continue
        a, b = grid_of[(i, j)], grid_of[(ni, nj)]
        if a.area + b.area > 90:
            continue
        if horiz:
            walls[a.y0:a.y1, a.x1] = False
            a.x1 = b.x1
        else:
            walls[a.y1, a.x0:a.x1] = False
            a.y1 = b.y1
        merged.add((ni, nj))
        cells.remove(b)

    # The core compartment gets a second wall layer -- "double on core".
    core = min(cells, key=lambda c: c.dist_to_centre())
    for d in (2,):
        x0, y0 = max(1, core.x0 - d), max(1, core.y0 - d)
        x1, y1 = min(C.GRID_SIZE - 2, core.x1 + d), min(C.GRID_SIZE - 2, core.y1 + d)
        walls[y0:y1 + 1, x0] = True
        walls[y0:y1 + 1, x1] = True
        walls[y0, x0:x1 + 1] = True
        walls[y1, x0:x1 + 1] = True

    if archetype == "askew":
        # nudge a few interior wall runs sideways so nothing lines up cleanly
        for _ in range(r.randint(2, 4)):
            x = r.choice(xs[1:-1])
            y0 = r.randint(ys[0], ys[-1] - 6)
            seg = r.randint(4, 8)
            d = r.choice([-1, 1])
            walls[y0:y0 + seg, x] = False
            walls[y0:y0 + seg, np.clip(x + d, 1, C.GRID_SIZE - 2)] = True

    # rank compartments by distance from the middle
    order = sorted(cells, key=lambda c: c.dist_to_centre())
    for k, c in enumerate(order):
        c.ring = 0 if k == 0 else (1 if k <= 4 else (2 if k <= 10 else 3))

    return walls, cells


# ----------------------------------------------------------------------
def open_compartments(walls: np.ndarray, cells: Sequence[Cell]) -> List[Cell]:
    """Which compartments can be walked into without breaking a wall?

    Flood-fill from the map border across every non-wall tile. Any compartment
    interior the flood reaches is "open" -- troops stroll in and the walls did
    nothing. Real bases do not have these, so this is the validity check the
    generator runs on every base it produces.
    """
    h = w = C.GRID_SIZE
    seen = np.zeros((h, w), dtype=bool)
    stack: List[Tuple[int, int]] = []
    for i in range(w):
        for (y, x) in ((0, i), (h - 1, i), (i, 0), (i, w - 1)):
            if not walls[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and not walls[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return [c for c in cells if seen[c.y0:c.y1, c.x0:c.x1].any()]


def seal(walls: np.ndarray, cells: Sequence[Cell]) -> None:
    """Close every compartment that the flood-fill got into."""
    for _ in range(4):
        bad = open_compartments(walls, cells)
        if not bad:
            return
        for c in bad:
            walls[c.y0 - 1:c.y1 + 1, c.x0 - 1] = True
            walls[c.y0 - 1:c.y1 + 1, c.x1] = True
            walls[c.y0 - 1, c.x0 - 1:c.x1 + 1] = True
            walls[c.y1, c.x0 - 1:c.x1 + 1] = True


# ----------------------------------------------------------------------
def air_defense_spots(r: random.Random, cells: Sequence[Cell],
                      archetype: str) -> Tuple[List[Tuple[int, int]], bool]:
    """Where the four Air Defenses go.

    The canonical arrangement is a DIAMOND once the fourth unlocks, placed deep
    behind the first line of Cannons and Archer Towers, never near the edge, and
    spread far enough apart that one Lightning chain cannot take two.

    And here is the detail that matters for the Giant Arrow: a PERFECT diamond
    -- north, east, south, west at equal radius -- produces exactly two
    axis-aligned collinear pairs, north-south and east-west, both running
    through the core. The textbook formation is itself the exploitable mistake.
    Good builders know this: the Giant Arrow wiki tells builders not to place
    key defenses "collinear, i.e. in a straight line", and to make bases
    "slightly unsymmetrical" with defenses "slightly askew".

    So symmetric archetypes get a clean diamond and stay exploitable; askew and
    ring archetypes get real jitter and mostly are not. That is roughly the
    50/50 split real bases show, and it is what makes the Queen's deployment a
    decision rather than a formality.
        https://clashofclans.fandom.com/wiki/Giant_Arrow
        https://clashofclans.fandom.com/wiki/Air_Defense/Home_Village
    """
    symmetric = archetype in ("box", "diamond")
    rad = r.randint(7, 10)
    ang0 = r.choice([0.0, math.pi / 4])
    spots: List[Tuple[int, int]] = []
    for k in range(4):
        a = ang0 + k * math.pi / 2
        # Even symmetric bases jitter their Air Defenses a tile or two off the
        # exact axis, precisely because a clean diamond is a free Giant Arrow.
        # Askew layouts jitter hard enough that no usable line survives.
        jit = r.randint(-2, 2) if symmetric else r.randint(-4, 4)
        ja = r.uniform(-.15, .15) if symmetric else r.uniform(-0.5, 0.5)
        x = 22 + (rad + jit) * math.cos(a + ja)
        y = 22 + (rad + jit) * math.sin(a + ja)
        spots.append((int(np.clip(x, 7, C.GRID_SIZE - 10)),
                      int(np.clip(y, 7, C.GRID_SIZE - 10))))
    return spots, symmetric


def collinear_pairs(spots: Sequence[Tuple[float, float]],
                    tol: float = 1.5) -> int:
    """How many pairs of Air Defenses sit on a line an attacker can actually
    use -- axis-aligned or 45-degree diagonal, since those are the ones you can
    set up from the deployment border."""
    n = 0
    for i in range(len(spots)):
        for j in range(i + 1, len(spots)):
            dx = abs(spots[i][0] - spots[j][0])
            dy = abs(spots[i][1] - spots[j][1])
            if dx <= tol or dy <= tol or abs(dx - dy) <= tol:
                n += 1
    return n


# ----------------------------------------------------------------------
def sweeper_facings(r: random.Random, positions: Sequence[Tuple[float, float]]
                    ) -> List[float]:
    """Air Sweeper facings, eight legal directions 45 degrees apart.

    Two documented styles: stack both on one side to make a segment nearly
    impassable to air, or face them apart for wider deterrence. Either way each
    points outward from the core, into its own quadrant.
        https://clashofclans.fandom.com/wiki/Air_Sweeper
    """
    step = 360.0 / 8
    stacked = r.random() < 0.35
    out: List[float] = []
    for i, (x, y) in enumerate(positions):
        outward = math.degrees(math.atan2(y - 22.0, x - 22.0))
        if stacked and out:
            base = out[0]
        else:
            base = outward
        snapped = round(base / step) * step
        out.append(snapped % 360.0)
    return out


import types as _types
L = _types.SimpleNamespace(
    ARCHETYPES=ARCHETYPES, ARCHETYPE_WEIGHTS=ARCHETYPE_WEIGHTS, Cell=Cell,
    build_skeleton=build_skeleton, open_compartments=open_compartments,
    seal=seal, air_defense_spots=air_defense_spots,
    collinear_pairs=collinear_pairs, sweeper_facings=sweeper_facings)



# ====================================================================
# ---- base.py -----------------------------------------------------
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


import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np



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


# ====================================================================
# ---- env.py ------------------------------------------------------
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


import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np



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


# ====================================================================
# ---- full_env.py -------------------------------------------------
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


import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


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


# ====================================================================
# ---- model.py ----------------------------------------------------
"""
The Q-network.

The old network was three convolutions with NO pooling feeding a
Linear(247808, 512). That single layer held 126.9M of its 128.0M parameters --
99.2% of the model -- which is why every checkpoint was 512 MB and why training
ran at 13 seconds per episode on CPU.

This one is a dueling, fully-convolutional design. Because the action "cast a
spell on tile (x, y)" is spatial, the Q-values for tile actions are produced by
a 1x1 convolution over a feature map: one shared set of weights evaluates every
tile. That is both far smaller and far better at generalising -- learning that
"cast where she is about to be shot" transfers across the whole map instead of
being relearned for each of 1,936 output units.

Roughly 150k parameters. About 850x smaller than the original.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class RCQNet(nn.Module):
    """Dueling, fully-convolutional Q-network.

    Parameterised over the observation and action shape so the same
    architecture serves both environments:

      * charge only   -- 11 channels, 2 scalar actions, 1 tile head (spells)
      * full attack   -- 14 channels, 3 scalar actions, 2 tile heads
                         (cast a spell here / deploy the next unit here)

    Each tile head is a 1x1 convolution over a shared feature map, so ONE set
    of weights scores all 484 tiles. "Deploy the dragons away from the sweeper
    cone" is then a spatial pattern the network can learn once and apply
    everywhere, rather than 484 independent output units each needing their own
    experience.
    """

    def __init__(self, cfg: C.TrainConfig, n_channels: int = None,
                 n_scalars: int = None, n_scalar_actions: int = 2,
                 n_tile_heads: int = 1):
        super().__init__()
        n_channels = n_channels or C.N_SPATIAL_CHANNELS
        n_scalars = n_scalars or C.N_SCALARS
        self.n_tile_heads = n_tile_heads
        ch1, ch2, ch3 = cfg.ch1, cfg.ch2, cfg.ch3
        gdim, gbc = cfg.global_dim, cfg.global_broadcast
        self.dueling = cfg.dueling
        self.gbc = gbc

        self.enc1 = nn.Conv2d(n_channels, ch1, 3, padding=1)
        self.enc2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.enc3 = nn.Conv2d(ch2, ch3, 3, padding=1)

        # global context: pooled features + the scalar state vector
        self.gfc = nn.Linear(ch3 + n_scalars, gdim)
        self.gproj = nn.Linear(gdim, gbc)

        # Spatial advantage head, evaluated at the action-grid resolution.
        # A 1x1 bottleneck first: fusing (ch2 + ch3 + gbc) channels with a 3x3
        # directly is the single most expensive op in the net, and projecting
        # down to head_ch first cuts it by ~3x for no measurable quality loss.
        self.fuse = nn.Conv2d(ch2 + ch3 + gbc, cfg.head_ch, 1)
        self.head = nn.Conv2d(cfg.head_ch, cfg.head_ch, 3, padding=1)
        self.adv_tiles = nn.Conv2d(cfg.head_ch, n_tile_heads, 1)

        # the two non-spatial actions: wait, and Seeking Shield
        self.adv_scalar = nn.Linear(gdim, n_scalar_actions)
        self.value = nn.Linear(gdim, 1)

    def forward(self, spatial: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b = spatial.size(0)
        e1 = F.relu(self.enc1(spatial))              # ch1 x 44 x 44
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2)))  # ch2 x 22 x 22  == action grid
        e3 = F.relu(self.enc3(F.max_pool2d(e2, 2)))  # ch3 x 11 x 11

        pooled = F.adaptive_avg_pool2d(e3, 1).flatten(1)          # ch3
        g = F.relu(self.gfc(torch.cat([pooled, scalars], dim=1)))  # gdim

        up3 = F.interpolate(e3, size=(C.ACTION_GRID, C.ACTION_GRID), mode="nearest")
        gb = self.gproj(g).view(b, self.gbc, 1, 1).expand(
            b, self.gbc, C.ACTION_GRID, C.ACTION_GRID)

        f = F.relu(self.fuse(torch.cat([e2, up3, gb], dim=1)))
        h = F.relu(self.head(f))
        # (b, heads, G, G) -> (b, heads*G*G), head-major so the layout matches
        # [scalar actions..., head0 tiles..., head1 tiles...]
        adv_tiles = self.adv_tiles(h).reshape(b, self.n_tile_heads, -1).reshape(b, -1)
        adv = torch.cat([self.adv_scalar(g), adv_tiles], dim=1)

        if not self.dueling:
            return adv
        v = self.value(g)
        return v + adv - adv.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def q_masked(self, spatial: torch.Tensor, scalars: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """Q-values with illegal actions driven to -inf.

        Masking replaces the old 'penalise the agent -10 for clicking with no
        spells left' hack. An action that cannot be taken should not be
        selectable, rather than something the network has to learn to avoid.
        """
        q = self.forward(spatial, scalars)
        return q.masked_fill(~mask, float("-inf"))


def build_model(cfg: C.TrainConfig, device: torch.device) -> RCQNet:
    net = RCQNet(cfg).to(device)
    return net


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# ====================================================================
# ---- replay.py ---------------------------------------------------
"""
N-step replay buffer.

Two things the old buffer got wrong:

1. It stored float64 observations in a Python deque of tuples. At 4 x 44 x 44
   that is ~62 KB per state and ~124 KB per transition, so the nominal 50,000
   capacity would have needed about 6 GB. It never actually filled.

2. It was never saved. Every resume began with an empty buffer AND a freshly
   initialised Adam optimiser, so the first few hundred episodes after each
   restart were spent refilling memory with near-random experience while the
   optimiser re-estimated its moments. Three separate 10-hour runs each paid
   that cost.

Here observations are quantised to uint8 (every channel is already in [0, 1],
so 1/255 resolution is far finer than anything the network cares about). That
is 19.4 KB per state, and the whole buffer serialises into the checkpoint.
"""


import numpy as np
from collections import deque
from typing import Deque, Optional, Tuple



def _q(obs: np.ndarray) -> np.ndarray:
    return np.clip(obs * 255.0, 0, 255).astype(np.uint8)


def _dq(obs: np.ndarray) -> np.ndarray:
    return obs.astype(np.float32) / 255.0


class NStepReplay:
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99,
                 n_channels: int = None, n_scalars: int = None,
                 n_actions: int = None):
        n_channels = n_channels or C.N_SPATIAL_CHANNELS
        n_scalars = n_scalars or C.N_SCALARS
        n_actions = n_actions or C.N_ACTIONS
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.size = 0
        self.pos = 0
        # Demonstrations are written first and then PROTECTED: the circular
        # write head wraps back to `protect`, never over them. Without this the
        # successful trajectories are evicted after a few hundred episodes and
        # the agent forgets what a win even looks like.
        self.protect = 0

        sh = (capacity, n_channels, C.GRID_SIZE, C.GRID_SIZE)
        self.s = np.zeros(sh, dtype=np.uint8)
        self.s2 = np.zeros(sh, dtype=np.uint8)
        self.sc = np.zeros((capacity, n_scalars), dtype=np.float32)
        self.sc2 = np.zeros((capacity, n_scalars), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.m2 = np.zeros((capacity, n_actions), dtype=bool)

        self._pending: Deque = deque(maxlen=n_step)

    # ------------------------------------------------------------------
    def _store(self, s, sc, a, r, s2, sc2, d, m2) -> None:
        i = self.pos
        self.s[i] = _q(s)
        self.sc[i] = sc
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = _q(s2)
        self.sc2[i] = sc2
        self.d[i] = d
        self.m2[i] = m2
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = self.protect
        self.size = min(self.size + 1, self.capacity)

    def push(self, s, sc, a, r, s2, sc2, done, mask2) -> None:
        """Accumulate n-step returns before writing.

        Multi-step returns propagate the Town Hall reward backwards n times
        faster than 1-step, which matters a lot when the payoff arrives 60+
        ticks after the decision that earned it.
        """
        self._pending.append((s, sc, a, r, s2, sc2, done, mask2))
        if len(self._pending) < self.n_step and not done:
            return

        def flush_one():
            s0, sc0, a0 = self._pending[0][0], self._pending[0][1], self._pending[0][2]
            R = 0.0
            last = None
            for k, (_, _, _, rk, s2k, sc2k, dk, m2k) in enumerate(self._pending):
                R += (self.gamma ** k) * rk
                last = (s2k, sc2k, dk, m2k)
                if dk:
                    break
            self._store(s0, sc0, a0, R, last[0], last[1], last[2], last[3])

        if done:
            while self._pending:
                flush_one()
                self._pending.popleft()
        else:
            flush_one()
            self._pending.popleft()

    def set_protected(self, n: Optional[int] = None) -> None:
        """Freeze the first `n` transitions (default: everything stored so far)."""
        n = self.size if n is None else n
        self.protect = int(min(n, self.capacity // 2))
        if self.pos < self.protect:
            self.pos = self.protect

    def sample_demo(self, batch: int):
        """Sample only from the protected demonstration region."""
        if self.protect <= 0:
            return None
        idx = np.random.randint(0, self.protect, size=batch)
        return (_dq(self.s[idx]), self.sc[idx], self.a[idx])

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        return (_dq(self.s[idx]), self.sc[idx], self.a[idx], self.r[idx],
                _dq(self.s2[idx]), self.sc2[idx], self.d[idx], self.m2[idx])

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------
    def state_dict(self, max_save: Optional[int] = None) -> dict:
        """Serialise the buffer so a resume is a true resume."""
        n = self.size if max_save is None else min(self.size, max_save)
        # keep the most recent n entries
        if self.size < self.capacity:
            sel = np.arange(max(0, self.size - n), self.size)
        else:
            sel = (self.pos - np.arange(n, 0, -1)) % self.capacity
        return dict(capacity=self.capacity, n_step=self.n_step, gamma=self.gamma,
                    s=self.s[sel], sc=self.sc[sel], a=self.a[sel], r=self.r[sel],
                    s2=self.s2[sel], sc2=self.sc2[sel], d=self.d[sel], m2=self.m2[sel])

    def load_state_dict(self, sd: dict) -> None:
        n = len(sd["a"])
        n = min(n, self.capacity)
        self.s[:n] = sd["s"][-n:]
        self.sc[:n] = sd["sc"][-n:]
        self.a[:n] = sd["a"][-n:]
        self.r[:n] = sd["r"][-n:]
        self.s2[:n] = sd["s2"][-n:]
        self.sc2[:n] = sd["sc2"][-n:]
        self.d[:n] = sd["d"][-n:]
        self.m2[:n] = sd["m2"][-n:]
        self.size = n
        self.pos = n % self.capacity

    def nbytes(self) -> int:
        return (self.s.nbytes + self.s2.nbytes + self.sc.nbytes + self.sc2.nbytes
                + self.a.nbytes + self.r.nbytes + self.d.nbytes + self.m2.nbytes)


# ====================================================================
# ---- policies.py -------------------------------------------------
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


import math
from typing import Callable

import numpy as np



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


# ====================================================================
# ---- train.py ----------------------------------------------------
"""
DQN training loop.

Fixes over the previous version, in the order they cost the most time:

* RESUME. The old loader was `for check_ep in range(3000, 400, -100)` -- it
  counts DOWN from 3000, so it could never find a checkpoint past episode 3000.
  Two separate 10-hour runs each reloaded checkpoint_3000 while checkpoint_5700
  sat on disk, re-ran the same 2,700 episodes, and overwrote each other. Here
  the loader globs and takes the true maximum, and the checkpoint carries the
  optimizer state, the step counter, epsilon, the curriculum stage and the RNG.

* EPSILON. It used to be recomputed from the absolute episode number, so
  resuming at episode 3000 with EPS_DECAY=3000 silently restarted exploration
  at 40% random. It is now decayed on gradient steps and persisted.

* LOSS. MSE on rewards spanning -2000..+5000 produced enormous gradients.
  Now: rewards on a unit scale, Huber loss, and gradient-norm clipping.

* DOUBLE DQN. Vanilla `max` in both selection and evaluation systematically
  over-estimates Q. Selection now comes from the online net, evaluation from
  the target net.

* CURRICULUM. Difficulty only advances when a greedy evaluation actually
  clears the promotion bar, so the agent is never pushed past what it can do.
"""


import csv
import glob
import json
import math
import os
import random
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class Trainer:
    def __init__(self, cfg: C.TrainConfig, out_dir: str = "runs/rc",
                 full: bool = False):
        self.cfg = cfg
        self.full = full
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.device = pick_device(cfg.device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        if full:
            shape = dict(n_channels=N_CH, n_scalars=N_SC,
                         n_scalar_actions=5, n_tile_heads=2)
        else:
            shape = dict(n_channels=C.N_SPATIAL_CHANNELS,
                         n_scalars=C.N_SCALARS, n_scalar_actions=2,
                         n_tile_heads=1)
        self.shape = shape
        self.n_actions = N_ACTIONS_FULL if full else C.N_ACTIONS
        self.policy = RCQNet(cfg, **shape).to(self.device)
        self.target = RCQNet(cfg, **shape).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.memory = NStepReplay(cfg.memory_size, cfg.n_step, cfg.gamma,
                                  n_channels=(N_CH if full else C.N_SPATIAL_CHANNELS),
                                  n_scalars=(N_SC if full else C.N_SCALARS),
                                  n_actions=(N_ACTIONS_FULL if full else C.N_ACTIONS))

        self.stage_idx = cfg.start_stage
        env_cls = FullAttackEnv if full else RCWalkEnv
        kw = self._stage_kwargs()
        if full:
            kw["max_spells"] = C.SPELL_CAPACITY
        self.env = env_cls(**kw, seed=cfg.seed)

        self.episode = 0
        self.env_steps = 0
        self.grad_steps = 0
        self.best_eval = -1e9
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    @property
    def stage(self) -> C.Stage:
        return C.CURRICULUM[min(self.stage_idx, len(C.CURRICULUM) - 1)]

    def _stage_kwargs(self) -> Dict:
        s = self.stage
        return dict(defense_frac=s.defense_frac, max_spells=s.spells,
                    traps=s.traps, cc=s.cc, hero=s.hero)

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.grad_steps / max(1, c.eps_decay_steps))
        return c.eps_start + (c.eps_end - c.eps_start) * frac

    def bc_lambda(self) -> float:
        """Weight on the supervised demonstration loss, annealed to zero."""
        c = self.cfg
        frac = min(1.0, self.grad_steps / max(1, c.bc_anneal_steps))
        return c.bc_lambda_start * (1.0 - frac)

    def teacher_prob(self) -> float:
        """How often an exploratory action comes from the scripted policy
        instead of being uniform. Decays to zero, so the final policy is the
        network's own -- the teacher only solves the cold-start problem."""
        c = self.cfg
        frac = min(1.0, self.grad_steps / max(1, c.teacher_decay_steps))
        return c.teacher_start + (c.teacher_end - c.teacher_start) * frac

    # ------------------------------------------------------------------
    def _to_t(self, spatial: np.ndarray, scalars: np.ndarray):
        s = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        sc = torch.from_numpy(scalars).unsqueeze(0).to(self.device)
        return s, sc

    def _explore(self, mask: np.ndarray) -> int:
        """Structured random exploration.

        The action space is 1 wait + 1 ability + 484 spell tiles, so sampling it
        UNIFORMLY casts a spell on 99.6% of ticks -- the entire 8-10 spell budget
        is gone within the first few seconds and the episode ends before the
        agent has seen anything useful. A uniform prior only makes sense when the
        actions are comparably common. Here they are not: a good episode is
        mostly waiting, punctuated by about eight casts.

        So: mostly wait, occasionally the ability, otherwise a tile.
        """
        r = random.random()
        if self.full:
            # deploying is a much rarer action than waiting, and casting rarer
            # still -- sample in roughly the proportion a real attack uses them
            if r < 0.55 and mask[A_WAIT]:
                return A_WAIT
            if r < 0.57 and mask[A_SHIELD]:
                return A_SHIELD
            if r < 0.585 and mask[A_TOME]:
                return A_TOME
            if r < 0.60 and mask[A_ARROW]:
                return A_ARROW
            if r < 0.61 and mask[A_DUKE]:
                return A_DUKE
            group = (A_SPELL, A_DEPLOY) if r < 0.75 else (A_DEPLOY, len(mask))
            tiles = np.flatnonzero(mask[group[0]:group[1]])
            if len(tiles):
                return int(np.random.choice(tiles)) + group[0]
            return int(np.random.choice(np.flatnonzero(mask)))
        if r < 0.60 and mask[C.ACTION_WAIT]:
            return C.ACTION_WAIT
        if r < 0.63 and mask[C.ACTION_ABILITY]:
            return C.ACTION_ABILITY
        tiles = np.flatnonzero(mask[C.ACTION_TILE_OFFSET:])
        if len(tiles):
            return int(np.random.choice(tiles)) + C.ACTION_TILE_OFFSET
        return int(np.random.choice(np.flatnonzero(mask)))

    def act(self, spatial, scalars, mask: np.ndarray, eps: float,
            env=None, t: int = 0, teacher: float = 0.0) -> int:
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return C.ACTION_WAIT
        if random.random() < eps:
            if env is not None and random.random() < teacher:
                a = (policy_full_attack if self.full else policy_scripted_human)(env, t)
                if mask[a]:
                    return int(a)
            return self._explore(mask)
        s, sc = self._to_t(spatial, scalars)
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        self.policy.eval()
        with torch.no_grad():
            q = self.policy.q_masked(s, sc, m)
        self.policy.train()
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    def optimize(self) -> Optional[float]:
        c = self.cfg
        if len(self.memory) < max(c.batch_size, c.learn_start):
            return None
        s, sc, a, r, s2, sc2, d, m2 = self.memory.sample(c.batch_size)
        dev = self.device
        s = torch.from_numpy(s).to(dev)
        sc = torch.from_numpy(sc).to(dev)
        a = torch.from_numpy(a).long().unsqueeze(1).to(dev)
        r = torch.from_numpy(r).to(dev)
        s2 = torch.from_numpy(s2).to(dev)
        sc2 = torch.from_numpy(sc2).to(dev)
        d = torch.from_numpy(d).to(dev)
        m2 = torch.from_numpy(m2).to(dev)

        q = self.policy(s, sc).gather(1, a).squeeze(1)

        with torch.no_grad():
            if c.double_dqn:
                # action chosen by the ONLINE net, value read from the TARGET net
                q_online = self.policy(s2, sc2).masked_fill(~m2, float("-inf"))
                best = q_online.argmax(dim=1, keepdim=True)
                q_next = self.target(s2, sc2).gather(1, best).squeeze(1)
            else:
                q_next = self.target(s2, sc2).masked_fill(~m2, float("-inf")).max(1)[0]
            q_next = torch.nan_to_num(q_next, neginf=0.0)
            gamma_n = c.gamma ** c.n_step
            tgt = r + gamma_n * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q, tgt, beta=c.huber_delta)

        # ---- DQfD auxiliary supervised loss ------------------------------
        # Behaviour cloning alone is not enough. After BC the greedy policy hit
        # 90% Town Hall kills, and then pure Bellman updates dragged it back to
        # 23% within 200 episodes: the cloned Q-values are trained as logits on
        # an arbitrary scale, and rescaling them to satisfy the Bellman equation
        # destroys the argmax structure before the value function is any good.
        # Keeping a decaying supervised term on the demonstration region anchors
        # the policy while the values calibrate. This is the other half of DQfD.
        lam = self.bc_lambda()
        if lam > 0:
            demo = self.memory.sample_demo(c.batch_size)
            if demo is not None:
                ds, dsc, da = demo
                ds = torch.from_numpy(ds).to(dev)
                dsc = torch.from_numpy(dsc).to(dev)
                da = torch.from_numpy(da).long().to(dev)
                loss = loss + lam * F.cross_entropy(self.policy(ds, dsc), da)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), c.grad_clip)
        self.opt.step()
        self.grad_steps += 1

        if self.grad_steps % max(1, c.target_update_steps // c.train_every) == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------------
    def run_episode(self, greedy: bool = False, record: bool = False) -> Dict:
        env = self.env
        env.record = record
        spatial, scalars = self._reset_env()
        eps = 0.0 if greedy else self.epsilon()
        total = 0.0
        losses: List[float] = []

        # ---- deployment: a real decision, and it gets a real gradient ----
        # This is the action the user asked for: she no longer spawns on a
        # random edge. The transition is pushed to the buffer like any other,
        # with reward 0, so the deploy choice is credited entirely through
        # bootstrapped value -- "was this a good place to start the walk?"
        teach = 0.0 if greedy else self.teacher_prob()
        mask = env.legal_actions()
        a = self.act(spatial, scalars, mask, eps, env, -1, teach)
        d_s, d_sc, d_a = spatial, scalars, a
        (spatial, scalars), r, done, info = env.step(a)
        if not greedy:
            self.memory.push(d_s, d_sc, d_a, 0.0, spatial, scalars,
                             0.0, env.legal_actions())

        prev_s, prev_sc = spatial, scalars
        t_step = 0
        while True:
            mask = env.legal_actions()
            a = self.act(prev_s, prev_sc, mask, eps, env, t_step, teach)
            t_step += 1
            (ns, nsc), r, done, info = env.step(a)
            total += r
            if not greedy:
                nmask = env.legal_actions()
                self.memory.push(prev_s, prev_sc, a, r, ns, nsc, float(done), nmask)
                self.env_steps += 1
                if self.env_steps % self.cfg.train_every == 0:
                    L = self.optimize()
                    if L is not None:
                        losses.append(L)
            prev_s, prev_sc = ns, nsc
            if done:
                break

        out = dict(
            ret=total,
            th=int(env.th_destroyed),
            stars=stars(env.buildings, env.th_destroyed),
            dest=destruction_percent(env.buildings),
            steps=env.steps,
            hp=env.rc_hp / C.RC_MAX_HP,
            spells=env.stats.get("spells_used", 0),
            loss=float(np.mean(losses)) if losses else float("nan"),
            end=info.get("end", "?"),
        )
        return out

    # ------------------------------------------------------------------
    def _reset_env(self):
        kw = self._stage_kwargs()
        if self.full:
            kw["max_spells"] = C.SPELL_CAPACITY
        obs = self.env.reset(**kw)
        if self.full:
            self.env._side_cache = None
        return obs

    @property
    def teacher(self):
        return policy_full_attack if self.full else policy_scripted_human

    def seed_demonstrations(self, n: int) -> None:
        """Fill the replay buffer with scripted-policy episodes before training.

        Without this the buffer contains only failures: uniformly random play
        essentially never destroys a Town Hall, and a Q-function cannot learn to
        reach an outcome it has never observed. That is exactly the trap the
        previous project fell into -- except there it was unavoidable, because
        the task itself was impossible.
        """
        if n <= 0:
            return
        env = self.env
        teach = self.teacher
        wins = 0
        stars_sum = 0.0
        for _ in range(n):
            spatial, scalars = self._reset_env()
            a = teach(env, -1)
            (ns, nsc), r, done, _ = env.step(a)
            self.memory.push(spatial, scalars, a, 0.0, ns, nsc, 0.0,
                             env.legal_actions())
            prev_s, prev_sc = ns, nsc
            for t in range(C.MAX_STEPS):
                a = teach(env, t)
                if not env.legal_actions()[a]:
                    a = 0
                (ns, nsc), r, done, _ = env.step(a)
                self.memory.push(prev_s, prev_sc, a, r, ns, nsc, float(done),
                                 env.legal_actions())
                prev_s, prev_sc = ns, nsc
                if done:
                    break
            wins += int(env.th_destroyed)
            stars_sum += stars(env.buildings, env.th_destroyed)
        print(f"seeded {n} demonstration episodes -> {len(self.memory)} "
              f"transitions | {100*wins/n:.0f}% destroyed the Town Hall | "
              f"{stars_sum/n:.2f} stars", flush=True)

    def pretrain_bc(self, steps: int) -> None:
        """Behaviour-clone the scripted policy before reinforcement learning.

        Seeding the buffer alone was not enough: with epsilon still high the
        network's own greedy policy stayed worse than the teacher, so as epsilon
        decayed the behaviour policy got WORSE, not better -- Town Hall kills
        fell from 38% to 2% over 400 episodes. Off-policy DQN can learn from
        demonstrations in principle, but it is slow and unstable to do it from
        the Bellman loss alone.

        So: first make argmax(Q) agree with the demonstrator (cross-entropy on
        the Q-values as logits), THEN let the Bellman updates improve on it.
        This is the cheap half of DQfD, and it also fixes the deployment
        decision, which otherwise gets one training sample per episode.
        """
        if steps <= 0 or len(self.memory) < self.cfg.batch_size:
            return
        dev = self.device
        self.policy.train()
        losses = []
        for i in range(steps):
            s, sc, a, _r, _s2, _sc2, _d, _m2 = self.memory.sample(self.cfg.batch_size)
            s = torch.from_numpy(s).to(dev)
            sc = torch.from_numpy(sc).to(dev)
            a = torch.from_numpy(a).long().to(dev)
            logits = self.policy(s, sc)
            loss = F.cross_entropy(logits, a)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
            self.opt.step()
            losses.append(float(loss.item()))
            if (i + 1) % max(1, steps // 5) == 0:
                print(f"  bc {i+1}/{steps}  loss {np.mean(losses[-200:]):.4f}", flush=True)
        self.target.load_state_dict(self.policy.state_dict())
        # the BC phase used Adam on a different objective; reset its moments so
        # the first Bellman updates are not fighting stale momentum
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        ev = self.evaluate(20)
        print(f"  after BC: greedy TH {100*ev['win']:.0f}% | ret {ev['ret']:.2f}",
              flush=True)

    def evaluate(self, n: Optional[int] = None) -> Dict:
        n = n or self.cfg.eval_episodes
        rs = [self.run_episode(greedy=True) for _ in range(n)]
        return dict(
            win=float(np.mean([r["th"] for r in rs])),
            stars=float(np.mean([r["stars"] for r in rs])),
            dest=float(np.mean([r["dest"] for r in rs])),
            ret=float(np.mean([r["ret"] for r in rs])),
            steps=float(np.mean([r["steps"] for r in rs])),
        )

    # ------------------------------------------------------------------
    def checkpoint_path(self, tag: str) -> str:
        return os.path.join(self.out_dir, f"ckpt_{tag}.pt")

    def save(self, tag: str, save_replay: int = 0) -> str:
        blob = dict(
            policy=self.policy.state_dict(),
            target=self.target.state_dict(),
            optimizer=self.opt.state_dict(),
            episode=self.episode,
            env_steps=self.env_steps,
            grad_steps=self.grad_steps,
            stage_idx=self.stage_idx,
            best_eval=self.best_eval,
            cfg=self.cfg.to_dict(),
            torch_rng=torch.get_rng_state(),
            np_rng=np.random.get_state(),
            py_rng=random.getstate(),
        )
        if save_replay > 0:
            blob["replay"] = self.memory.state_dict(max_save=save_replay)
        path = self.checkpoint_path(tag)
        torch.save(blob, path)
        torch.save(blob, os.path.join(self.out_dir, "latest.pt"))
        return path

    def load(self, path: Optional[str] = None) -> bool:
        """Resume from the LATEST checkpoint. Note the `max` -- this is the bug
        that cost the previous project two full overnight runs."""
        if path is None:
            cands = glob.glob(os.path.join(self.out_dir, "ckpt_*.pt"))
            latest = os.path.join(self.out_dir, "latest.pt")
            if os.path.exists(latest):
                path = latest
            elif cands:
                def ep_of(p):
                    try:
                        return int(os.path.basename(p).split("_")[1].split(".")[0])
                    except Exception:
                        return -1
                path = max(cands, key=ep_of)
            else:
                return False
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(blob["policy"])
        self.target.load_state_dict(blob["target"])
        self.opt.load_state_dict(blob["optimizer"])
        self.episode = blob["episode"]
        self.env_steps = blob["env_steps"]
        self.grad_steps = blob["grad_steps"]
        self.stage_idx = blob["stage_idx"]
        self.best_eval = blob.get("best_eval", -1e9)
        if "replay" in blob:
            self.memory.load_state_dict(blob["replay"])
        try:
            torch.set_rng_state(blob["torch_rng"])
            np.random.set_state(blob["np_rng"])
            random.setstate(blob["py_rng"])
        except Exception:
            pass
        print(f"resumed from {path}: episode {self.episode}, "
              f"grad_steps {self.grad_steps}, stage {self.stage.name}, "
              f"eps {self.epsilon():.3f}, replay {len(self.memory)}", flush=True)
        return True

    # ------------------------------------------------------------------
    def train(self, save_replay: int = 0) -> None:
        c = self.cfg
        t0 = time.time()
        deadline = t0 + c.max_hours * 3600
        csv_path = os.path.join(self.out_dir, "metrics.csv")
        new_file = not os.path.exists(csv_path)
        fcsv = open(csv_path, "a", newline="")
        writer = csv.writer(fcsv)
        if new_file:
            writer.writerow(["episode", "env_steps", "stage", "eps", "ret", "th",
                             "stars", "dest", "steps", "loss", "eval_win",
                             "eval_stars", "eval_dest", "wall_s"])

        print(f"device={self.device}  mode={'FULL ATTACK' if self.full else 'charge only'}  "
              f"params={count_parameters(self.policy):,}  "
              f"actions={self.n_actions}  replay={self.memory.nbytes()/1e9:.2f} GB",
              flush=True)
        print(f"starting at episode {self.episode}, stage {self.stage.name}", flush=True)

        if self.episode == 0 and len(self.memory) == 0:
            self.seed_demonstrations(c.demo_episodes)
            self.memory.set_protected()
            self.pretrain_bc(c.bc_steps)

        window: List[Dict] = []
        while self.episode < c.max_episodes and time.time() < deadline:
            self.episode += 1
            res = self.run_episode()
            window.append(res)
            if len(window) > 100:
                window.pop(0)

            if self.episode % c.log_every == 0:
                w = window[-c.log_every:]
                finite = [x['loss'] for x in w if np.isfinite(x['loss'])]
                mean_loss = float(np.mean(finite)) if finite else float('nan')
                rem = deadline - time.time()
                print(
                    f"ep {self.episode:6d} | stage {self.stage.name:14s} "
                    f"| eps {self.epsilon():.3f} | tch {self.teacher_prob():.2f} "
                    f"| ret {np.mean([x['ret'] for x in w]):7.2f} "
                    f"| TH {100*np.mean([x['th'] for x in w]):5.1f}% "
                    f"| dest {100*np.mean([x['dest'] for x in w]):4.1f}% "
                    f"| len {np.mean([x['steps'] for x in w]):5.1f} "
                    f"| loss {mean_loss:.4f} "
                    f"| {rem/3600:.2f}h left",
                    flush=True)
                writer.writerow([self.episode, self.env_steps, self.stage.name,
                                 f"{self.epsilon():.4f}",
                                 f"{np.mean([x['ret'] for x in w]):.3f}",
                                 f"{np.mean([x['th'] for x in w]):.3f}",
                                 f"{np.mean([x['stars'] for x in w]):.3f}",
                                 f"{np.mean([x['dest'] for x in w]):.4f}",
                                 f"{np.mean([x['steps'] for x in w]):.1f}",
                                 f"{mean_loss:.5f}",
                                 "", "", "", f"{time.time()-t0:.0f}"])
                fcsv.flush()

            if self.episode % c.eval_every == 0:
                ev = self.evaluate()
                print(f"  EVAL @ {self.episode}: TH {100*ev['win']:.1f}% | "
                      f"stars {ev['stars']:.2f} | dest {100*ev['dest']:.1f}% | "
                      f"ret {ev['ret']:.2f}  [stage {self.stage.name}]", flush=True)
                writer.writerow([self.episode, self.env_steps, self.stage.name,
                                 f"{self.epsilon():.4f}", "", "", "", "", "", "",
                                 f"{ev['win']:.4f}", f"{ev['stars']:.4f}",
                                 f"{ev['dest']:.4f}", f"{time.time()-t0:.0f}"])
                fcsv.flush()

                if ev["ret"] > self.best_eval:
                    self.best_eval = ev["ret"]
                    self.save("best")

                if (c.curriculum and self.stage_idx < len(C.CURRICULUM) - 1
                        and ev["win"] >= self.stage.promote_win_rate):
                    self.stage_idx += 1
                    print(f"  ==> PROMOTED to stage {self.stage.name} "
                          f"(defense_frac={self.stage.defense_frac}, "
                          f"spells={self.stage.spells})", flush=True)

            if self.episode % c.checkpoint_every == 0:
                self.save(str(self.episode), save_replay=save_replay)

        self.save("final", save_replay=save_replay)
        fcsv.close()
        print(f"done. episodes={self.episode} grad_steps={self.grad_steps} "
              f"wall={(time.time()-t0)/3600:.2f}h", flush=True)


# ----------------------------------------------------------------------


# ====================================================================
# ---- evaluate.py -------------------------------------------------
"""
Evaluation and baselines.

The previous project had no baseline to compare against, which is how 8,500
episodes of zero wins went unnoticed for 30 hours. Every number the trained
agent produces here is printed next to three reference policies:

  random          -- uniform over legal actions
  never-cast      -- deploy, then never spend a spell (measures the raw hero)
  scripted-human  -- deploy nearest the Town Hall, keep her invisible whenever
                     she is actually standing in a threatened tile, pop Seeking
                     Shield below 35% HP

If a trained model does not beat scripted-human, it has not learned anything
worth keeping. Run this before and after any long training session.
"""


import math
import statistics
from typing import Callable, Dict, List, Optional

import numpy as np
import torch





def make_model_policy(net: RCQNet, device: torch.device) -> Callable:
    def policy(env: RCWalkEnv, t: int) -> int:
        spatial, scalars = env._obs()
        s = torch.from_numpy(spatial).unsqueeze(0).to(device)
        sc = torch.from_numpy(scalars).unsqueeze(0).to(device)
        m = torch.from_numpy(env.legal_actions()).unsqueeze(0).to(device)
        with torch.no_grad():
            q = net.q_masked(s, sc, m)
        return int(q.argmax(dim=1).item())
    return policy


# ----------------------------------------------------------------------
def run_policy(policy: Callable, episodes: int = 100, defense_frac: float = 1.0,
               spells: int = C.MAX_SPELLS, seed: int = 0,
               record_best: bool = False) -> Dict:
    env = RCWalkEnv(defense_frac=defense_frac, max_spells=spells, seed=seed)
    rets, ths, sts, dests, lens, hps = [], [], [], [], [], []
    keys, defs_, killed = [], [], {}
    best_frames, best_score = None, -1e9
    for ep in range(episodes):
        env.record = record_best
        env.reset()
        total = 0.0
        env.step(policy(env, -1))
        for t in range(C.MAX_STEPS):
            _, r, done, _ = env.step(policy(env, t))
            total += r
            if done:
                break
        rets.append(total)
        ths.append(int(env.th_destroyed))
        sts.append(stars(env.buildings, env.th_destroyed))
        dests.append(destruction_percent(env.buildings))
        lens.append(env.steps)
        hps.append(env.rc_hp / C.RC_MAX_HP)
        keys.append(env.stats["key_kills"])
        defs_.append(env.stats["defense_kills"])
        for b in env.buildings:
            if b.is_dead and b.is_defense:
                killed[b.name] = killed.get(b.name, 0) + 1
        if record_best and total > best_score:
            best_score, best_frames = total, list(env.frames)
    out = dict(
        episodes=episodes,
        th_kill_rate=float(np.mean(ths)),
        stars=float(np.mean(sts)),
        three_star=float(np.mean([s == 3 for s in sts])),
        destruction=float(np.mean(dests)),
        ret=float(np.mean(rets)),
        ret_std=float(np.std(rets)),
        length=float(np.mean(lens)),
        hp_left=float(np.mean(hps)),
        key_kills=float(np.mean(keys)),
        defense_kills=float(np.mean(defs_)),
        killed={k: v / episodes for k, v in
                sorted(killed.items(), key=lambda kv: -kv[1])},
    )
    if record_best:
        out["frames"] = best_frames
    return out


def _fmt(name: str, r: Dict) -> str:
    return (f"{name:<20s} TH {100*r['th_kill_rate']:5.1f}%  "
            f"stars {r['stars']:4.2f}  dest {100*r['destruction']:5.1f}%  "
            f"return {r['ret']:7.2f}  "
            f"KEY defenses {r['key_kills']:4.1f}  all def {r['defense_kills']:4.1f}")


def compare(model_path: Optional[str] = None, episodes: int = 100,
            defense_frac: float = 1.0, spells: int = C.MAX_SPELLS,
            device: str = "auto") -> Dict[str, Dict]:
    print(f"\n=== EVALUATION  (defense_frac={defense_frac}, spells={spells}, "
          f"{episodes} episodes each) ===")
    results: Dict[str, Dict] = {}
    for name, pol in [("random", policy_random),
                      ("never-cast", policy_never_cast),
                      ("scripted-human", policy_scripted_human)]:
        results[name] = run_policy(pol, episodes, defense_frac, spells, seed=1234)
        print(_fmt(name, results[name]))

    if model_path:
        dev = pick_device(device)
        blob = torch.load(model_path, map_location=dev, weights_only=False)
        cfg = C.TrainConfig(**{k: v for k, v in blob["cfg"].items()
                               if k in C.TrainConfig.__dataclass_fields__})
        net = RCQNet(cfg).to(dev)
        net.load_state_dict(blob["policy"])
        net.eval()
        results["dqn"] = run_policy(make_model_policy(net, dev), episodes,
                                    defense_frac, spells, seed=1234)
        print(_fmt("dqn (trained)", results["dqn"]))
        h = results["scripted-human"]["ret"]
        d = results["dqn"]["ret"]
        verdict = "BEATS" if d > h else "does NOT beat"
        print(f"\n  -> the trained agent {verdict} the scripted baseline "
              f"({d:.2f} vs {h:.2f} mean return)")
        print(f"\n  key defenses destroyed per attack "
              f"(profile: {C.TARGET_PROFILE}):")
        for name in ("Town Hall", "Air Defense", "Air Sweeper", "Monolith",
                     "Scattershot", "Eagle Artillery", "Inferno Tower"):
            kd = results["dqn"]["killed"].get(name, 0.0)
            kh = results["scripted-human"]["killed"].get(name, 0.0)
            print(f"     {name:<16s} dqn {kd:4.2f}   scripted {kh:4.2f}")
    return results

# ======================================================================
# FULL ATTACK evaluation: RC charge -> funnel -> mass dragons
#
# This is the one that answers "how many stars do I get". Unlike the solo
# charge, three stars is genuinely reachable here, because Dragons target the
# NEAREST BUILDING and therefore clear collectors and storages too.
# ======================================================================


def policy_random_full(env, t):
    return int(np.random.choice(np.flatnonzero(env.legal_actions())))


def policy_dragons_only(env, t):
    """No charge at all: dump the whole army immediately. Measures how much the
    Royal Champion charge is actually worth."""
    if not env.rc_deployed:
        return A_DEPLOY + int(np.flatnonzero(env.legal_actions()[A_DEPLOY:])[0])
    if env.queue:
        legal = np.flatnonzero(env.legal_actions()[A_DEPLOY:])
        if len(legal):
            return A_DEPLOY + int(legal[len(legal) // 2])
    return 0


def run_full(policy, episodes=60, defense_frac=1.0, seed=0, record_best=False):
    env = FullAttackEnv(defense_frac=defense_frac, max_spells=C.SPELL_CAPACITY,
                        seed=seed)
    st, pc, th, rets, keys, ads, three = [], [], [], [], [], [], []
    best_frames, best = None, -1e9
    for ep in range(episodes):
        env.record = record_best
        env.reset()
        env._side_cache = None
        total = 0.0
        env.step(policy(env, -1))
        for t in range(C.MAX_STEPS):
            _, r, done, _ = env.step(policy(env, t))
            total += r
            if done:
                break
        s = stars(env.buildings, env.th_destroyed)
        st.append(s); three.append(s == 3)
        pc.append(destruction_percent(env.buildings))
        th.append(env.th_destroyed); rets.append(total)
        keys.append(env.stats["key_kills"]); ads.append(env.stats["ad_killed"])
        if record_best and total > best:
            best, best_frames = total, list(env.frames)
    out = dict(stars=float(np.mean(st)), destruction=float(np.mean(pc)),
               th=float(np.mean(th)), ret=float(np.mean(rets)),
               three_star=float(np.mean(three)),
               key_kills=float(np.mean(keys)), ad_killed=float(np.mean(ads)),
               star_split={k: float(np.mean([x == k for x in st])) for k in (0, 1, 2, 3)})
    if record_best:
        out["frames"] = best_frames
    return out


def compare_full(model_path=None, episodes=60, defense_frac=1.0, device="auto"):
    print(f"\n=== FULL ATTACK  (RC charge + 14 dragons, {episodes} attacks each) ===")
    res = {}
    for name, pol in [("random", policy_random_full),
                      ("dragons only (no charge)", policy_dragons_only),
                      ("scripted full attack", policy_full_attack)]:
        res[name] = run_full(pol, episodes, defense_frac, seed=1234)
        r = res[name]
        print(f"{name:<26s} {r['stars']:4.2f} stars  {100*r['destruction']:5.1f}% dest  "
              f"TH {100*r['th']:5.1f}%  3star {100*r['three_star']:4.1f}%  "
              f"key def {r['key_kills']:4.1f}  AD/sweeper {r['ad_killed']:4.1f}")
    if model_path:
        dev = pick_device(device)
        blob = torch.load(model_path, map_location=dev, weights_only=False)
        cfg = C.TrainConfig(**{k: v for k, v in blob["cfg"].items()
                               if k in C.TrainConfig.__dataclass_fields__})
        net = RCQNet(cfg, n_channels=N_CH, n_scalars=N_SC,
                     n_scalar_actions=5, n_tile_heads=2).to(dev)
        net.load_state_dict(blob["policy"]); net.eval()
        res["dqn"] = run_full(make_model_policy(net, dev), episodes,
                              defense_frac, seed=1234)
        r = res["dqn"]
        print(f"{'dqn (trained)':<26s} {r['stars']:4.2f} stars  "
              f"{100*r['destruction']:5.1f}% dest  TH {100*r['th']:5.1f}%  "
              f"3star {100*r['three_star']:4.1f}%  key def {r['key_kills']:4.1f}  "
              f"AD/sweeper {r['ad_killed']:4.1f}")
        print(f"\n  star split (dqn): " + "  ".join(
            f"{k}*={100*v:.0f}%" for k, v in r["star_split"].items()))
    return res


# ====================================================================
# ---- sprites.py --------------------------------------------------
"""
The sprite reference library: what every building actually looks like, at every
level that matters at Town Hall 15.

Why this module exists
----------------------
`render_synthetic()` used to draw each building as a flat coloured box. That is
fine for testing the geometry and the training loop, and useless for training a
classifier that has to work on real screenshots -- a flat teal hexagon and an
Air Defense share no pixels. A network trained on boxes learns "teal-ish blob in
the middle third" and then meets Clash's actual art, which is textured, shaded,
partly transparent, casts shadows, and overlaps its neighbours.

So the synthetic renderer needs the real art. This module is the manifest of it:
for each class in `vision.CLASSES`, the levels worth distinguishing and the file
that shows each one.

Where the art comes from
------------------------
The Clash of Clans wiki hosts a render of every building at every level. The
naming is not uniform -- it accreted over a decade -- so the manifest below
records the exact file name for each entry rather than deriving it:

    most buildings      Cannon21.png,  Air_Defense13.png,  Wall16.png
    Inferno Tower       Inferno_Tower_Single9.png / Inferno_Tower_Multi9.png
    X-Bow               "X-Bow10 Ground.png"      (space-separated, mode suffix)
    Spell Tower         "Spell Tower3 Poison.png" (no underscore in the name!)

Those four conventions were established by probing the wiki's file namespace
(Special:PrefixIndex, namespace 6) rather than guessed; several plausible
patterns -- Air_Defence_level13_info.png among them -- do not exist.

Which levels
------------
Only the ones a TH15 attacker actually meets. A maxed TH15 shows its caps, and
a near-max war base shows one or two below, so each defense carries its cap plus
one or two down. The four classes in `vision.LEVEL_CLASSES` carry a wider span,
because for those the level changes the plan: an Inferno's ramp, an Air
Defense's DPS, a Scattershot's splash, a Monolith's percentage damage.

The happy accident that makes level classification tractable is that Supercell
recolours rather than re-textures at the top end. Air Defense 11 is vivid teal,
12 emerald green, 13 purple -- a colour-histogram difference, not a subtle one.
    https://clashofclans.fandom.com/wiki/Air_Defense/Home_Village

Inferno Towers carry BOTH modes at each level, because both appear on real
bases and they look different -- single-target has one long beam barrel, multi
has three short ones. The mode matters to the plan too (multi shreds a dragon
pack, single melts the Warden), so it is worth keeping them apart rather than
averaging them into one "Inferno" appearance.
"""


import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# (class name, level, wiki file name)
#
# `level` is the in-game level. For classes where we do not care about the
# level it is still recorded, so the renderer can pick a plausible one and the
# label writer can ignore it.
# ----------------------------------------------------------------------
MANIFEST: List[Tuple[str, int, str]] = []


def _add(cls: str, levels, pattern: str) -> None:
    for L in levels:
        MANIFEST.append((cls, L, pattern.format(L=L)))


# --- the four classes whose LEVEL changes the attack plan ---------------
_add("Air Defense",   [9, 10, 11, 12, 13],  "Air_Defense{L}.png")
_add("Inferno Tower", [6, 7, 8, 9],         "Inferno_Tower_Single{L}.png")
_add("Inferno Tower", [6, 7, 8, 9],         "Inferno_Tower_Multi{L}.png")
_add("Scattershot",   [1, 2, 3],            "Scattershot{L}.png")
_add("Monolith",      [1, 2],               "Monolith{L}.png")

# --- the rest of the defenses, at and just below the TH15 cap ----------
_add("Cannon",          [19, 20, 21], "Cannon{L}.png")
_add("Archer Tower",    [19, 20, 21], "Archer_Tower{L}.png")
_add("Mortar",          [13, 14, 15], "Mortar{L}.png")
_add("Wizard Tower",    [13, 14, 15], "Wizard_Tower{L}.png")
_add("Air Sweeper",     [6, 7],       "Air_Sweeper{L}.png")
_add("Hidden Tesla",    [11, 12, 13], "Hidden_Tesla{L}.png")
_add("Bomb Tower",      [8, 9, 10],   "Bomb_Tower{L}.png")
_add("X-Bow",           [8, 9, 10],   "X-Bow{L} Ground.png")
_add("Eagle Artillery", [4, 5],       "Eagle_Artillery{L}.png")
_add("Town Hall",       [13, 14, 15], "Town_Hall{L}.png")
_add("Clan Castle",     [9, 10, 11],  "Clan_Castle{L}.png")
_add("Wall",            [14, 15, 16], "Wall{L}.png")

# Spell Tower shows which spell it is loaded with, and all three appear.
MANIFEST += [
    ("Spell Tower", 2, "Spell Tower2 Poison.png"),
    ("Spell Tower", 3, "Spell Tower3 Poison.png"),
    ("Spell Tower", 3, "Spell Tower3 Rage.png"),
    ("Spell Tower", 3, "Spell Tower3 Invisibility.png"),
]

# --- everything that is not a defense ---------------------------------
# These matter more than they look. Dragons target the NEAREST BUILDING, so
# collectors and storages are what the funnel is made of and what the last 6%
# of destruction is hiding in. The classifier only needs to know "not a
# defense", but it has to know that reliably.
for _cls, _lvl, _pat in [
    ("Non-Defense", 15, "Elixir_Collector{L}.png"),
    ("Non-Defense", 15, "Gold_Mine{L}.png"),
    ("Non-Defense", 9,  "Dark_Elixir_Drill{L}.png"),
    ("Non-Defense", 16, "Elixir_Storage{L}.png"),
    ("Non-Defense", 16, "Gold_Storage{L}.png"),
    ("Non-Defense", 10, "Dark_Elixir_Storage{L}.png"),
    ("Non-Defense", 12, "Army_Camp{L}.png"),
    ("Non-Defense", 16, "Barracks{L}.png"),
    ("Non-Defense", 10, "Dark_Barracks{L}.png"),
    ("Non-Defense", 13, "Laboratory{L}.png"),
    ("Non-Defense", 7,  "Spell_Factory{L}.png"),
    ("Non-Defense", 6,  "Dark_Spell_Factory{L}.png"),
    ("Non-Defense", 7,  "Workshop{L}.png"),
    ("Non-Defense", 8,  "Pet_House{L}.png"),
    ("Non-Defense", 7,  "Blacksmith{L}.png"),
]:
    MANIFEST.append((_cls, _lvl, _pat.format(L=_lvl)))


def slug(fname: str) -> str:
    """File name on disk for a captured sprite."""
    return fname.replace(".png", "").replace(" ", "_").replace("'", "") + ".png"


def grids(per_grid: int = 32) -> List[List[Tuple[str, int, str]]]:
    """Split the manifest into screen-sized capture batches."""
    return [MANIFEST[i:i + per_grid] for i in range(0, len(MANIFEST), per_grid)]


# ----------------------------------------------------------------------
# Loading the captured library
# ----------------------------------------------------------------------
SPRITE_DIR_DEFAULT = "vision_data/sprites"


def variant(fname: str) -> str:
    """The sub-appearance within a level, where one exists.

    Two buildings show a mode you can read off the art and that changes how you
    attack them: an Inferno Tower's single- vs multi-target barrels, and which
    spell a Spell Tower is loaded with. Everything else returns "".
    """
    if "Inferno_Tower_Single" in fname:
        return "S"
    if "Inferno_Tower_Multi" in fname:
        return "M"
    for s in ("Poison", "Rage", "Invisibility"):
        if s in fname:
            return s[0]
    return ""


def tag(cls: str, level: int, fname: str) -> str:
    """Stable label string for a (class, level, variant) appearance."""
    return f"{cls}:{level}{variant(fname)}"


class Sprite:
    """One captured appearance, ready to composite.

    `img` is straight-alpha float32 RGBA in [0,1]. `box` is the bounding box of
    the SOLID part -- everything outside it is drop shadow and glow. The
    renderer scales and anchors on `box`, never on the full image, because the
    shadow extends further on some buildings than others and anchoring on it
    would make those sit visibly high.
    """
    __slots__ = ("img", "cls", "level", "src", "box", "tag")

    def __init__(self, img, cls, level, src, box):
        self.img = img
        self.cls = cls
        self.level = level
        self.src = src
        self.box = box
        self.tag = tag(cls, level, src)

    @property
    def solid_w(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def solid_h(self) -> int:
        return self.box[3] - self.box[1]


class SpriteLibrary:
    """The captured art, indexed by class and by (class, level).

    Held as float32 in [0,1] with straight alpha, because compositing several
    hundred buildings per rendered base is the inner loop of dataset generation
    and converting on every paste would dominate it.
    """

    def __init__(self, directory: str = SPRITE_DIR_DEFAULT):
        self.dir = directory
        self.by_key: Dict[Tuple[str, int], List[Sprite]] = {}
        self.by_class: Dict[str, List[Sprite]] = {}
        self.meta: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        index = os.path.join(self.dir, "index.json")
        if not os.path.isfile(index):
            return
        from PIL import Image
        self.meta = json.load(open(index, encoding="utf-8"))
        for fname, rec in sorted(self.meta.items()):
            path = os.path.join(self.dir, fname)
            if not os.path.isfile(path):
                continue
            arr = np.asarray(Image.open(path).convert("RGBA"),
                             dtype=np.float32) / 255.0
            box = rec.get("box") or [0, 0, arr.shape[1], arr.shape[0]]
            sp = Sprite(arr, rec["cls"], int(rec["level"]),
                        rec.get("src", fname), tuple(box))
            self.by_key.setdefault((sp.cls, sp.level), []).append(sp)
            self.by_class.setdefault(sp.cls, []).append(sp)

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_class.values())

    @property
    def ok(self) -> bool:
        return len(self) > 0

    def pick(self, cls: str, rng) -> Optional[Sprite]:
        """A uniformly random appearance of this class.

        Uniform on purpose. Real war bases are nearly all max level, but a
        classifier trained on that distribution never learns to recognise
        anything else, and the one time it meets a level 11 Air Defense it has
        to be right. The simulator can weight levels realistically; the training
        data should not.
        """
        pool = self.by_class.get(cls)
        if not pool:
            return None
        return pool[rng.randrange(len(pool))]

    def get(self, cls: str, level: Optional[int], rng) -> Optional[Sprite]:
        """A sprite for this class, preferring the exact level, then the
        nearest one that exists, then any appearance of the class."""
        if level is not None:
            hit = self.by_key.get((cls, level))
            if hit:
                return hit[rng.randrange(len(hit))]
            lv = [k[1] for k in self.by_key if k[0] == cls]
            if lv:
                near = min(lv, key=lambda v: abs(v - level))
                hit = self.by_key[(cls, near)]
                return hit[rng.randrange(len(hit))]
        return self.pick(cls, rng)

    def levels(self, cls: str) -> List[int]:
        return sorted({k[1] for k in self.by_key if k[0] == cls})

    def tags(self) -> List[str]:
        return sorted({s.tag for v in self.by_class.values() for s in v})


import types as _types
S = _types.SimpleNamespace(
    MANIFEST=MANIFEST, SPRITE_DIR_DEFAULT=SPRITE_DIR_DEFAULT, slug=slug,
    grids=grids, tag=tag, variant=variant, Sprite=Sprite,
    SpriteLibrary=SpriteLibrary)



# ====================================================================
# ---- vision.py ---------------------------------------------------
"""
Reading a real base off the screen.

Pipeline:

    screenshot -> isometric calibration -> tile crops -> classifier
              -> 44x44 building grid -> FullAttackEnv -> the trained agent
              -> a plan you execute yourself

WHAT THIS DOES NOT DO: it does not touch your input. No taps, no clicks, no
automation. Supercell's terms prohibit third-party software that automates
gameplay, and an account that does it gets banned. This reads a screenshot and
tells you what to do; you play the attack.

--------------------------------------------------------------------------
THE HONEST STATE OF THIS MODULE
--------------------------------------------------------------------------
The geometry is exact and tested. The classifier is trained on SYNTHETIC
renders from the simulator, which means it works on those and has never seen a
real Clash screenshot. Real game art -- textures, lighting, animation, the
attack UI overlay, troops standing on top of buildings -- is a large domain gap.

To make it work on your screen you need to supply real screenshots. Put them in
`vision_data/` and run `label` then `finetune`. Everything is wired for that;
the only missing ingredient is images, which I have no way to obtain.

Do not trust the synthetic-only classifier on a real attack. It will be
confidently wrong, which is worse than useless.
"""


import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Building classes the recogniser distinguishes. Ordered, index == label.
CLASSES = [
    "Empty", "Wall", "Town Hall", "Air Defense", "Air Sweeper", "Eagle Artillery",
    "Monolith", "Scattershot", "Inferno Tower", "X-Bow", "Archer Tower", "Cannon",
    "Mortar", "Wizard Tower", "Hidden Tesla", "Bomb Tower", "Spell Tower",
    "Clan Castle", "Non-Defense",
]
CLASS_INDEX = {n: i for i, n in enumerate(CLASSES)}

# Levels only matter where they change the plan. Everything else is assumed max,
# which in TH15 war is true nearly always -- and is what the simulator models.
#   Inferno: single vs multi target changes everything about the charge
#   Air Defense / Scattershot / Monolith: the things the charge exists to remove
LEVEL_CLASSES = ["Inferno Tower", "Air Defense", "Scattershot", "Monolith"]

# Tile crop geometry. See IsoTransform.crop_box for why these are measured in
# tile WIDTHS rather than tile heights, and why the box sits mostly above the
# tile centre. Swept on identical data and an identical training budget:
#
#     (1.6, 0.90, 0.70)   all tiles 73.7%   buildings 78.6%   <- the old box
#     (2.4, 1.25, 0.72)   all tiles 79.4%   buildings 88.0%
#     (3.0, 1.30, 0.75)   all tiles 74.3%   buildings 88.0%
#
# Nine points of building accuracy for showing the classifier the top of the
# building instead of the platform it stands on. Wider than 2.4 buys nothing
# and costs Empty-vs-building precision, because the crop starts filling with
# the neighbours' buildings.
CROP_W_TILES = 2.4        # crop width, in tile widths (2*hw each)
CROP_H_RATIO = 1.25       # crop height as a multiple of its width
CROP_TOP_FRAC = 0.72      # fraction of the height that sits above tile centre


# ----------------------------------------------------------------------
# 1. Isometric calibration
# ----------------------------------------------------------------------
@dataclass
class IsoTransform:
    """Maps the 44x44 village grid to screen pixels and back.

    Clash renders the village as a diamond in a plain isometric projection, so
    the mapping is a four-parameter affine:

        sx = ox + (gx - gy) * hw
        sy = oy + (gx + gy) * hh

    with hw, hh the half-width and half-height of one tile. Four parameters, and
    the diamond's four corners over-determine them -- which means calibration is
    exact rather than learned, and it is the one part of this module that will
    work perfectly on a real screenshot.
    """
    ox: float
    oy: float
    hw: float
    hh: float

    def to_screen(self, gx: float, gy: float) -> Tuple[float, float]:
        return (self.ox + (gx - gy) * self.hw,
                self.oy + (gx + gy) * self.hh)

    def to_grid(self, sx: float, sy: float) -> Tuple[float, float]:
        u = (sx - self.ox) / self.hw
        v = (sy - self.oy) / self.hh
        return (u + v) / 2.0, (v - u) / 2.0

    @classmethod
    def from_corners(cls, top: Tuple[float, float], right: Tuple[float, float],
                     bottom: Tuple[float, float], left: Tuple[float, float],
                     n: int = C.GRID_SIZE) -> "IsoTransform":
        """Solve from the village diamond's four screen corners.

        `top` is grid (0,0), `right` is (n,0), `bottom` is (n,n), `left` is
        (0,n) -- the order you see them going clockwise from the top vertex.
        """
        hw = ((right[0] - top[0]) + (bottom[0] - left[0])) / (2.0 * n)
        hh = ((left[1] - top[1]) + (bottom[1] - right[1])) / (2.0 * n)
        return cls(ox=top[0], oy=top[1], hw=hw, hh=hh)

    def tile_polygon(self, gx: int, gy: int) -> List[Tuple[float, float]]:
        return [self.to_screen(gx, gy), self.to_screen(gx + 1, gy),
                self.to_screen(gx + 1, gy + 1), self.to_screen(gx, gy + 1)]

    def crop_box(self, gx: int, gy: int, pad: float = CROP_W_TILES
                 ) -> Tuple[int, int, int, int]:
        """Axis-aligned box around a tile, sized to contain the BUILDING, not
        the tile.

        This is worth getting right and easy to get wrong. A tile is 2*hw wide
        and only 2*hh = hw tall, but a 3x3 building drawn on it is about 6*hw
        wide AND 6*hw tall -- six times the tile's height -- because isometric
        art rises off the ground. Sizing the crop from the tile's height, as the
        first version did, gave a box that reached 36 px above the tile centre
        on a building whose distinctive top was 82 px up. Every 3x3 defense sits
        on a near-identical stone platform, so the classifier was being shown
        the one part of a Cannon, a Mortar and a Bomb Tower that looks the same.

        So the box is sized from hw, and is mostly ABOVE the tile centre.
        """
        cx, cy = self.to_screen(gx + 0.5, gy + 0.5)
        w = 2.0 * self.hw * pad
        h = w * CROP_H_RATIO
        return (int(cx - w / 2), int(cy - h * CROP_TOP_FRAC), int(w), int(h))


def detect_village_diamond(img: np.ndarray,
                           debug: bool = False) -> Optional[IsoTransform]:
    """Find the village boundary in a screenshot and calibrate from it.

    The playable area is a large diamond of grass. This masks for grass-like
    pixels, keeps the largest connected blob so UI elements and background do
    not drag the extremes around, and reads the four vertices off it.

    This is a HEURISTIC and it is the part most likely to break on a real
    screenshot -- different zoom, the attack UI, troops on screen. Always check
    it with `overlay_grid` before trusting a recognition, and if it is off, use
    `IsoTransform.from_corners` and click the four vertices yourself. Manual
    calibration is exact; this is only a convenience.
    """
    a = img[..., :3].astype(np.int16)
    # The village diamond is simply "everything that is not the background".
    # Sample the border to learn the background colour, then mask anything that
    # differs from it. This is far more robust than looking for grass, because
    # on a built-up base most of the grass is under buildings.
    h, w = a.shape[:2]
    border = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3),
                             a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    mask = (np.abs(a - bg).sum(axis=2) > 40)
    if mask.sum() < 2000:
        m = a.mean(axis=2)
        mask = m > np.percentile(m, 70)
    mask = _largest_blob(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 1000:
        return None
    # An isometric diamond's four vertices ARE the extremes of x and y: the
    # topmost pixel is the north vertex, the leftmost is the west vertex, and so
    # on. (The x+y / x-y trick finds the corners of a 45-degree-rotated SQUARE,
    # which is a different shape and gives a degenerate horizontal band here.)
    top = (float(xs[np.argmin(ys)]), float(ys.min()))
    bottom = (float(xs[np.argmax(ys)]), float(ys.max()))
    left = (float(xs.min()), float(ys[np.argmin(xs)]))
    right = (float(xs.max()), float(ys[np.argmax(xs)]))
    iso = IsoTransform.from_corners(top, right, bottom, left)
    if iso.hw <= 1.0 or iso.hh <= 0.5 or not (1.4 < iso.hw / iso.hh < 3.0):
        return None                      # isometric tiles are ~2:1; reject junk
    return iso


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Biggest 4-connected component, so stray bright UI does not move the
    diamond's corners. Iterative flood fill -- no scipy dependency."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    best = np.zeros_like(mask)
    best_n = 0
    step = max(1, min(h, w) // 200)
    for sy in range(0, h, step):
        for sx in range(0, w, step):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > best_n:
                best_n = len(comp)
                best = np.zeros_like(mask)
                for (y, x) in comp:
                    best[y, x] = True
    return best if best_n else mask


def overlay_grid(img: np.ndarray, iso: IsoTransform, out: str,
                 every: int = 4) -> str:
    """Draw the calibrated grid over the screenshot. LOOK AT THIS before
    trusting any recognition -- if the lines do not sit on tile boundaries, the
    calibration is wrong and everything downstream is noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    for i in range(0, C.GRID_SIZE + 1, every):
        a1 = iso.to_screen(i, 0)
        a2 = iso.to_screen(i, C.GRID_SIZE)
        b1 = iso.to_screen(0, i)
        b2 = iso.to_screen(C.GRID_SIZE, i)
        ax.plot([a1[0], a2[0]], [a1[1], a2[1]], color="#C0392B", lw=.8, alpha=.8)
        ax.plot([b1[0], b2[0]], [b1[1], b2[1]], color="#2E86AB", lw=.8, alpha=.8)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 2. Tile extraction
# ----------------------------------------------------------------------
def extract_tiles(img: np.ndarray, iso: IsoTransform,
                  size: int = 48) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Crop every grid tile to a fixed-size patch for the classifier."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("pip install pillow") from e
    pil = Image.fromarray(img.astype(np.uint8))
    crops, coords = [], []
    for gy in range(C.GRID_SIZE):
        for gx in range(C.GRID_SIZE):
            x, y, w, h = iso.crop_box(gx, gy)
            if w <= 2 or h <= 2:
                continue
            patch = pil.crop((x, y, x + w, y + h)).resize((size, size))
            crops.append(np.asarray(patch, dtype=np.uint8)[..., :3])
            coords.append((gx, gy))
    return np.stack(crops) if crops else np.zeros((0, size, size, 3), np.uint8), coords


# ----------------------------------------------------------------------
# 3. The classifier
# ----------------------------------------------------------------------
def build_classifier(n_classes: int = len(CLASSES), n_levels: int = 0):
    """Small CNN over tile crops. Two heads: building type, and -- only for the
    handful of defenses where it changes the plan -- level.

    The pooling is deliberately 2x2 rather than 1x1. A tile crop is padded
    beyond the tile itself, so a lot of what distinguishes "this tile IS the
    Cannon" from "this tile is beside the Cannon" is WHERE in the crop the
    building sits. Global average pooling throws that away; keeping a 2x2 grid
    of features costs 1,500 parameters and keeps it.
    """
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    class TileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128),
                                   blk(128, 160), nn.AdaptiveAvgPool2d(2),
                                   nn.Flatten())
            self.trunk = nn.Sequential(nn.Linear(160 * 4, 256),
                                       nn.ReLU(inplace=True), nn.Dropout(0.1))
            self.type_head = nn.Linear(256, n_classes)
            self.level_head = nn.Linear(256, n_levels) if n_levels else None

        def forward(self, x):
            z = self.trunk(self.f(x))
            if self.level_head is None:
                return self.type_head(z)
            return self.type_head(z), self.level_head(z)

    return TileNet()


# ----------------------------------------------------------------------
# 4. Synthetic training data
#
# The previous version of this drew every building as a flat coloured box. It
# was honest about being a plumbing test and useless as training data: a teal
# hexagon and an Air Defense share no pixels, so a network trained on boxes
# learns "teal blob, middle third" and then meets Clash's actual art.
#
# This version composites the REAL sprites -- captured from the wiki into
# vision_data/sprites, see coc.sprites -- onto a procedurally generated base.
# That closes most of the domain gap without needing a single hand-labelled
# screenshot, because the thing the classifier has to recognise is now
# literally the thing it will see.
#
# What is still synthetic, and therefore still a gap:
#   * the ground. Real grass has decorations, obstacles, seasonal skins.
#   * troops, spell effects and health bars standing on top of buildings.
#   * scenery skins, which repaint every building. A classifier trained on the
#     default skin will not read a base wearing one.
# The augmentations below attack the first two. The third needs your
# screenshots, and is the reason `finetune` exists.
# ----------------------------------------------------------------------


def _level_vocab() -> List[str]:
    """Label space for the level head.

    Index 0 is "n/a" -- every class where the level does not change the plan.
    The rest are one entry per (class, level, variant) appearance that the
    library actually contains, so the head can never be asked to predict a
    level we have no picture of.
    """
    out = ["n/a"]
    for cls, lvl, fname in S.MANIFEST:
        if cls not in LEVEL_CLASSES:
            continue
        t = S.tag(cls, lvl, fname)
        if t not in out:
            out.append(t)
    return out


LEVEL_VOCAB = _level_vocab()
LEVEL_INDEX = {t: i for i, t in enumerate(LEVEL_VOCAB)}

# How wide the solid part of a sprite should be drawn, as a multiple of the
# building's ground diamond. Clash art overhangs its footprint slightly -- a
# Cannon's barrel sticks out past the platform -- and walls are drawn to touch
# their neighbours with no seam.
FIT = {"Wall": 1.02, "Town Hall": 1.02}
FIT_DEFAULT = 1.02


def _ground(w: int, h: int, iso: IsoTransform, rng) -> np.ndarray:
    """The village floor.

    Vectorised through the inverse transform rather than drawn as 1,936
    polygons: every pixel is mapped back to grid coordinates in one pass, which
    makes the tile a pixel belongs to a lookup instead of a fill.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs - iso.ox) / iso.hw
    v = (ys - iso.oy) / iso.hh
    gx = (u + v) * 0.5
    gy = (v - u) * 0.5
    inside = (gx >= 0) & (gx < C.GRID_SIZE) & (gy >= 0) & (gy < C.GRID_SIZE)
    ti = np.clip(gx.astype(np.int32), 0, C.GRID_SIZE - 1)
    tj = np.clip(gy.astype(np.int32), 0, C.GRID_SIZE - 1)

    light = np.array([0.475, 0.655, 0.310], np.float32)
    dark = np.array([0.420, 0.600, 0.272], np.float32)
    out_c = np.array([0.230, 0.300, 0.170], np.float32)

    checker = ((ti + tj) & 1).astype(np.float32)[..., None]
    img = dark + checker * (light - dark)

    # per-tile tint, so it reads as mown grass rather than a chessboard
    r = np.random.RandomState(rng.randrange(1 << 30))
    tile_tint = r.normal(0.0, 0.016, (C.GRID_SIZE, C.GRID_SIZE, 3)).astype(np.float32)
    img += tile_tint[tj, ti]
    # outside the playable diamond the tile lookup is clamped, so do NOT tint
    # by it -- clamping turns per-tile noise into radial streaks
    img = np.where(inside[..., None], img, out_c)

    # fine texture and a soft top-left light
    img += r.normal(0.0, 0.012, img.shape).astype(np.float32)
    grad = ((xs / w) * 0.5 + (ys / h) * 0.5).astype(np.float32)[..., None]
    img *= (1.06 - 0.12 * grad)
    return np.clip(img, 0.0, 1.0)


def _paste(canvas: np.ndarray, sp, cx: float, by: float, target_w: float) -> None:
    """Alpha-composite one sprite so it sits on its ground diamond.

    `cx` is the diamond's centre x and `by` its bottom vertex y -- for a square
    footprint (every Clash building is square) the bottom vertex is directly
    below the centre, so those two numbers place the building exactly. The
    sprite is scaled so its SOLID width matches the diamond and anchored by the
    bottom-centre of its solid box, which is where the building's near corner
    touches the ground.
    """
    from PIL import Image
    x0s, y0s, x1s, y1s = sp.box
    solid_w = max(1, x1s - x0s)
    k = target_w / solid_w
    nh = max(1, int(round(sp.img.shape[0] * k)))
    nw = max(1, int(round(sp.img.shape[1] * k)))
    im = Image.fromarray((sp.img * 255).astype(np.uint8), "RGBA")
    arr = np.asarray(im.resize((nw, nh), Image.LANCZOS), np.float32) / 255.0

    ax = (x0s + x1s) * 0.5 * k          # anchor inside the resized sprite
    ay = y1s * k
    ox = int(round(cx - ax))
    oy = int(round(by - ay))

    H, W = canvas.shape[:2]
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw = min(nw - sx0, W - dx0)
    chh = min(nh - sy0, H - dy0)
    if cw <= 0 or chh <= 0:
        return
    src = arr[sy0:sy0 + chh, sx0:sx0 + cw]
    a = src[..., 3:4]
    dst = canvas[dy0:dy0 + chh, dx0:dx0 + cw]
    canvas[dy0:dy0 + chh, dx0:dx0 + cw] = src[..., :3] * a + dst * (1.0 - a)


def _occlude(canvas: np.ndarray, rng, n: int) -> None:
    """Blobs standing on the base.

    In a real attack screenshot the village is covered in troops, spell
    circles, damage numbers and health bars. A classifier that has only ever
    seen clean buildings treats any of that as a different building. These are
    not realistic troops -- they are just "something opaque is in the way",
    which is the invariance that actually needs training.
    """
    H, W = canvas.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n):
        cx, cy = rng.randrange(W), rng.randrange(H)
        rad = rng.randint(6, 26)
        col = np.array([rng.uniform(0.1, 0.95) for _ in range(3)], np.float32)
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < rad * rad
        alpha = rng.uniform(0.35, 0.95)
        canvas[m] = canvas[m] * (1 - alpha) + col * alpha


def render_synthetic(seed: int = 0, px: int = 1600, lib=None,
                     augment: bool = True):
    """Draw a generated base in isometric projection, with its label grids.

    Returns (image uint8 HxWx3, type labels 44x44, level labels 44x44, iso).

    The level grid indexes LEVEL_VOCAB, and is filled only for the four classes
    in LEVEL_CLASSES -- everything else is 0 ("n/a"). Crucially the level label
    is whatever sprite was ACTUALLY drawn, so the label can never disagree with
    the pixels.
    """
    import random as _random

    rng = _random.Random(seed)
    if lib is None:
        lib = get_library()

    grid, buildings, traps, cc_pos, altar = generate_base(
        1.0, seed=seed, traps=True, cc=True, hero=True)

    H = int(px * 0.62)
    zoom = rng.uniform(0.86, 1.05) if augment else 1.0
    hw = px / (2.0 * C.GRID_SIZE) * zoom
    hh = hw * 0.5
    ox = px * 0.5 + (rng.uniform(-0.03, 0.03) * px if augment else 0.0)
    oy = (H - 2 * C.GRID_SIZE * hh) * 0.5 + (rng.uniform(-0.02, 0.02) * H
                                             if augment else 0.0)
    iso = IsoTransform(ox=ox, oy=oy, hw=hw, hh=hh)

    canvas = _ground(px, H, iso, rng)
    labels = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.int16)
    levels = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.int16)

    # painter's order: back of the diamond first. Ties broken by footprint so a
    # big building never gets drawn over by the small one beside it.
    for b in sorted(buildings, key=lambda b: (b.x + b.y, -b.w)):
        # The generator names buildings the way the game does ("Collector",
        # "Army Camp", "Hero Altar"); the library is keyed by CLASS. Resolve
        # first, or every resource building silently falls back to a flat box
        # -- which is exactly the bug the first render of this had.
        cls = b.name if b.name in CLASS_INDEX else "Non-Defense"
        sp = lib.pick(cls, rng) if lib.ok else None
        s = max(b.w, b.h)
        cx, _ = iso.to_screen(b.x + s / 2.0, b.y + s / 2.0)
        _, by = iso.to_screen(b.x + s, b.y + s)
        target = 2.0 * s * hw * FIT.get(b.name, FIT_DEFAULT)
        if sp is not None:
            _paste(canvas, sp, cx, by, target)
            if cls in LEVEL_CLASSES:
                levels[b.y:b.y + b.h, b.x:b.x + b.w] = LEVEL_INDEX.get(sp.tag, 0)
        else:
            _fallback_box(canvas, iso, b)
        labels[b.y:b.y + b.h, b.x:b.x + b.w] = CLASS_INDEX[cls]

    # Traps are NOT drawn and NOT labelled: in a real base they are invisible
    # until they fire, so a classifier that predicted them would be predicting
    # something the screen does not contain.

    if augment:
        _occlude(canvas, rng, rng.randint(0, 22))
        canvas *= rng.uniform(0.82, 1.14)                       # exposure
        canvas += rng.uniform(-0.05, 0.05)                      # black level
        m = canvas.mean(axis=2, keepdims=True)
        canvas = m + (canvas - m) * rng.uniform(0.82, 1.20)     # saturation
        canvas += np.random.normal(0, rng.uniform(0.004, 0.022),
                                   canvas.shape).astype(np.float32)

    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8), labels, levels, iso


def _fallback_box(canvas: np.ndarray, iso: IsoTransform, b) -> None:
    """Flat coloured prism, used only when the sprite library is missing.

    Kept so the module still runs end to end on a clean checkout with no
    captured sprites -- the pipeline works, the realism does not.
    """
    pal = {"Town Hall": (.75, .23, .17), "Air Defense": (.18, .53, .67),
           "Air Sweeper": (.29, .64, .78), "Eagle Artillery": (.49, .37, .66),
           "Monolith": (.24, .23, .30), "Scattershot": (.72, .47, .12),
           "Inferno Tower": (.85, .42, .23), "X-Bow": (.56, .44, .24),
           "Archer Tower": (.61, .56, .44), "Cannon": (.54, .51, .47),
           "Mortar": (.44, .50, .42), "Wizard Tower": (.48, .43, .66),
           "Hidden Tesla": (.37, .62, .63), "Bomb Tower": (.36, .32, .28),
           "Spell Tower": (.66, .44, .66), "Clan Castle": (.60, .48, .31),
           "Wall": (.79, .71, .54)}
    col = np.array(pal.get(b.name, (.64, .71, .56)), np.float32)
    hgt = (2 * iso.hh) * (2.2 if b.name != "Wall" else 0.8)
    pts = [iso.to_screen(b.x, b.y), iso.to_screen(b.x + b.w, b.y),
           iso.to_screen(b.x + b.w, b.y + b.h), iso.to_screen(b.x, b.y + b.h)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys) - hgt), int(max(ys))
    H, W = canvas.shape[:2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = col


_LIB = None


def get_library(directory: str = S.SPRITE_DIR_DEFAULT):
    """Load the sprite library once and keep it."""
    global _LIB
    if _LIB is None or _LIB.dir != directory:
        _LIB = S.SpriteLibrary(directory)
    return _LIB


def make_dataset(n_bases: int = 45, out: str = "vision_data/synthetic.npz",
                 size: int = 48, px: int = 1600, augment: bool = True,
                 sprites_dir: str = S.SPRITE_DIR_DEFAULT,
                 empty_keep: float = 0.35, seed0: int = 0) -> str:
    """Render n bases and dump (crop, type, level) triples.

    Empty tiles are subsampled. Forty-five percent of a village is bare grass,
    and grass is both the easiest class and the one whose gradient drowns
    everything else; keeping a third of it is plenty to learn "nothing here"
    and buys the budget back for tiles that are hard.
    """
    import random as _random
    lib = get_library(sprites_dir)
    if not lib.ok:
        print(f"  !! no sprite library at {sprites_dir} -- falling back to "
              "coloured boxes, which will NOT transfer to real screenshots. "
              "Run scripts/crop_sprites.py first.")
    rng = _random.Random(12345)
    X, Y, L = [], [], []
    for i in range(n_bases):
        img, labels, levels, iso = render_synthetic(seed=seed0 + i, px=px,
                                                    lib=lib, augment=augment)
        crops, coords = extract_tiles(img, iso, size=size)
        for patch, (gx, gy) in zip(crops, coords):
            y = int(labels[gy, gx])
            if y == 0 and rng.random() > empty_keep:
                continue
            X.append(patch)
            Y.append(y)
            L.append(int(levels[gy, gx]))
        if (i + 1) % 5 == 0:
            print(f"  rendered {i+1}/{n_bases} bases, {len(X):,} tiles kept",
                  flush=True)
    X = np.stack(X)
    Y = np.array(Y, dtype=np.int64)
    L = np.array(L, dtype=np.int64)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # uncompressed on purpose: these are noisy 48x48 crops, so zlib buys ~5%
    # for minutes of CPU that the two cores here would rather spend training
    np.savez(out, X=X, Y=Y, L=L, classes=np.array(CLASSES),
             level_vocab=np.array(LEVEL_VOCAB))
    print(f"  {out}: {len(X):,} tiles, {X.nbytes/1e6:.0f} MB")
    return out


# ----------------------------------------------------------------------
# 5. Recognition -> a playable environment
# ----------------------------------------------------------------------
# Every building in Clash has a KNOWN footprint and there is a known number of
# each on a TH15 base. That is a lot of structure to throw away, and the first
# version of this did throw it away -- it grew rectangles out of whatever the
# classifier happened to say, so one misread tile in the middle of a Cannon
# turned a 3x3 building into two 1x3 slivers.
#
# So instead: slide the known footprint over the class votes and take the best
# non-overlapping placements, at most as many as the roster allows. A 3x3
# Cannon survives three wrong tiles out of nine. This is where most of the
# per-tile classifier error goes to die, and it is why per-tile accuracy in the
# 80s is enough to read a base correctly.
# ----------------------------------------------------------------------
FOOTPRINT = {"Town Hall": 4, "Clan Castle": 3, "Wall": 1, "Non-Defense": 3}
ROSTER = {"Town Hall": 1, "Clan Castle": 1}

# How much of a candidate footprint has to vote for the class before we place a
# building there. Defenses come with a known roster cap, so a loose threshold is
# safe -- the count limits the damage. Resource buildings have no cap and three
# possible sizes, so the threshold is what stops a 4x4 window from swallowing a
# 3x3 collector plus its neighbour; these numbers were tuned against
# ground-truth labels, where the right answer is known exactly.
#
# Swept twice. On ground-truth labels, {2:1.0, 3:1.0, 4:1.0} recovers 98.3% of
# buildings and {0.9, 0.8, 0.7} only 93.0%. On real classifier output the
# ordering holds but the margin narrows:
#
#     {0.95, 0.85, 0.80}   77.9% exact   87.9% within one tile   2.0 roster err
#     {0.75, 0.60, 0.55}   69.3% exact   88.7% within one tile   2.0 roster err
#     {0.55, 0.45, 0.40}   66.8% exact   88.4% within one tile   2.0 roster err
#
# Loosening buys almost nothing in coverage and costs eight points of exact
# placement, and exact placement is what the planner consumes -- an Air Defense
# read one tile off moves the Giant Arrow line. So: strict.
DEF_MIN_FRAC = {1: 0.50, 2: 0.45, 3: 0.40, 4: 0.38}
ND_MIN_FRAC = {2: 0.95, 3: 0.85, 4: 0.80}
for _s in C.DEFENSES:
    FOOTPRINT[_s.name] = int(_s.size[0])
    ROSTER[_s.name] = int(_s.count)


# Per-level stats, so reading a level actually changes the plan instead of
# merely being printed. Only the three classes whose numbers could be
# cross-checked are here: for each of them the config's max-level value and the
# wiki's table for that level agree exactly --
#
#     Air Defense    config hp 1750, dps 540  ==  wiki level 13
#     Inferno Tower  config hp 4000, dps 100  ==  wiki level 9  (initial ramp)
#     Monolith       config dps 175, pct 12%  ==  wiki level 2
#
# -- which confirms both the tables and that the sprite file numbering is the
# game's level numbering. Scattershot is deliberately absent: the wiki page
# still claims it has two levels while the file namespace has seven, so the
# numbers there could not be trusted, and a wrong DPS silently corrupts every
# plan built on it. Anything not listed keeps its TH15 max-level stats.
#     https://clashofclans.fandom.com/wiki/Air_Defense/Home_Village
#     https://clashofclans.fandom.com/wiki/Inferno_Tower/Home_Village
LEVEL_STATS = {
    ("Air Defense", 9): dict(hp=1300.0, dps=360.0),
    ("Air Defense", 10): dict(hp=1400.0, dps=400.0),
    ("Air Defense", 11): dict(hp=1500.0, dps=440.0),
    ("Air Defense", 12): dict(hp=1650.0, dps=500.0),
    ("Air Defense", 13): dict(hp=1750.0, dps=540.0),
    ("Inferno Tower", 6): dict(hp=3000.0, dps=55.0),
    ("Inferno Tower", 7): dict(hp=3300.0, dps=65.0),
    ("Inferno Tower", 8): dict(hp=3700.0, dps=80.0),
    ("Inferno Tower", 9): dict(hp=4000.0, dps=100.0),
    ("Monolith", 1): dict(dps=150.0),
    ("Monolith", 2): dict(dps=175.0),
}


def _place(votes: np.ndarray, size: int, limit: int, taken: np.ndarray,
           min_frac: float = 0.42) -> List[Tuple[int, int, float]]:
    """Greedy non-overlapping placement of `size` x `size` boxes.

    Scores every legal position by how many of its tiles voted for this class,
    takes the best, blanks it out, repeats. `taken` is shared across classes so
    two buildings can never claim the same tile.
    """
    n = C.GRID_SIZE
    out: List[Tuple[int, int, float]] = []
    free = votes.astype(np.float32) * (~taken)
    need = min_frac * size * size
    for _ in range(limit):
        # window sums via a summed-area table -- 44x44 is small, but this runs
        # once per class per placement and the naive loop showed up in profiles
        sat = free.cumsum(0).cumsum(1)
        sat = np.pad(sat, ((1, 0), (1, 0)))
        win = (sat[size:, size:] - sat[:-size, size:]
               - sat[size:, :-size] + sat[:-size, :-size])
        blocked = np.zeros_like(win, dtype=bool)
        for y in range(win.shape[0]):
            for x in range(win.shape[1]):
                if taken[y:y + size, x:x + size].any():
                    blocked[y, x] = True
        win = np.where(blocked, -1.0, win)
        if win.size == 0:
            break
        yy, xx = np.unravel_index(int(np.argmax(win)), win.shape)
        best = float(win[yy, xx])
        if best < need:
            break
        out.append((int(xx), int(yy), best / (size * size)))
        taken[yy:yy + size, xx:xx + size] = True
        free[yy:yy + size, xx:xx + size] = 0.0
    return out


def grid_to_env(pred: np.ndarray, levels: Optional[np.ndarray] = None,
                defense_frac: float = 1.0):
    """Turn a recognised 44x44 class grid into buildings you can plan on.

    Returns a list of `Building` with real TH15 stats attached from the config.
    Each carries a `.level` string (the level head's majority vote, or "" where
    the level does not matter) and a `.confidence` in [0,1] -- the fraction of
    its footprint that voted for it. Anything below the threshold is dropped
    rather than guessed: a phantom Inferno Tower poisons a plan far worse than
    a missing collector.
    """
    specs = {s.name: s for s in C.DEFENSES}
    taken = np.zeros_like(pred, dtype=bool)
    buildings: List[Building] = []
    uid = 0

    # Biggest and rarest first. The Eagle and the Town Hall are the two things
    # you least want stolen by a neighbouring 3x3, and a greedy pass gives
    # whoever goes first the pick of the tiles.
    order = sorted((n for n in FOOTPRINT if n not in ("Wall", "Non-Defense")),
                   key=lambda n: (-FOOTPRINT[n], ROSTER.get(n, 99)))
    # Resource buildings come in three footprints -- an Army Camp is 4x4, a
    # collector 3x3, a builder hut 2x2 -- and the class does not say which. Try
    # each size largest first; the greedy score naturally prefers the size that
    # actually covers the votes, and the leftovers fall through to the smaller
    # passes instead of being lost.
    order += [("Non-Defense", 4), ("Non-Defense", 3), ("Non-Defense", 2)]

    for item in order:
        name, size = item if isinstance(item, tuple) else (item, FOOTPRINT[item])
        ci = CLASS_INDEX.get(name)
        if ci is None:
            continue
        limit = ROSTER.get(name, 40)
        votes = (pred == ci)
        if not votes.any():
            continue
        mf = (ND_MIN_FRAC if name == "Non-Defense" else DEF_MIN_FRAC)[size]
        for (gx, gy, conf) in _place(votes, size, limit, taken, min_frac=mf):
            sp = specs.get(name)
            cat = (CAT_TOWN_HALL if name == "Town Hall" else
                   CAT_AIR_DEFENSE if name in ("Air Defense", "Air Sweeper") else
                   CAT_HIGH_DEFENSE if sp and sp.tier == "high" else
                   CAT_DEFENSE if sp else CAT_NON_DEFENSE)
            hp = (C.TH_HP if name == "Town Hall" else
                  sp.hp if sp else C.NON_DEFENSE_HP)
            b = Building(
                uid=uid, name=name, cat=cat, x=gx, y=gy, w=size, h=size,
                hp=float(hp), max_hp=float(hp),
                dps=(C.TH_DPS if name == "Town Hall" else sp.dps if sp else 0.0),
                rng=(C.TH_RANGE if name == "Town Hall" else sp.rng if sp else 0.0),
                min_range=sp.min_range if sp else 0.0,
                hits_ground=sp.hits_ground if sp else True,
                splash=sp.splash if sp else False,
                tier="high" if name == "Town Hall" else (sp.tier if sp else "normal"),
                is_defense=bool(sp) or name == "Town Hall")
            b.confidence = float(conf)
            b.level = ""
            b.mode = ""
            if levels is not None and name in LEVEL_CLASSES:
                patch = levels[gy:gy + size, gx:gx + size].ravel()
                patch = patch[patch > 0]
                if len(patch):
                    vals, cnt = np.unique(patch, return_counts=True)
                    tagname = LEVEL_VOCAB[int(vals[int(np.argmax(cnt))])]
                    b.level = tagname.split(":", 1)[1]
                    # "9S" -> level 9, single-target
                    digits = "".join(c for c in b.level if c.isdigit())
                    b.mode = "".join(c for c in b.level if c.isalpha())
                    st = LEVEL_STATS.get((name, int(digits))) if digits else None
                    if st:
                        if "hp" in st:
                            b.hp = b.max_hp = st["hp"]
                        if "dps" in st:
                            b.dps = st["dps"]
            buildings.append(b)
            uid += 1

    # Walls last, into whatever is left. They are 1x1 so there is nothing to
    # snap, and a wrong wall costs a plan almost nothing.
    wi = CLASS_INDEX["Wall"]
    for gy in range(C.GRID_SIZE):
        for gx in range(C.GRID_SIZE):
            if pred[gy, gx] == wi and not taken[gy, gx]:
                taken[gy, gx] = True
                b = Building(uid=uid, name="Wall", cat=CAT_WALL, x=gx, y=gy,
                             w=1, h=1, hp=float(C.WALL_HP),
                             max_hp=float(C.WALL_HP), is_defense=False)
                b.confidence = 1.0
                b.level = ""
                b.mode = ""
                buildings.append(b)
                uid += 1
    return buildings


def describe_base(buildings) -> str:
    """Human-readable summary of what was read off the screen.

    Print this before trusting a plan. If the roster is wrong -- two Air
    Defenses on a TH15, no Eagle -- the recognition failed and the plan built
    on it is fiction, however confident it looks.
    """
    from collections import Counter
    rows = []
    counts = Counter(b.name for b in buildings)
    for name in sorted(counts, key=lambda n: (-C.target_value(
            n, n in {s.name for s in C.DEFENSES} or n == "Town Hall"), n)):
        if name == "Wall":
            continue
        want = ROSTER.get(name)
        got = counts[name]
        lv = [b.level for b in buildings if b.name == name and b.level]
        note = ""
        if want and got != want:
            note = f"   <-- expected {want}"
        rows.append(f"  {name:<17s} {got:2d}"
                    + (f"  levels {','.join(sorted(set(lv)))}" if lv else "")
                    + note)
    rows.append(f"  {'Wall':<17s} {counts.get('Wall', 0):2d}")
    conf = [b.confidence for b in buildings if b.name != "Wall"]
    rows.append(f"\n  mean footprint agreement {100*float(np.mean(conf or [0])):.0f}%")

    # The one reading that changes how you fly the dragons into the core.
    inf = [b for b in buildings if b.name == "Inferno Tower"]
    if inf:
        multi = sum(1 for b in inf if getattr(b, "mode", "") == "M")
        single = sum(1 for b in inf if getattr(b, "mode", "") == "S")
        rows.append(f"  Infernos: {multi} multi-target, {single} single-target")
        if multi:
            rows.append("    -> multi shreds a dragon pack; freeze it or route "
                        "around it, do not fly the stack past it")
        if single:
            rows.append("    -> single melts the Warden once it ramps; keep him "
                        "out of its 9-tile range or eat the ramp with a tank")
    return "\n".join(rows)


def plan_attack(buildings, model_path: str, device: str = "auto") -> Dict:
    """Run the trained agent on a recognised base and return the plan.

    Output is advice, in grid coordinates you can convert back to screen with
    the same IsoTransform: where to drop the Champion, where the Giant Arrow
    lines up, which side to bring the dragons in from.
    """
    import torch

    env = FullAttackEnv(defense_frac=1.0, max_spells=C.TOTAL_SPELLS, seed=0)
    env.reset()
    env.buildings = buildings                      # swap in the real base
    env.by_uid = {b.uid: b for b in buildings}
    env.grid = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.int16)
    for b in buildings:
        env.grid[b.y:b.y + b.h, b.x:b.x + b.w] = b.cat
    env.town_hall = next((b for b in buildings if b.name == "Town Hall"), None)
    env._dirty = True
    env._arrow_map = arrow_value_map(env._deploy_mask(), buildings, C.GRID_SIZE)

    dev = pick_device(device)
    blob = torch.load(model_path, map_location=dev, weights_only=False)
    cfg = C.TrainConfig(**{k: v for k, v in blob["cfg"].items()
                           if k in C.TrainConfig.__dataclass_fields__})
    net = RCQNet(cfg, n_channels=N_CH, n_scalars=N_SC,
                 n_scalar_actions=5, n_tile_heads=2).to(dev)
    net.load_state_dict(blob["policy"])
    net.eval()

    sp, sc = env._obs()
    m = torch.from_numpy(env.legal_actions()).unsqueeze(0).to(dev)
    q = net.q_masked(torch.from_numpy(sp).unsqueeze(0).to(dev),
                     torch.from_numpy(sc).unsqueeze(0).to(dev), m)
    a = int(q.argmax(dim=1).item())
    rc_tile = env._tile(a - A_DEPLOY) if a >= A_DEPLOY else None

    am = env._arrow_map
    ay, ax = np.unravel_index(int(np.argmax(am)), am.shape)
    sweep = env._sweeper_cov
    edge = [(x, y) for x in range(2, C.GRID_SIZE - 2, 2)
            for y in (2, C.GRID_SIZE - 3)] + \
           [(x, y) for y in range(2, C.GRID_SIZE - 2, 2)
            for x in (2, C.GRID_SIZE - 3)]
    dragon_side = min(edge, key=lambda p: sweep[p[1], p[0]])

    return dict(
        deploy_champion=rc_tile,
        giant_arrow_tile=(int(ax * C.ACTION_STRIDE), int(ay * C.ACTION_STRIDE)),
        giant_arrow_air_defenses=int(am.max()),
        dragon_entry=dragon_side,
        air_defenses=[(b.cx, b.cy) for b in buildings if b.name == "Air Defense"],
        key_defenses=[(b.name, b.cx, b.cy) for b in buildings
                      if C.target_value(b.name, b.is_defense) >= 2.0],
    )


# ----------------------------------------------------------------------
# 6. Training
# ----------------------------------------------------------------------
def train_classifier(npz: str = "vision_data/synthetic.npz",
                     out: str = "vision_data/tilenet.pt",
                     epochs: int = 8, batch: int = 256, lr: float = 2e-3,
                     device: str = "auto", val_frac: float = 0.15) -> Dict:
    """Train the tile classifier: building type, and level where it matters.

    Two things about the loss are worth knowing.

    The type loss is class-weighted by 1/sqrt(frequency). Tiles are 45% Empty
    and 13% Wall; an unweighted loss reaches 60% accuracy by never predicting a
    defense at all, which is precisely the failure mode that matters here.

    The level loss is masked to the four LEVEL_CLASSES. Asking the level head
    what level a patch of grass is would train it on noise, and the gradient
    from 1,700 grass tiles per base would swamp the 36 Air Defense tiles that
    carry the actual signal.
    """
    import torch
    import torch.nn as nn

    d = np.load(npz, allow_pickle=True)
    X, Y = d["X"], d["Y"]
    L = d["L"] if "L" in d.files else np.zeros_like(Y)
    vocab = list(d["level_vocab"]) if "level_vocab" in d.files else LEVEL_VOCAB
    vocab = [str(v) for v in vocab]
    dev = pick_device(device)

    n = len(X)
    rs = np.random.RandomState(0)
    idx = rs.permutation(n)
    split = int(n * (1 - val_frac))
    tr, va = idx[:split], idx[split:]
    print(f"  {n:,} tiles  ({len(tr):,} train / {len(va):,} val)  device {dev}")

    net = build_classifier(len(CLASSES), len(vocab)).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * (len(tr) // batch + 1)))

    counts = np.bincount(Y, minlength=len(CLASSES)).astype(np.float32)
    wt = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5,
                      dtype=torch.float32, device=dev)
    type_loss = nn.CrossEntropyLoss(weight=wt)
    level_loss = nn.CrossEntropyLoss()
    level_ids = {CLASS_INDEX[c] for c in LEVEL_CLASSES}
    lvl_mask_np = np.isin(Y, list(level_ids))
    print(f"  {nparam:,} parameters; {lvl_mask_np.sum():,} tiles carry a level")

    hist = []
    for ep in range(epochs):
        net.train()
        rs.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), batch):
            b = tr[i:i + batch]
            xb = torch.from_numpy(X[b]).float().permute(0, 3, 1, 2).to(dev) / 255.
            yb = torch.from_numpy(Y[b]).to(dev)
            lb = torch.from_numpy(L[b]).to(dev)
            mb = torch.from_numpy(lvl_mask_np[b]).to(dev)
            pt, pl = net(xb)
            loss = type_loss(pt, yb)
            if bool(mb.any()):
                loss = loss + 0.5 * level_loss(pl[mb], lb[mb])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss.detach()) * len(b)

        net.eval()
        pred = np.zeros(len(va), np.int64)
        predl = np.zeros(len(va), np.int64)
        with torch.no_grad():
            for i in range(0, len(va), 1024):
                b = va[i:i + 1024]
                xb = torch.from_numpy(X[b]).float().permute(0, 3, 1, 2).to(dev) / 255.
                pt, pl = net(xb)
                pred[i:i + len(b)] = pt.argmax(1).cpu().numpy()
                predl[i:i + len(b)] = pl.argmax(1).cpu().numpy()
        yv = Y[va]
        acc = float((pred == yv).mean())
        bm = yv > 1                                   # ignore Empty and Wall
        bacc = float((pred[bm] == yv[bm]).mean()) if bm.any() else 0.0
        lm = lvl_mask_np[va] & (pred == yv)
        lacc = float((predl[lm] == L[va][lm]).mean()) if lm.any() else 0.0
        hist.append(dict(epoch=ep + 1, loss=tot / max(1, len(tr)), acc=acc,
                         building_acc=bacc, level_acc=lacc))
        print(f"  epoch {ep+1}/{epochs}  loss {hist[-1]['loss']:.4f}  "
              f"all-tile {100*acc:5.1f}%  building-only {100*bacc:5.1f}%  "
              f"level {100*lacc:5.1f}%", flush=True)

    # per-class report on the final epoch -- an average hides the one class
    # that is broken, and here that class might be the Air Defense
    print("\n  per-class recall (validation):")
    for ci, name in enumerate(CLASSES):
        m = yv == ci
        if not m.any():
            continue
        r = float((pred[m] == ci).mean())
        p = float((yv[pred == ci] == ci).mean()) if (pred == ci).any() else 0.0
        flag = "  <-- weak" if r < 0.8 and m.sum() > 50 else ""
        print(f"     {name:<17s} n={int(m.sum()):6d}  recall {100*r:5.1f}%  "
              f"precision {100*p:5.1f}%{flag}")

    print("\n  level confusion, correctly-typed tiles only:")
    for li, tagname in enumerate(vocab):
        if li == 0:
            continue
        m = lm & (L[va] == li)
        if not m.any():
            continue
        r = float((predl[m] == li).mean())
        print(f"     {tagname:<20s} n={int(m.sum()):5d}  {100*r:5.1f}%")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(dict(state=net.state_dict(), classes=CLASSES,
                    level_vocab=vocab, history=hist), out)
    print(f"\n  saved {out}")
    return dict(path=out, history=hist)


def recognise(img: np.ndarray, model: str = "vision_data/tilenet.pt",
              iso: Optional[IsoTransform] = None,
              device: str = "auto", min_conf: float = 0.5):
    """Screenshot -> (44x44 class grid, 44x44 level grid, iso).

    Tiles the classifier is not confident about are left EMPTY rather than
    guessed. A hallucinated Inferno Tower would poison the plan far worse than
    a missing collector.
    """
    import torch
    iso = iso or detect_village_diamond(img)
    if iso is None:
        raise RuntimeError("could not calibrate the grid -- pass an IsoTransform "
                           "built from the four village corners by hand, see "
                           "calibrate_from_clicks()")
    dev = pick_device(device)
    blob = torch.load(model, map_location=dev, weights_only=False)
    vocab = blob.get("level_vocab", LEVEL_VOCAB)
    net = build_classifier(len(blob.get("classes", CLASSES)), len(vocab)).to(dev)
    net.load_state_dict(blob["state"])
    net.eval()
    crops, coords = extract_tiles(img, iso)
    grid = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.int16)
    lvls = np.zeros((C.GRID_SIZE, C.GRID_SIZE), dtype=np.int16)
    with torch.no_grad():
        for i in range(0, len(crops), 512):
            xb = torch.from_numpy(crops[i:i + 512]).float()
            xb = xb.permute(0, 3, 1, 2).to(dev) / 255.0
            pt, pl = net(xb)
            prob = torch.softmax(pt, dim=1)
            conf, cls = prob.max(1)
            lv = pl.argmax(1)
            for j in range(len(cls)):
                gx, gy = coords[i + j]
                if float(conf[j]) >= min_conf:
                    grid[gy, gx] = int(cls[j])
                    lvls[gy, gx] = int(lv[j])
    return grid, lvls, iso


def evaluate_on_render(model: str = "vision_data/tilenet.pt",
                       seeds: Sequence[int] = (900, 901, 902, 903, 904),
                       px: int = 1600, device: str = "auto") -> Dict:
    """End-to-end check on held-out renders the model never trained on.

    This is the number that means something. Per-tile validation accuracy is
    measured on crops drawn from the SAME renders as the training crops, so
    neighbouring tiles leak. These seeds render fresh bases and run the whole
    pipeline -- calibrate, crop, classify, reassemble -- exactly as a real
    screenshot would.
    """
    tot = ok = bok = bn = lok = ln = 0
    per_class = {}
    for s in seeds:
        img, labels, levels, iso = render_synthetic(seed=s, px=px, augment=True)
        pred, plvl, _ = recognise(img, model=model, iso=iso, device=device,
                                  min_conf=0.0)
        tot += labels.size
        ok += int((pred == labels).sum())
        m = labels > 1
        bn += int(m.sum()); bok += int((pred[m] == labels[m]).sum())
        lm = (levels > 0) & (pred == labels)
        ln += int(lm.sum()); lok += int((plvl[lm] == levels[lm]).sum())
        for ci, name in enumerate(CLASSES):
            cm = labels == ci
            if cm.any():
                a, b = per_class.get(name, (0, 0))
                per_class[name] = (a + int((pred[cm] == ci).sum()), b + int(cm.sum()))
    res = dict(tiles=tot, all_acc=ok / max(1, tot),
               building_acc=bok / max(1, bn), level_acc=lok / max(1, ln),
               per_class={k: v[0] / max(1, v[1]) for k, v in per_class.items()})
    print(f"\n=== held-out renders ({len(seeds)} bases, {tot:,} tiles) ===")
    print(f"  all tiles      {100*res['all_acc']:5.1f}%")
    print(f"  buildings only {100*res['building_acc']:5.1f}%")
    print(f"  level (of correctly typed) {100*res['level_acc']:5.1f}%")
    for k, v in sorted(res["per_class"].items(), key=lambda kv: kv[1]):
        print(f"     {k:<17s} {100*v:5.1f}%")
    return res


# ----------------------------------------------------------------------
# 7. Calibration on REAL screenshots
#
# Tested against an actual Clash screenshot pulled off the web, and the
# honest result is: AUTOMATIC CALIBRATION DOES NOT WORK ON REAL ART.
#
# Three approaches were tried and all three failed:
#   1. grass detection      -- most grass is under buildings on a built-up base
#   2. background subtract  -- the surround is ornate scenery, not flat colour
#   3. the yellow boundary  -- gold statues, lamps and decorative buildings
#                              outside the village are the same colour as the
#                              boundary line, and buildings break the line into
#                              fragments. Fitting the four edges gave slopes of
#                              0.2/0.45/1.14/0.08 where an isometric diamond
#                              must give -2/+2/+2/-2.
#
# So calibration is MANUAL, and that is fine, because it is a ONE-TIME cost:
# your zoom and resolution do not change between attacks, so you calibrate once
# and reuse the transform forever. Four clicks, then `save_calibration`.
#
# The geometry itself is exact -- grid<->screen round-trips with zero error --
# so once the four corners are right, every tile is right.
# ----------------------------------------------------------------------
def calibrate_from_clicks(top: Tuple[float, float], right: Tuple[float, float],
                          bottom: Tuple[float, float], left: Tuple[float, float]
                          ) -> IsoTransform:
    """Build the transform from the four village corners you clicked.

    Order is clockwise from the north vertex: top, right, bottom, left. Open the
    screenshot in any image viewer that shows pixel coordinates, hover the four
    tips of the diamond, and read them off.

    Then check it with `overlay_grid` -- if the lines sit on tile boundaries you
    are done, and you never have to do this again at that zoom level.
    """
    iso = IsoTransform.from_corners(top, right, bottom, left)
    ratio = iso.hw / max(iso.hh, 1e-9)
    if not (1.6 < ratio < 2.6):
        raise ValueError(
            f"tile aspect ratio {ratio:.2f} is not isometric (expected ~2.0). "
            "The four corners are probably in the wrong order or one is off -- "
            "they must be the diamond's tips, clockwise from the top.")
    return iso


def save_calibration(iso: IsoTransform, path: str = "vision_data/calib.json") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(ox=iso.ox, oy=iso.oy, hw=iso.hw, hh=iso.hh), f, indent=1)
    return path


def load_calibration(path: str = "vision_data/calib.json") -> IsoTransform:
    with open(path) as f:
        return IsoTransform(**json.load(f))


# ----------------------------------------------------------------------
# 8. Command line
# ----------------------------------------------------------------------


# ====================================================================
# ---- viz.py ------------------------------------------------------
"""
Base rendering and battle-replay animation.

Produces the same kind of artefact as the original project's
media/battle_replay.gif, but with the Royal Champion's position, her health,
her current target, the live threat map and the active Invisibility Spells all
drawn, so a replay is actually diagnosable instead of just decorative.
"""


from typing import Dict, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import animation


# ink-on-paper palette, readable in both light and dark contexts
CAT_COLORS = [
    "#EDE7DC",   # 0 empty
    "#C0392B",   # 1 town hall
    "#2E86AB",   # 2 air defense (harmless to her)
    "#B7791F",   # 3 high-value defense
    "#8A8177",   # 4 ordinary defense
    "#A6B5A0",   # 5 non-defense
    "#4A453E",   # 6 wall
]
CAT_LABELS = ["empty", "Town Hall", "Air Defense (air-only)",
              "High-value defense", "Defense", "Non-defense", "Wall"]


def _cmap():
    return mcolors.ListedColormap(CAT_COLORS)


def render_base(env, ax=None, title: str = "TH15 base", show_threat: bool = True):
    """Static picture of a generated base, optionally with the ground-threat map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(env.grid, cmap=_cmap(), vmin=0, vmax=6, interpolation="nearest")
    if show_threat:
        t = env._threat_channel()
        ax.contourf(t, levels=[1, 400, 900, 1600, 1e9], colors="#C0392B",
                    alpha=0.10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13)
    handles = [mpatches.Patch(color=CAT_COLORS[i], label=CAT_LABELS[i])
               for i in range(1, 7)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9)
    return ax


def animate_battle(frames: List[Dict], path: str = "battle_replay.gif",
                   fps: int = 10, title: str = "Royal Champion walk") -> str:
    """Render a recorded episode to a GIF.

    `frames` comes from RCWalkEnv(record=True).frames, or from
    evaluate.run_policy(..., record_best=True)["frames"].
    """
    if not frames:
        raise ValueError("no frames recorded -- run the env with record=True")

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    fig.patch.set_facecolor("#FFFFFF")
    im = ax.imshow(frames[0]["grid"], cmap=_cmap(), vmin=0, vmax=6,
                   interpolation="nearest")
    invis_im = ax.imshow(np.zeros((C.GRID_SIZE, C.GRID_SIZE, 4)),
                         interpolation="nearest")
    (hero,) = ax.plot([], [], "o", color="#111111", markersize=11,
                      markeredgecolor="#FFFFFF", markeredgewidth=1.6, zorder=5)
    (aim,) = ax.plot([], [], "-", color="#111111", lw=1.0, alpha=0.55, zorder=4)
    txt = ax.text(0.02, 0.985, "", transform=ax.transAxes, va="top", fontsize=10,
                  family="monospace",
                  bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec="#CCC7BC"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13)

    invis_rgba = np.zeros((C.GRID_SIZE, C.GRID_SIZE, 4))
    invis_rgba[..., 0] = 0.30
    invis_rgba[..., 1] = 0.65
    invis_rgba[..., 2] = 0.95

    def update(i):
        f = frames[i]
        im.set_data(f["grid"])
        invis_rgba[..., 3] = np.clip(f["invis"] / C.SPELL_DURATION, 0, 1) * 0.55
        invis_im.set_data(invis_rgba)
        x, y = f["pos"]
        hero.set_data([x], [y])
        if f.get("target"):
            tx, ty = f["target"]
            aim.set_data([x, tx], [y, ty])
        else:
            aim.set_data([], [])
        hp_pct = 100 * f["hp"] / C.RC_MAX_HP
        bar = "#" * int(hp_pct / 5) + "." * (20 - int(hp_pct / 5))
        txt.set_text(f"t {i*C.DT:5.1f}s\nHP {hp_pct:5.1f}% [{bar}]\n"
                     f"spells {f['spells']:2d}   TH {'DOWN' if f['th'] else 'up'}")
        return im, invis_im, hero, aim, txt

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path


def plot_training(csv_path: str, out_path: str = "training_curve.png"):
    """Plot the learning curve from metrics.csv.

    Worth looking at after every run. The old project's curve was flat for
    2,769 episodes and nobody plotted it.
    """
    import csv as _csv
    eps, ret, th, dest = [], [], [], []
    e_ep, e_win, e_dest = [], [], []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            if row["ret"]:
                eps.append(int(row["episode"])); ret.append(float(row["ret"]))
                th.append(float(row["th"])); dest.append(float(row["dest"]))
            if row["eval_win"]:
                e_ep.append(int(row["episode"])); e_win.append(float(row["eval_win"]))
                e_dest.append(float(row["eval_dest"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.6)

    axes[0].plot(eps, ret, lw=1.0, color="#2E86AB")
    axes[0].set_title("Return"); axes[0].set_xlabel("episode")

    axes[1].plot(eps, [100 * v for v in th], lw=1.0, color="#8A8177",
                 alpha=0.7, label="train")
    if e_ep:
        axes[1].plot(e_ep, [100 * v for v in e_win], lw=2.0, color="#C0392B",
                     label="greedy eval")
    axes[1].set_title("Town Hall destroyed (%)"); axes[1].set_xlabel("episode")
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].plot(eps, [100 * v for v in dest], lw=1.0, color="#8A8177",
                 alpha=0.7, label="train")
    if e_ep:
        axes[2].plot(e_ep, [100 * v for v in e_dest], lw=2.0, color="#B7791F",
                     label="greedy eval")
    axes[2].set_title("Destruction (%)"); axes[2].set_xlabel("episode")
    axes[2].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path
