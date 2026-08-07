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
