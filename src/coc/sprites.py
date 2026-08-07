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

from __future__ import annotations

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
