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

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import config as C


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
