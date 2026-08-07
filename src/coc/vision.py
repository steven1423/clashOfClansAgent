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

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C

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

# The shape of one tile on screen: hh / hw.
#
# This was 0.5 -- textbook 2:1 isometric -- for the whole first version of this
# module, and it is wrong. Clash does not render 2:1. Measured three
# independent ways on a real attack screenshot (iPhone, 2781x1280):
#
#   1. Hough over the deployment boundary and the wall runs: every grid-aligned
#      line has slope +/-0.750 exactly, and that slope IS hh/hw.
#   2. Fitting the 44x44 boundary diamond as a whole: hw 19.20, hh 14.40.
#   3. Autocorrelating wall-segment spacing along a run: hw 19.0.
#
# All three give 0.75, i.e. a 4:3 tile, and 2 and 3 agree on absolute size to
# within 1%. Rendering training data at 2:1 meant every base was drawn with its
# buildings packed closer together vertically than the game packs them, so the
# crops the classifier learned from had the wrong neighbours in them.
TILE_ASPECT = 0.75

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
    if iso.hw <= 1.0 or iso.hh <= 0.5 or not (1.05 < iso.hw / iso.hh < 1.65):
        return None            # Clash tiles are 4:3 (see TILE_ASPECT); reject junk
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
from . import sprites as S  # noqa: E402


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


_REAL_GROUND = None


def real_ground_bank(path: str = "vision_data/real/grass.npz"):
    """Ground tiles cut from actual attack screenshots.

    Procedural grass was never going to be right. Clash's ground is textured,
    has worn paths and scattered detail, and changes completely with the
    defender's SCENERY -- and Supercell has shipped dozens of those, with more
    every season. Enumerating them from the wiki is a losing race.

    Cutting the ground out of real screenshots wins it instead: exact texture,
    exact lighting, exact compression, exact scale, and it automatically covers
    whichever sceneries actually turn up in your matches rather than the ones I
    guessed at. 290 tiles came out of the first 14 screenshots.
    """
    global _REAL_GROUND
    if _REAL_GROUND is None:
        _REAL_GROUND = (np.load(path)["X"] if os.path.isfile(path)
                        else np.zeros((0, 48, 48, 3), np.uint8))
    return _REAL_GROUND


def _ground_real(w: int, h: int, iso: IsoTransform, rng, bank) -> Optional[np.ndarray]:
    """Mosaic the floor out of real ground crops, scaled to this render."""
    if len(bank) == 0:
        return None
    from PIL import Image
    # a harvested crop covers CROP_W_TILES tiles of a 23.5-px-tile screenshot
    tile_px = max(8, int(round(2 * iso.hw * CROP_W_TILES)))
    canvas = Image.new("RGB", (w, h))
    for y in range(0, h, tile_px):
        for x in range(0, w, tile_px):
            g = bank[rng.randrange(len(bank))]
            im = Image.fromarray(g).resize((tile_px, tile_px), Image.BILINEAR)
            if rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            canvas.paste(im, (x, y))
    out = np.asarray(canvas, np.float32) / 255.0
    # darken outside the playable diamond so the border still reads
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs - iso.ox) / iso.hw
    v = (ys - iso.oy) / iso.hh
    gx = (u + v) * 0.5
    gy = (v - u) * 0.5
    inside = (gx >= 0) & (gx < C.GRID_SIZE) & (gy >= 0) & (gy < C.GRID_SIZE)
    return np.where(inside[..., None], out, out * 0.55)


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
    from .base import generate_base

    rng = _random.Random(seed)
    if lib is None:
        lib = get_library()

    grid, buildings, traps, cc_pos, altar = generate_base(
        1.0, seed=seed, traps=True, cc=True, hero=True)

    H = int(px * 0.62)
    zoom = rng.uniform(0.86, 1.05) if augment else 1.0
    hw = px / (2.0 * C.GRID_SIZE) * zoom
    hh = hw * TILE_ASPECT
    ox = px * 0.5 + (rng.uniform(-0.03, 0.03) * px if augment else 0.0)
    oy = (H - 2 * C.GRID_SIZE * hh) * 0.5 + (rng.uniform(-0.02, 0.02) * H
                                             if augment else 0.0)
    iso = IsoTransform(ox=ox, oy=oy, hw=hw, hh=hh)

    canvas = _ground_real(px, H, iso, rng, real_ground_bank())
    if canvas is None:
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
    from .base import (Building, CAT_AIR_DEFENSE, CAT_DEFENSE, CAT_HIGH_DEFENSE,
                       CAT_NON_DEFENSE, CAT_TOWN_HALL, CAT_WALL)
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
    from .army import arrow_value_map
    from .full_env import A_DEPLOY, FullAttackEnv, N_CH, N_SC
    from .model import RCQNet
    from .train import pick_device

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
                     device: str = "auto", val_frac: float = 0.15,
                     resume: bool = True) -> Dict:
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
    from .train import pick_device

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

    # Checkpoint every epoch and resume from it. Not optional politeness: this
    # trains in a sandbox that can be reclaimed between turns, and a twelve-epoch
    # run that dies at epoch three with nothing on disk has to start over. Costs
    # 1.8 MB a write.
    start_ep = 0
    hist: List[Dict] = []
    if resume and os.path.isfile(out):
        try:
            blob = torch.load(out, map_location=dev, weights_only=False)
            if blob.get("level_vocab") == vocab and blob.get("epochs_total") == epochs:
                net.load_state_dict(blob["state"])
                if "opt" in blob:
                    opt.load_state_dict(blob["opt"])
                if "sched" in blob:
                    sched.load_state_dict(blob["sched"])
                start_ep = int(blob.get("epoch", 0))
                hist = list(blob.get("history", []))
                print(f"  resuming from {out} at epoch {start_ep}")
        except Exception as e:
            print(f"  could not resume ({e}); starting fresh")

    counts = np.bincount(Y, minlength=len(CLASSES)).astype(np.float32)
    wt = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5,
                      dtype=torch.float32, device=dev)
    type_loss = nn.CrossEntropyLoss(weight=wt)
    level_loss = nn.CrossEntropyLoss()
    level_ids = {CLASS_INDEX[c] for c in LEVEL_CLASSES}
    lvl_mask_np = np.isin(Y, list(level_ids))
    print(f"  {nparam:,} parameters; {lvl_mask_np.sum():,} tiles carry a level")

    for ep in range(start_ep, epochs):
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
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(dict(state=net.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), classes=CLASSES,
                        level_vocab=vocab, history=hist, epoch=ep + 1,
                        epochs_total=epochs), out)

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
    from .train import pick_device
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
    if not (1.05 < ratio < 1.65):
        raise ValueError(
            f"tile aspect ratio {ratio:.2f} is not right (Clash renders 4:3, so "
            f"expect ~{1/TILE_ASPECT:.2f}). The four corners are probably in the "
            "wrong order or one is off -- they must be the diamond's tips, "
            "clockwise from the top.")
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
# 9. Calibrating a REAL attack screenshot
#
# Section 7 says automatic calibration does not work. On a home-village
# screenshot that is still true. On an ATTACK screenshot it is not, because the
# attack view has two things the home village does not: a red dashed deployment
# boundary that traces the grid exactly, and walls whose ice caps sit one per
# tile in dead-straight runs.
#
# Scale comes from the walls and position comes from the boundary, and it is
# worth being clear about why that split rather than the other way round.
#
# I measured the tile size four ways before getting it right:
#
#   autocorrelating wall spacing along one run    19.0   WRONG -- a harmonic
#   fitting the boundary diamond by peak support  12.1   WRONG -- interior walls
#   boundary edge extent from the south vertex    23.7 / 24.7
#   grid-alignment sweep of the wall texture      23.60
#   COUNTING wall cap blocks along long runs      23.5 - 23.8
#
# The first is the instructive failure. Autocorrelation reports octave errors
# the same way a naive pitch detector does, and it did it twice, and both times
# it agreed with itself across two independent directions, which is exactly what
# made it convincing. Counting cannot alias. When a periodic measurement matters,
# count the things.
# ----------------------------------------------------------------------
TILE_SLOPE = TILE_ASPECT          # grid-aligned lines have slope +/- hh/hw


def _components(mask: np.ndarray, min_area: int = 30, max_area: int = 2000):
    """Connected components, returned as centroids. No scipy, no OpenCV --
    this module already runs anywhere numpy does and it is not worth changing
    that for one flood fill."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    out = []
    cur = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if lab[sy, sx]:
            continue
        cur += 1
        stack = [(sy, sx)]
        lab[sy, sx] = cur
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    stack.append((ny, nx))
            if len(pix) > max_area:
                break
        if min_area <= len(pix) <= max_area:
            a = np.array(pix, dtype=np.float64)
            out.append((a[:, 1].mean(), a[:, 0].mean()))
    return np.array(out) if out else np.zeros((0, 2))


def wall_cap_scale(img: np.ndarray) -> Optional[float]:
    """Tile half-width from the spacing of wall cap blocks.

    Each wall segment is one tile and carries one pale ice cap on top, so caps
    along a straight run are exactly hw apart in x. Find the runs, take the
    median gap on the longest ones. Counting blocks, not correlating them.
    """
    a = img[..., :3].astype(np.int16)
    h, w = a.shape[:2]
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    cap = (B > 190) & (R > 150) & (G > 180) & (B - R > 10)
    cap[int(h * 0.72):, :] = False
    cap[:int(h * 0.05), :] = False
    pts = _components(cap, 40, 1200)
    if len(pts) < 30:
        return None
    best: List[Tuple[int, float]] = []
    for s in (TILE_SLOPE, -TILE_SLOPE):
        c = pts[:, 1] - s * pts[:, 0]
        lo, hi = c.min(), c.max()
        nb = max(2, int((hi - lo) / 4) + 1)
        hist, edges = np.histogram(c, bins=nb, range=(lo, hi))
        for bi in np.argsort(hist)[::-1][:6]:
            sel = np.abs(c - edges[bi]) < 6
            if sel.sum() < 8:
                continue
            px = np.sort(pts[sel, 0])
            d = np.diff(px)
            d = d[(d > 4) & (d < 80)]
            if len(d) < 6:
                continue
            spread = float(np.percentile(d, 75) - np.percentile(d, 25))
            # A clean run has tightly clustered gaps. A run where two caps
            # merged shows a bimodal gap distribution and a median 1.5x too
            # big, which is how the 35 px "runs" in the first attempt happened.
            if spread > 9.0:
                continue
            best.append((int(sel.sum()), float(np.median(d))))
    if not best:
        return None
    best.sort(reverse=True)                    # longest runs first
    top = [v for _, v in best[:4]]
    return float(np.median(top))


def _boundary_edges(img: np.ndarray, min_span: float = 500.0):
    """The two outermost deployment-boundary lines, one per grid direction."""
    a = img[..., :3].astype(np.int16)
    h, w = a.shape[:2]
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (R > 110) & (R - G > 45) & (R - B > 45)
    m[int(h * 0.74):, :] = False
    m[:int(h * 0.05), :] = False
    m[:int(h * 0.30), :int(w * 0.14)] = False      # loot panel
    m[:int(h * 0.22), int(w * 0.78):] = False      # resource panel
    ys, xs = np.nonzero(m)
    if len(xs) < 1500:
        return None
    ys = ys.astype(float)
    xs = xs.astype(float)
    out = {}
    for key, s in (("p", TILE_SLOPE), ("n", -TILE_SLOPE)):
        c = ys - s * xs
        lo = c.min()
        nb = int(c.max() - lo) + 1
        b = (c - lo).astype(int)
        cnt = np.bincount(b, minlength=nb).astype(float)
        cnt = np.convolve(cnt, np.ones(5), mode="same")
        xmin = np.full(nb, 1e9)
        xmax = np.full(nb, -1e9)
        np.minimum.at(xmin, b, xs)
        np.maximum.at(xmax, b, xs)
        ok = []
        for i in range(2, nb - 2):
            if cnt[i] < 25:
                continue
            # span, not support: a boundary edge runs the width of the base,
            # a red bush does not. Support alone picks interior walls.
            span = xmax[max(0, i - 2):i + 3].max() - xmin[max(0, i - 2):i + 3].min()
            if span >= min_span:
                ok.append(i)
        if not ok:
            return None
        out[key] = lo + max(ok)        # outermost on the low side of the screen
    return out["p"], out["n"]


def calibrate_attack_view(img: np.ndarray, n: int = C.GRID_SIZE,
                          hw: Optional[float] = None
                          ) -> Optional[IsoTransform]:
    """Screenshot of an attack -> exact grid, no clicking.

    Scale from the wall caps, position from the south vertex of the deployment
    boundary. Returns None rather than a guess if either fails -- a wrong grid
    is worse than no grid, because everything downstream looks like it worked.

    KNOWN LIMIT: the scale step reads the pale cap on top of each wall block,
    and different wall SKINS have differently-coloured caps. On a sample of 14
    real attack screenshots it solved 8; the six it refused were all bases
    wearing a wall skin it does not know. Two skin-agnostic replacements were
    tried and both counted the walls' internal texture instead of the blocks
    (returning roughly half a tile, and inconsistently). So rather than ship a
    detector that is sometimes silently wrong, pass `hw` explicitly for those:
    one number, measured once per zoom level, and the origin is still automatic.
    """
    if hw is None:
        hw = wall_cap_scale(img)
    if hw is None or not (8.0 < hw < 60.0):
        return None
    hh = hw * TILE_ASPECT
    edges = _boundary_edges(img)
    if edges is None:
        return None
    cp, cn = edges                  # y = +s x + cp   and   y = -s x + cn
    s = TILE_SLOPE
    sx = (cn - cp) / (2 * s)        # their intersection is grid (n, n)
    sy = s * sx + cp
    return IsoTransform(ox=sx, oy=sy - 2 * n * hh, hw=hw, hh=hh)


# ----------------------------------------------------------------------
# 10. Building DETECTION
#
# Sections 2-6 classify tiles. That framing was inherited from the simulator's
# 44x44 grid and it is the wrong one for real screenshots, for a reason that
# took four failed labelling attempts to see clearly:
#
#   A tile crop is CROP_W_TILES wide -- more than two tiles -- so it contains
#   the tile plus its neighbours. Two crops centred on the same Cannon with
#   different things beside them look LESS alike than two crops of different
#   buildings with similar surroundings. The label describes the middle of the
#   crop; the pixels are dominated by the edges.
#
# On synthetic data that is survivable, because the same generator makes both
# the images and the labels and the network can fit the whole joint
# distribution. That is exactly why it scores 97% there and collapses on a real
# screenshot -- it learned neighbourhoods, not buildings.
#
# So: detect buildings, do not classify tiles. Crop centred on a CANDIDATE
# BUILDING, sized to that building's footprint, and ask what it is. A crop
# framed on a building looks like that building whatever is beside it. It also
# normalises scale for free -- a 3x3 crop and a 4x4 crop both resize to the same
# input, so a Cannon is the same size to the network at any zoom level.
# ----------------------------------------------------------------------
DET_CLASSES = [c for c in CLASSES if c not in ("Empty", "Wall")] + ["None"]
DET_INDEX = {n: i for i, n in enumerate(DET_CLASSES)}
DET_SIZE = 64
DET_PAD = 1.30          # crop width as a multiple of the footprint diamond
DET_TALL = 1.15         # crop height as a multiple of its width
DET_DROP = 0.30         # fraction of the crop below the diamond's bottom vertex


def building_box(iso: IsoTransform, gx: int, gy: int, size: int
                 ) -> Tuple[int, int, int, int]:
    """Pixel box framing a candidate building of `size` tiles at (gx, gy).

    Anchored on the ground diamond: centre x from the footprint's middle,
    bottom from its near vertex, then padded out and up because the art rises.
    """
    cx, _ = iso.to_screen(gx + size / 2.0, gy + size / 2.0)
    _, by = iso.to_screen(gx + size, gy + size)
    w = 2.0 * size * iso.hw * DET_PAD
    h = w * DET_TALL
    return (int(round(cx - w / 2)), int(round(by + DET_DROP * h - h)),
            int(round(w)), int(round(h)))


def crop_building(pil, iso: IsoTransform, gx: int, gy: int, size: int):
    x, y, w, h = building_box(iso, gx, gy, size)
    if w < 4 or h < 4:
        return None
    return np.asarray(pil.crop((x, y, x + w, y + h)).resize((DET_SIZE, DET_SIZE)),
                      dtype=np.uint8)[..., :3]


def build_detector(n_classes: int = len(DET_CLASSES), n_levels: int = 0):
    """CNN over building-centred crops. Same two heads as the tile classifier."""
    import torch.nn as nn

    def blk(i, o, p=True):
        layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                  nn.ReLU(inplace=True)]
        if p:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    class DetNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128),
                                   blk(128, 192), blk(192, 192),
                                   nn.AdaptiveAvgPool2d(2), nn.Flatten())
            self.trunk = nn.Sequential(nn.Linear(192 * 4, 320),
                                       nn.ReLU(inplace=True), nn.Dropout(0.15))
            self.type_head = nn.Linear(320, n_classes)
            self.level_head = nn.Linear(320, n_levels) if n_levels else None

        def forward(self, x):
            z = self.trunk(self.f(x))
            if self.level_head is None:
                return self.type_head(z)
            return self.type_head(z), self.level_head(z)

    return DetNet()


def make_detector_dataset(n_bases: int = 40,
                          out: str = "vision_data/detector.npz",
                          px: int = 1600, augment: bool = True,
                          neg_per_base: int = 90, seed0: int = 0) -> str:
    """Positives are every building, cropped on its own footprint. Negatives are
    positions with no building there -- including OFFSET positions half a tile
    off a real building, which is what teaches the detector to localise rather
    than merely notice that something is nearby."""
    import random as _random
    from PIL import Image as I
    lib = get_library()
    rng = _random.Random(99)
    X, Y, L = [], [], []
    for i in range(n_bases):
        img, labels, levels, iso = render_synthetic(seed=seed0 + i, px=px,
                                                    lib=lib, augment=augment)
        pil = I.fromarray(img)
        from .base import generate_base
        _, buildings, _, _, _ = generate_base(1.0, seed=seed0 + i, traps=True,
                                              cc=True, hero=True)
        occupied = np.zeros((C.GRID_SIZE, C.GRID_SIZE), bool)
        for b in buildings:
            occupied[b.y:b.y + b.h, b.x:b.x + b.w] = True
            cls = b.name if b.name in CLASS_INDEX else "Non-Defense"
            if cls == "Wall":
                continue
            c = crop_building(pil, iso, b.x, b.y, max(b.w, b.h))
            if c is None:
                continue
            X.append(c)
            Y.append(DET_INDEX[cls])
            L.append(int(levels[b.y, b.x]) if cls in LEVEL_CLASSES else 0)
            # a near-miss negative: same building, one tile off
            if rng.random() < 0.5:
                ox, oy = rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                c2 = crop_building(pil, iso, b.x + ox, b.y + oy, max(b.w, b.h))
                if c2 is not None:
                    X.append(c2)
                    Y.append(DET_INDEX["None"])
                    L.append(0)
        for _ in range(neg_per_base):
            s = rng.choice([2, 3, 4])
            gx = rng.randrange(0, C.GRID_SIZE - s)
            gy = rng.randrange(0, C.GRID_SIZE - s)
            if occupied[gy:gy + s, gx:gx + s].mean() > 0.25:
                continue
            c = crop_building(pil, iso, gx, gy, s)
            if c is not None:
                X.append(c)
                Y.append(DET_INDEX["None"])
                L.append(0)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_bases} bases, {len(X):,} crops", flush=True)
    X = np.stack(X)
    Y = np.array(Y, np.int64)
    L = np.array(L, np.int64)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, X=X, Y=Y, L=L, classes=np.array(DET_CLASSES),
             level_vocab=np.array(LEVEL_VOCAB))
    print(f"  {out}: {len(X):,} crops ({100*(Y==DET_INDEX['None']).mean():.0f}% "
          f"negative), {X.nbytes/1e6:.0f} MB")
    return out


# ----------------------------------------------------------------------
# 8. Command line
# ----------------------------------------------------------------------
def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m coc.vision",
        description="Read a Clash base off a screenshot and plan the attack.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset", help="render training data from real sprites")
    d.add_argument("--bases", type=int, default=45)
    d.add_argument("--out", default="vision_data/synthetic.npz")
    d.add_argument("--px", type=int, default=1600)
    d.add_argument("--no-augment", action="store_true")

    t = sub.add_parser("train", help="train the tile classifier")
    t.add_argument("--npz", default="vision_data/synthetic.npz")
    t.add_argument("--out", default="vision_data/tilenet.pt")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--device", default="auto")

    e = sub.add_parser("check", help="score the classifier on held-out renders")
    e.add_argument("--model", default="vision_data/tilenet.pt")
    e.add_argument("--bases", type=int, default=5)

    pv = sub.add_parser("preview", help="save one rendered base to look at")
    pv.add_argument("--seed", type=int, default=0)
    pv.add_argument("--out", default="vision_data/preview.png")
    pv.add_argument("--px", type=int, default=1600)
    pv.add_argument("--grid", action="store_true", help="overlay the tile grid")

    c = sub.add_parser("calibrate",
                       help="build a transform from the four village corners")
    c.add_argument("--top", nargs=2, type=float, required=True)
    c.add_argument("--right", nargs=2, type=float, required=True)
    c.add_argument("--bottom", nargs=2, type=float, required=True)
    c.add_argument("--left", nargs=2, type=float, required=True)
    c.add_argument("--image", default=None,
                   help="screenshot to draw the checking overlay on")
    c.add_argument("--out", default="vision_data/calib.json")

    rd = sub.add_parser("read", help="recognise a real screenshot")
    rd.add_argument("image")
    rd.add_argument("--model", default="vision_data/tilenet.pt")
    rd.add_argument("--calib", default="vision_data/calib.json")
    rd.add_argument("--agent", default=None,
                    help="path to the trained attack agent, e.g. "
                         "runs/full/ckpt_best.pt -- adds the plan")
    rd.add_argument("--min-conf", type=float, default=0.5)
    rd.add_argument("--overlay", default=None)

    a = p.parse_args(argv)

    if a.cmd == "dataset":
        make_dataset(n_bases=a.bases, out=a.out, px=a.px,
                     augment=not a.no_augment)
    elif a.cmd == "train":
        train_classifier(npz=a.npz, out=a.out, epochs=a.epochs,
                         batch=a.batch, lr=a.lr, device=a.device)
    elif a.cmd == "check":
        evaluate_on_render(model=a.model,
                           seeds=tuple(range(900, 900 + a.bases)))
    elif a.cmd == "preview":
        from PIL import Image
        img, labels, levels, iso = render_synthetic(seed=a.seed, px=a.px)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        if a.grid:
            overlay_grid(img, iso, a.out)
        else:
            Image.fromarray(img).save(a.out)
        print(f"wrote {a.out}")
    elif a.cmd == "calibrate":
        iso = calibrate_from_clicks(tuple(a.top), tuple(a.right),
                                    tuple(a.bottom), tuple(a.left))
        print(f"  tile half-width {iso.hw:.2f} px, half-height {iso.hh:.2f} px "
              f"(ratio {iso.hw/iso.hh:.2f}, isometric is 2.00)")
        print(f"  saved {save_calibration(iso, a.out)}")
        if a.image:
            from PIL import Image
            img = np.asarray(Image.open(a.image).convert("RGB"))
            out = os.path.splitext(a.image)[0] + "_grid.png"
            overlay_grid(img, iso, out)
            print(f"  CHECK THIS: {out} -- the lines must sit on tile edges")
    elif a.cmd == "read":
        from PIL import Image
        img = np.asarray(Image.open(a.image).convert("RGB"))
        iso = load_calibration(a.calib) if os.path.isfile(a.calib) else None
        grid, levels, iso = recognise(img, model=a.model, iso=iso,
                                      min_conf=a.min_conf)
        buildings = grid_to_env(grid, levels)
        print(f"\n=== read {a.image} ===")
        print(describe_base(buildings))
        if a.overlay:
            overlay_grid(img, iso, a.overlay)
            print(f"\n  overlay {a.overlay}")
        if a.agent:
            plan = plan_attack([b for b in buildings], a.agent)
            print("\n=== plan ===")
            print(f"  drop the Royal Champion at tile {plan['deploy_champion']}")
            print(f"  Giant Arrow from tile {plan['giant_arrow_tile']} "
                  f"-> {plan['giant_arrow_air_defenses']:.1f} Air Defenses on the line")
            print(f"  bring the dragons in from tile {plan['dragon_entry']} "
                  "(least Air Sweeper coverage)")
            print("  key defenses:")
            for name, x, y in plan["key_defenses"]:
                print(f"     {name:<17s} at ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()


def train_detector(npz: str = "vision_data/detector.npz",
                   out: str = "vision_data/detnet.pt",
                   epochs: int = 14, batch: int = 128, lr: float = 2e-3,
                   device: str = "auto", val_frac: float = 0.15,
                   resume: bool = True) -> Dict:
    """Train the building detector. Checkpoints every epoch and resumes."""
    import torch
    import torch.nn as nn
    from .train import pick_device

    d = np.load(npz, allow_pickle=True)
    X, Y = d["X"], d["Y"]
    L = d["L"] if "L" in d.files else np.zeros_like(Y)
    vocab = [str(v) for v in d["level_vocab"]]
    dev = pick_device(device)
    rs = np.random.RandomState(0)
    idx = rs.permutation(len(X))
    sp = int(len(X) * (1 - val_frac))
    tr, va = idx[:sp], idx[sp:]
    print(f"  {len(X):,} crops ({len(tr):,} train / {len(va):,} val)  device {dev}")

    net = build_detector(len(DET_CLASSES), len(vocab)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * (len(tr) // batch + 1)))
    start_ep, hist = 0, []
    if resume and os.path.isfile(out):
        try:
            b = torch.load(out, map_location=dev, weights_only=False)
            if b.get("epochs_total") == epochs and b.get("classes") == DET_CLASSES:
                net.load_state_dict(b["state"]); opt.load_state_dict(b["opt"])
                sched.load_state_dict(b["sched"]); start_ep = int(b["epoch"])
                hist = list(b.get("history", []))
                print(f"  resuming at epoch {start_ep}")
        except Exception as e:
            print(f"  fresh start ({e})")

    cnt = np.bincount(Y, minlength=len(DET_CLASSES)).astype(np.float32)
    wt = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, device=dev)
    lossf = nn.CrossEntropyLoss(weight=wt)
    lvlf = nn.CrossEntropyLoss()
    lvl_ids = {DET_INDEX[c] for c in LEVEL_CLASSES}
    lm_all = np.isin(Y, list(lvl_ids))

    for ep in range(start_ep, epochs):
        net.train(); rs.shuffle(tr); tot = 0.0
        for i in range(0, len(tr), batch):
            b = tr[i:i + batch]
            xb = torch.from_numpy(X[b]).float().permute(0, 3, 1, 2).to(dev) / 255.
            yb = torch.from_numpy(Y[b]).to(dev)
            lb = torch.from_numpy(L[b]).to(dev)
            mb = torch.from_numpy(lm_all[b]).to(dev)
            pt, pl = net(xb)
            loss = lossf(pt, yb)
            if bool(mb.any()):
                loss = loss + 0.5 * lvlf(pl[mb], lb[mb])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss.detach()) * len(b)
        net.eval()
        pred = np.zeros(len(va), np.int64); predl = np.zeros(len(va), np.int64)
        with torch.no_grad():
            for i in range(0, len(va), 512):
                b = va[i:i + 512]
                xb = torch.from_numpy(X[b]).float().permute(0, 3, 1, 2).to(dev) / 255.
                pt, pl = net(xb)
                pred[i:i + len(b)] = pt.argmax(1).cpu().numpy()
                predl[i:i + len(b)] = pl.argmax(1).cpu().numpy()
        yv = Y[va]
        acc = float((pred == yv).mean())
        pos = yv != DET_INDEX["None"]
        pacc = float((pred[pos] == yv[pos]).mean()) if pos.any() else 0.0
        neg = ~pos
        nacc = float((pred[neg] == yv[neg]).mean()) if neg.any() else 0.0
        lm = lm_all[va] & (pred == yv)
        lacc = float((predl[lm] == L[va][lm]).mean()) if lm.any() else 0.0
        hist.append(dict(epoch=ep + 1, loss=tot / max(1, len(tr)), acc=acc,
                         building=pacc, reject=nacc, level=lacc))
        print(f"  epoch {ep+1}/{epochs}  loss {hist[-1]['loss']:.4f}  "
              f"all {100*acc:5.1f}%  building {100*pacc:5.1f}%  "
              f"reject-empty {100*nacc:5.1f}%  level {100*lacc:5.1f}%", flush=True)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(dict(state=net.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), classes=DET_CLASSES,
                        level_vocab=vocab, history=hist, epoch=ep + 1,
                        epochs_total=epochs), out)
    print("\n  per-class recall (validation):")
    for ci, name in enumerate(DET_CLASSES):
        m = yv == ci
        if m.any():
            print(f"     {name:<17s} n={int(m.sum()):5d}  {100*float((pred[m]==ci).mean()):5.1f}%")
    return dict(path=out, history=hist)


def detect_buildings(img: np.ndarray, iso: IsoTransform,
                     model: str = "vision_data/detnet.pt",
                     device: str = "auto", min_conf: float = 0.5):
    """Slide the detector over every legal footprint and keep the best
    non-overlapping placements, capped by the TH15 roster.

    This replaces grid_to_env's tile-vote reassembly. The roster cap does the
    same job it did there -- a base has exactly seven Cannons, so the eighth
    best Cannon window is wrong however confident it looks.
    """
    import torch
    from PIL import Image as I
    from .train import pick_device
    dev = pick_device(device)
    blob = torch.load(model, map_location=dev, weights_only=False)
    vocab = blob.get("level_vocab", LEVEL_VOCAB)
    net = build_detector(len(blob["classes"]), len(vocab)).to(dev)
    net.load_state_dict(blob["state"]); net.eval()
    pil = I.fromarray(img.astype(np.uint8))

    cands, meta = [], []
    for size in (2, 3, 4):
        for gy in range(0, C.GRID_SIZE - size + 1):
            for gx in range(0, C.GRID_SIZE - size + 1):
                c = crop_building(pil, iso, gx, gy, size)
                if c is not None:
                    cands.append(c); meta.append((gx, gy, size))
    Xc = np.stack(cands)
    probs = np.zeros((len(Xc), len(DET_CLASSES)), np.float32)
    lvls = np.zeros(len(Xc), np.int64)
    with torch.no_grad():
        for i in range(0, len(Xc), 512):
            xb = torch.from_numpy(Xc[i:i + 512]).float().permute(0, 3, 1, 2).to(dev) / 255.
            pt, pl = net(xb)
            probs[i:i + len(xb)] = torch.softmax(pt, 1).cpu().numpy()
            lvls[i:i + len(xb)] = pl.argmax(1).cpu().numpy()

    taken = np.zeros((C.GRID_SIZE, C.GRID_SIZE), bool)
    found = []
    none_i = DET_INDEX["None"]
    order = sorted(range(len(Xc)), key=lambda i: -probs[i].max())
    counts: Dict[str, int] = {}
    for i in order:
        k = int(probs[i].argmax())
        if k == none_i or probs[i, k] < min_conf:
            continue
        name = DET_CLASSES[k]
        gx, gy, size = meta[i]
        if size != FOOTPRINT.get(name, size):
            continue
        if counts.get(name, 0) >= ROSTER.get(name, 40):
            continue
        if taken[gy:gy + size, gx:gx + size].any():
            continue
        taken[gy:gy + size, gx:gx + size] = True
        counts[name] = counts.get(name, 0) + 1
        tag = vocab[int(lvls[i])] if name in LEVEL_CLASSES else ""
        found.append(dict(name=name, x=gx, y=gy, size=size,
                          conf=float(probs[i, k]),
                          level=tag.split(":", 1)[1] if ":" in tag else ""))
    return found


# ----------------------------------------------------------------------
# 11. Merging partial screenshots into one base
#
# One screenshot cannot hold a whole TH15 base at attack zoom, so a base
# arrives as two or more overlapping views. The merge happens in GRID space,
# not pixel space, and the reason it is trivial there is worth stating: the
# deployment boundary's south vertex is grid (44, 44) in EVERY view of the same
# base, so each image's calibration already expresses its buildings in the same
# shared coordinate system. No image stitching, no feature matching between
# views, no parallax to fight -- pixel stitching would have to reconcile two
# different camera positions looking at 3D art, which is exactly the hard
# problem that working in grid coordinates deletes.
# ----------------------------------------------------------------------
def merge_detections(per_image: Sequence[Sequence[Dict]]) -> List[Dict]:
    """Union the per-image detections, resolving disagreements by confidence.

    Two views of the same building land on the same (x, y) up to a tile of
    calibration error, so detections whose footprints overlap are the same
    building: keep the more confident reading. Roster caps apply to the merged
    result -- each view honoured them alone, the union must too.
    """
    taken = np.zeros((C.GRID_SIZE, C.GRID_SIZE), bool)
    counts: Dict[str, int] = {}
    out: List[Dict] = []
    pool = sorted((d for ds in per_image for d in ds),
                  key=lambda d: -d["conf"])
    for d in pool:
        s = d["size"]
        x, y = d["x"], d["y"]
        if taken[y:y + s, x:x + s].any():
            continue
        if counts.get(d["name"], 0) >= ROSTER.get(d["name"], 40):
            continue
        taken[y:y + s, x:x + s] = True
        counts[d["name"]] = counts.get(d["name"], 0) + 1
        out.append(dict(d))
    return out


def coverage_report(isos: Sequence[IsoTransform],
                    shapes: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Which of the 44x44 tiles is actually visible in at least one image?

    A tile counts as seen if its centre projects inside the frame with margin
    for the crop. The report is what tells the user "one more shot, panned
    up-right" instead of silently reading half a base.
    """
    seen = np.zeros((C.GRID_SIZE, C.GRID_SIZE), bool)
    for iso, (h, w) in zip(isos, shapes):
        for gy in range(C.GRID_SIZE):
            for gx in range(C.GRID_SIZE):
                sx, sy = iso.to_screen(gx + 0.5, gy + 0.5)
                if 60 <= sx < w - 60 and 60 <= sy < h * 0.74:
                    seen[gy, gx] = True
    return seen


def read_base(paths: Sequence[str], model: str = "vision_data/detnet.pt",
              hw: Optional[float] = None, device: str = "auto",
              min_conf: float = 0.5):
    """The whole pipeline: N screenshots of one base -> one building list.

    Returns (detections, seen_mask, isos). Prints what is missing if coverage
    has holes, because a plan built on half a base is worse than no plan.
    """
    from PIL import Image as I
    dets, isos, shapes = [], [], []
    for p in paths:
        img = np.asarray(I.open(p).convert("RGB"))
        iso = calibrate_attack_view(img) or (
            calibrate_attack_view(img, hw=hw) if hw else None)
        if iso is None:
            print(f"  !! {os.path.basename(p)}: cannot calibrate, skipped")
            continue
        isos.append(iso)
        shapes.append(img.shape[:2])
        dets.append(detect_buildings(img, iso, model=model, device=device,
                                     min_conf=min_conf))
    merged = merge_detections(dets)
    seen = coverage_report(isos, shapes)
    miss = int((~seen).sum())
    if miss:
        ys, xs = np.nonzero(~seen)
        side_y = "top" if ys.mean() < C.GRID_SIZE / 2 else "bottom"
        side_x = "left" if xs.mean() < C.GRID_SIZE / 2 else "right"
        print(f"  coverage: {seen.mean()*100:.0f}% of the grid; {miss} tiles "
              f"unseen, mostly {side_y}-{side_x} -- take one more shot panned "
              f"toward the {side_y} {side_x}")
    return merged, seen, isos
