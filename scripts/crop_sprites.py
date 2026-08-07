#!/usr/bin/env python3
"""
Turn a pair of grid screenshots into a sprite library with exact transparency.

The browser can render the wiki's building art but cannot hand over the PNG's
alpha channel -- a screenshot is flattened pixels. So the grid is captured
twice, once over white and once over black, and the alpha is recovered
algebraically:

    over white   Cw = a*C + (1-a)*1
    over black   Cb = a*C + (1-a)*0 = a*C

    =>  a = 1 - (Cw - Cb)        (per channel; averaged over the three)
        C = Cb / a               (where a > 0)

This is exact up to capture noise, and it recovers the soft stuff -- the drop
shadows and the glow around an Inferno's beam -- rather than hard-keying them
away, which matters because those soft edges are exactly what a naive
background-key would turn into a halo the classifier then learns to look for.

    python scripts/crop_sprites.py white.png black.png --files a.png b.png ... \
        --cols 8 --out vision_data/sprites
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from coc import sprites as S  # noqa: E402


def recover(white: np.ndarray, black: np.ndarray) -> np.ndarray:
    """RGBA float in [0,1] from the two flattened composites."""
    a = 1.0 - np.clip(white - black, 0.0, 1.0)
    a = a.mean(axis=2)                      # the three channels must agree
    a = np.clip(a, 0.0, 1.0)
    rgb = np.zeros_like(black)
    m = a > 1e-3
    rgb[m] = black[m] / a[m][:, None]
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.dstack([rgb, a])


def bbox(alpha: np.ndarray, thr: float = 0.02):
    ys, xs = np.where(alpha > thr)
    if not len(ys):
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("white")
    p.add_argument("black")
    p.add_argument("--files", nargs="+", required=True,
                   help="wiki file names, in grid order (row-major)")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--out", default="vision_data/sprites")
    p.add_argument("--pad", type=int, default=2)
    a = p.parse_args()

    W = np.asarray(Image.open(a.white).convert("RGB"), dtype=np.float32) / 255.0
    B = np.asarray(Image.open(a.black).convert("RGB"), dtype=np.float32) / 255.0
    if W.shape != B.shape:
        raise SystemExit(f"shape mismatch {W.shape} vs {B.shape}")

    rows = int(np.ceil(len(a.files) / a.cols))
    ch, cw = W.shape[0] / rows, W.shape[1] / a.cols
    print(f"grid {a.cols}x{rows}, cell {cw:.1f}x{ch:.1f} px, "
          f"capture {W.shape[1]}x{W.shape[0]}")

    os.makedirs(a.out, exist_ok=True)
    index_path = os.path.join(a.out, "index.json")
    index = json.load(open(index_path)) if os.path.isfile(index_path) else {}

    lookup = {f: (c, L) for c, L, f in S.MANIFEST}
    worst = 0.0
    for i, fname in enumerate(a.files):
        r, c = divmod(i, a.cols)
        y0, y1 = int(round(r * ch)), int(round((r + 1) * ch))
        x0, x1 = int(round(c * cw)), int(round((c + 1) * cw))
        # step inside the cell so the neighbouring cell's edge never leaks in
        y0, y1, x0, x1 = y0 + 3, y1 - 3, x0 + 3, x1 - 3
        rgba = recover(W[y0:y1, x0:x1], B[y0:y1, x0:x1])

        bb = bbox(rgba[..., 3])
        if bb is None:
            print(f"  !! {fname}: empty cell")
            continue
        bx0, by0, bx1, by1 = bb
        bx0 = max(0, bx0 - a.pad); by0 = max(0, by0 - a.pad)
        bx1 = min(rgba.shape[1], bx1 + a.pad); by1 = min(rgba.shape[0], by1 + a.pad)
        crop = rgba[by0:by1, bx0:bx1]

        # how well do the three per-channel alpha estimates agree? a large
        # spread means the capture was lossy and the matte is approximate.
        d = np.clip(W[y0:y1, x0:x1] - B[y0:y1, x0:x1], 0, 1)
        spread = float((d.max(axis=2) - d.min(axis=2)).mean())
        worst = max(worst, spread)

        # Where the SOLID part of the sprite sits inside the crop. The renderer
        # anchors on this, not on the full crop, because the full crop includes
        # the drop shadow and the Inferno's glow -- align on those and every
        # building sits a few pixels too high.
        ob = bbox(crop[..., 3], 0.5) or (0, 0, crop.shape[1], crop.shape[0])

        cls, lvl = lookup.get(fname, ("Non-Defense", 0))
        out_name = S.slug(fname)
        Image.fromarray((crop * 255).astype(np.uint8), "RGBA").save(
            os.path.join(a.out, out_name))
        index[out_name] = dict(cls=cls, level=int(lvl), src=fname,
                               w=int(crop.shape[1]), h=int(crop.shape[0]),
                               box=[int(v) for v in ob],
                               opaque=float((crop[..., 3] > .5).mean()))
        print(f"  {out_name:<34s} {crop.shape[1]:3d}x{crop.shape[0]:3d}  "
              f"{cls} L{lvl}  alpha-spread {spread:.4f}")

    json.dump(index, open(index_path, "w"), indent=1, sort_keys=True)
    print(f"\n{len(index)} sprites in {a.out}   worst alpha spread {worst:.4f} "
          f"({'clean' if worst < 0.02 else 'LOSSY -- check the capture'})")


if __name__ == "__main__":
    main()
