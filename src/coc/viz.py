"""
Base rendering and battle-replay animation.

Produces the same kind of artefact as the original project's
media/battle_replay.gif, but with the Royal Champion's position, her health,
her current target, the live threat map and the active Invisibility Spells all
drawn, so a replay is actually diagnosable instead of just decorative.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import animation

from . import config as C
from .base import (CAT_AIR_DEFENSE, CAT_DEFENSE, CAT_EMPTY, CAT_HIGH_DEFENSE,
                   CAT_NON_DEFENSE, CAT_TOWN_HALL, CAT_WALL)

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
