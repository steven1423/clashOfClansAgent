#!/usr/bin/env python3
"""
Flatten the `coc` package into ONE self-contained file that can be pasted
straight into a Jupyter cell -- same working style as the original notebook,
but backed by the real package so the two can never drift apart.

    python scripts/build_single_cell.py  -> notebooks/rc_walk_single_cell.py

Regenerate this whenever you change anything under src/coc/.
"""

import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "coc")
OUT = os.path.join(ROOT, "notebooks", "rc_walk_single_cell.py")

ORDER = ["config.py", "defenders.py", "army.py", "layout.py", "base.py",
         "env.py", "full_env.py", "model.py", "replay.py", "policies.py",
         "train.py", "evaluate.py", "sprites.py", "vision.py", "viz.py"]

# `layout.py` and `base.py` both refer to the layout module as `L.`; flattened
# there is no module boundary, so bind the alias the same way the config
# namespace is bound below.
LAYOUT_SHIM = '''

import types as _types
L = _types.SimpleNamespace(
    ARCHETYPES=ARCHETYPES, ARCHETYPE_WEIGHTS=ARCHETYPE_WEIGHTS, Cell=Cell,
    build_skeleton=build_skeleton, open_compartments=open_compartments,
    seal=seal, air_defense_spots=air_defense_spots,
    collinear_pairs=collinear_pairs, sweeper_facings=sweeper_facings)

'''

# `vision.py` refers to the sprite manifest as `S.`
SPRITES_SHIM = '''

import types as _types
S = _types.SimpleNamespace(
    MANIFEST=MANIFEST, SPRITE_DIR_DEFAULT=SPRITE_DIR_DEFAULT, slug=slug,
    grids=grids, tag=tag, variant=variant, Sprite=Sprite,
    SpriteLibrary=SpriteLibrary)

'''

DROP_LINE = re.compile(
    r"^\s*(from\s+\.\s*import|from\s+\.\w+\s+import|from\s+__future__\s+import"
    r"|import\s+argparse\b)"
)

HEADER = '''# ======================================================================
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

'''

ALIAS_SHIM = '''

# `base.py` imports the Trap dataclass under an alias; flattened there is no
# module boundary, so bind the alias by hand.
TrapSite = Trap
'''

CONFIG_SHIM = '''

# ----------------------------------------------------------------------
# In the package the config lives in its own module and is referenced as `C.X`.
# Flattened into one cell there is no module, so bind a namespace with the same
# name over everything defined above. Keeps the code below byte-identical to
# the package version.
# ----------------------------------------------------------------------
import types as _types
C = _types.SimpleNamespace(**{_k: _v for _k, _v in list(globals().items())
                              if not _k.startswith("__")})

'''


def strip_module(path: str) -> str:
    src = open(path, encoding="utf-8").read()
    out = []
    skip_depth = 0
    skip_cont = False
    for line in src.splitlines():
        if skip_depth > 0 or skip_cont:
            skip_depth += line.count("(") - line.count(")")
            skip_cont = line.rstrip().endswith("\\")
            continue
        if DROP_LINE.match(line):
            # a dropped import may span several lines: "from .base import (\n ...)"
            skip_depth = line.count("(") - line.count(")")
            skip_cont = line.rstrip().endswith("\\")
            continue
        out.append(line)
    text = "\n".join(out)
    # Drop the argparse-driven CLI blocks. They make no sense inside a cell, and
    # `import argparse` was stripped above, so leaving one in means a NameError
    # the moment the cell runs -- which is exactly what the previous regex did:
    # it matched `def main(argv=None)` but its lookahead stopped early, so
    # train.py's CLI survived into the flattened file and every paste-and-run
    # died on `argparse is not defined`. Use the parser: find top-level `def
    # main` and the `if __name__` guard by their real extents.
    tree = ast.parse(text)
    kill = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            kill.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"):
                kill.append((node.lineno, node.end_lineno))
    if kill:
        lines = text.splitlines()
        drop = set()
        for a, b in kill:
            drop.update(range(a - 1, b))
        text = "\n".join(l for i, l in enumerate(lines) if i not in drop)
    return text.rstrip() + "\n"


def main() -> None:
    parts = [HEADER]
    for i, name in enumerate(ORDER):
        path = os.path.join(SRC, name)
        parts.append("\n\n# " + "=" * 68)
        parts.append(f"\n# ---- {name} " + "-" * (60 - len(name)) + "\n")
        parts.append(strip_module(path))
        if name == "config.py":
            parts.append(CONFIG_SHIM)
        if name == "defenders.py":
            parts.append(ALIAS_SHIM)
        if name == "layout.py":
            parts.append(LAYOUT_SHIM)
        if name == "sprites.py":
            parts.append(SPRITES_SHIM)
    text = "".join(parts)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    compile(text, OUT, "exec")          # fail loudly if the flatten broke
    print(f"wrote {OUT}  ({len(text.splitlines()):,} lines, "
          f"{len(text)/1024:.0f} KB) -- syntax OK")


if __name__ == "__main__":
    main()
