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

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Deque, Optional, Tuple

from . import config as C


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
