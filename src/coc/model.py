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

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


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
