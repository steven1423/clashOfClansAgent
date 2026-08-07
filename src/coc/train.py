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

from __future__ import annotations

import argparse
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

from . import config as C
from .base import destruction_percent, stars
from .env import RCWalkEnv
from .full_env import FullAttackEnv, N_ACTIONS_FULL, N_CH, N_SC
from .policies import policy_full_attack
from .model import RCQNet, build_model, count_parameters
from .policies import policy_scripted_human
from .replay import NStepReplay


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
            from .full_env import (A_ARROW, A_DEPLOY, A_DUKE, A_SHIELD,
                                   A_SPELL, A_TOME, A_WAIT)
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
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the Royal Champion walk agent")
    p.add_argument("--out", default="runs/rc")
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--episodes", type=int, default=100_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--preset", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--full", action="store_true",
                   help="train the FULL attack (RC charge + mass dragons) "
                        "instead of the charge phase alone")
    p.add_argument("--stage", type=int, default=None,
                   help="curriculum stage to start at "
                        "(default: 0 for the charge, 4 for --full)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory", type=int, default=None)
    p.add_argument("--save-replay", type=int, default=0,
                   help="persist this many recent transitions in checkpoints "
                        "(0 = off; 20000 is a good value, ~0.8 GB)")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--demo-episodes", type=int, default=None,
                   help="scripted episodes used to seed the replay buffer")
    p.add_argument("--bc-steps", type=int, default=None,
                   help="behaviour-cloning warm-start steps")
    a = p.parse_args(argv)

    use_gpu = (a.preset == "gpu") or (a.preset == "auto" and torch.cuda.is_available())
    cfg = C.gpu_preset() if use_gpu else C.TrainConfig()
    if a.full:
        # Full-attack episodes are ~200 steps instead of ~70, and the
        # observation is 15 channels instead of 11, so the sensible defaults
        # differ. Keep the demonstration set inside the buffer capacity --
        # otherwise the demos evict themselves before training even starts.
        cfg.demo_episodes = 70
        cfg.bc_steps = 6_000  # ~10 min on CPU, and it is what makes the rest work
        cfg.memory_size = 40_000 if use_gpu else 15_000
        cfg.start_stage = 4          # full base immediately; the army makes it viable
    cfg.max_hours = a.hours
    cfg.max_episodes = a.episodes
    cfg.device = a.device
    cfg.seed = a.seed
    cfg.curriculum = not a.no_curriculum
    if a.stage is not None:
        cfg.start_stage = a.stage
    if a.memory:
        cfg.memory_size = a.memory
    if a.eval_every:
        cfg.eval_every = a.eval_every
    if a.demo_episodes is not None:
        cfg.demo_episodes = a.demo_episodes
    if a.bc_steps is not None:
        cfg.bc_steps = a.bc_steps

    t = Trainer(cfg, a.out, full=a.full)
    if a.resume:
        t.load()
    t.train(save_replay=a.save_replay)


if __name__ == "__main__":
    main()
