"""
Evaluation and baselines.

The previous project had no baseline to compare against, which is how 8,500
episodes of zero wins went unnoticed for 30 hours. Every number the trained
agent produces here is printed next to three reference policies:

  random          -- uniform over legal actions
  never-cast      -- deploy, then never spend a spell (measures the raw hero)
  scripted-human  -- deploy nearest the Town Hall, keep her invisible whenever
                     she is actually standing in a threatened tile, pop Seeking
                     Shield below 35% HP

If a trained model does not beat scripted-human, it has not learned anything
worth keeping. Run this before and after any long training session.
"""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from . import config as C
from .base import destruction_percent, stars
from .env import RCWalkEnv
from .model import RCQNet
from .train import pick_device


from .policies import (policy_random, policy_never_cast,
                       policy_scripted_human, deploy_nearest_th)


def make_model_policy(net: RCQNet, device: torch.device) -> Callable:
    def policy(env: RCWalkEnv, t: int) -> int:
        spatial, scalars = env._obs()
        s = torch.from_numpy(spatial).unsqueeze(0).to(device)
        sc = torch.from_numpy(scalars).unsqueeze(0).to(device)
        m = torch.from_numpy(env.legal_actions()).unsqueeze(0).to(device)
        with torch.no_grad():
            q = net.q_masked(s, sc, m)
        return int(q.argmax(dim=1).item())
    return policy


# ----------------------------------------------------------------------
def run_policy(policy: Callable, episodes: int = 100, defense_frac: float = 1.0,
               spells: int = C.MAX_SPELLS, seed: int = 0,
               record_best: bool = False) -> Dict:
    env = RCWalkEnv(defense_frac=defense_frac, max_spells=spells, seed=seed)
    rets, ths, sts, dests, lens, hps = [], [], [], [], [], []
    keys, defs_, killed = [], [], {}
    best_frames, best_score = None, -1e9
    for ep in range(episodes):
        env.record = record_best
        env.reset()
        total = 0.0
        env.step(policy(env, -1))
        for t in range(C.MAX_STEPS):
            _, r, done, _ = env.step(policy(env, t))
            total += r
            if done:
                break
        rets.append(total)
        ths.append(int(env.th_destroyed))
        sts.append(stars(env.buildings, env.th_destroyed))
        dests.append(destruction_percent(env.buildings))
        lens.append(env.steps)
        hps.append(env.rc_hp / C.RC_MAX_HP)
        keys.append(env.stats["key_kills"])
        defs_.append(env.stats["defense_kills"])
        for b in env.buildings:
            if b.is_dead and b.is_defense:
                killed[b.name] = killed.get(b.name, 0) + 1
        if record_best and total > best_score:
            best_score, best_frames = total, list(env.frames)
    out = dict(
        episodes=episodes,
        th_kill_rate=float(np.mean(ths)),
        stars=float(np.mean(sts)),
        three_star=float(np.mean([s == 3 for s in sts])),
        destruction=float(np.mean(dests)),
        ret=float(np.mean(rets)),
        ret_std=float(np.std(rets)),
        length=float(np.mean(lens)),
        hp_left=float(np.mean(hps)),
        key_kills=float(np.mean(keys)),
        defense_kills=float(np.mean(defs_)),
        killed={k: v / episodes for k, v in
                sorted(killed.items(), key=lambda kv: -kv[1])},
    )
    if record_best:
        out["frames"] = best_frames
    return out


def _fmt(name: str, r: Dict) -> str:
    return (f"{name:<20s} TH {100*r['th_kill_rate']:5.1f}%  "
            f"stars {r['stars']:4.2f}  dest {100*r['destruction']:5.1f}%  "
            f"return {r['ret']:7.2f}  "
            f"KEY defenses {r['key_kills']:4.1f}  all def {r['defense_kills']:4.1f}")


def compare(model_path: Optional[str] = None, episodes: int = 100,
            defense_frac: float = 1.0, spells: int = C.MAX_SPELLS,
            device: str = "auto") -> Dict[str, Dict]:
    print(f"\n=== EVALUATION  (defense_frac={defense_frac}, spells={spells}, "
          f"{episodes} episodes each) ===")
    results: Dict[str, Dict] = {}
    for name, pol in [("random", policy_random),
                      ("never-cast", policy_never_cast),
                      ("scripted-human", policy_scripted_human)]:
        results[name] = run_policy(pol, episodes, defense_frac, spells, seed=1234)
        print(_fmt(name, results[name]))

    if model_path:
        dev = pick_device(device)
        blob = torch.load(model_path, map_location=dev, weights_only=False)
        cfg = C.TrainConfig(**{k: v for k, v in blob["cfg"].items()
                               if k in C.TrainConfig.__dataclass_fields__})
        net = RCQNet(cfg).to(dev)
        net.load_state_dict(blob["policy"])
        net.eval()
        results["dqn"] = run_policy(make_model_policy(net, dev), episodes,
                                    defense_frac, spells, seed=1234)
        print(_fmt("dqn (trained)", results["dqn"]))
        h = results["scripted-human"]["ret"]
        d = results["dqn"]["ret"]
        verdict = "BEATS" if d > h else "does NOT beat"
        print(f"\n  -> the trained agent {verdict} the scripted baseline "
              f"({d:.2f} vs {h:.2f} mean return)")
        print(f"\n  key defenses destroyed per attack "
              f"(profile: {C.TARGET_PROFILE}):")
        for name in ("Town Hall", "Air Defense", "Air Sweeper", "Monolith",
                     "Scattershot", "Eagle Artillery", "Inferno Tower"):
            kd = results["dqn"]["killed"].get(name, 0.0)
            kh = results["scripted-human"]["killed"].get(name, 0.0)
            print(f"     {name:<16s} dqn {kd:4.2f}   scripted {kh:4.2f}")
    return results

# ======================================================================
# FULL ATTACK evaluation: RC charge -> funnel -> mass dragons
#
# This is the one that answers "how many stars do I get". Unlike the solo
# charge, three stars is genuinely reachable here, because Dragons target the
# NEAREST BUILDING and therefore clear collectors and storages too.
# ======================================================================
from .full_env import FullAttackEnv, A_DEPLOY, N_ACTIONS_FULL, N_CH, N_SC
from .policies import policy_full_attack


def policy_random_full(env, t):
    return int(np.random.choice(np.flatnonzero(env.legal_actions())))


def policy_dragons_only(env, t):
    """No charge at all: dump the whole army immediately. Measures how much the
    Royal Champion charge is actually worth."""
    if not env.rc_deployed:
        return A_DEPLOY + int(np.flatnonzero(env.legal_actions()[A_DEPLOY:])[0])
    if env.queue:
        legal = np.flatnonzero(env.legal_actions()[A_DEPLOY:])
        if len(legal):
            return A_DEPLOY + int(legal[len(legal) // 2])
    return 0


def run_full(policy, episodes=60, defense_frac=1.0, seed=0, record_best=False):
    env = FullAttackEnv(defense_frac=defense_frac, max_spells=C.SPELL_CAPACITY,
                        seed=seed)
    st, pc, th, rets, keys, ads, three = [], [], [], [], [], [], []
    best_frames, best = None, -1e9
    for ep in range(episodes):
        env.record = record_best
        env.reset()
        env._side_cache = None
        total = 0.0
        env.step(policy(env, -1))
        for t in range(C.MAX_STEPS):
            _, r, done, _ = env.step(policy(env, t))
            total += r
            if done:
                break
        s = stars(env.buildings, env.th_destroyed)
        st.append(s); three.append(s == 3)
        pc.append(destruction_percent(env.buildings))
        th.append(env.th_destroyed); rets.append(total)
        keys.append(env.stats["key_kills"]); ads.append(env.stats["ad_killed"])
        if record_best and total > best:
            best, best_frames = total, list(env.frames)
    out = dict(stars=float(np.mean(st)), destruction=float(np.mean(pc)),
               th=float(np.mean(th)), ret=float(np.mean(rets)),
               three_star=float(np.mean(three)),
               key_kills=float(np.mean(keys)), ad_killed=float(np.mean(ads)),
               star_split={k: float(np.mean([x == k for x in st])) for k in (0, 1, 2, 3)})
    if record_best:
        out["frames"] = best_frames
    return out


def compare_full(model_path=None, episodes=60, defense_frac=1.0, device="auto"):
    print(f"\n=== FULL ATTACK  (RC charge + 14 dragons, {episodes} attacks each) ===")
    res = {}
    for name, pol in [("random", policy_random_full),
                      ("dragons only (no charge)", policy_dragons_only),
                      ("scripted full attack", policy_full_attack)]:
        res[name] = run_full(pol, episodes, defense_frac, seed=1234)
        r = res[name]
        print(f"{name:<26s} {r['stars']:4.2f} stars  {100*r['destruction']:5.1f}% dest  "
              f"TH {100*r['th']:5.1f}%  3star {100*r['three_star']:4.1f}%  "
              f"key def {r['key_kills']:4.1f}  AD/sweeper {r['ad_killed']:4.1f}")
    if model_path:
        dev = pick_device(device)
        blob = torch.load(model_path, map_location=dev, weights_only=False)
        cfg = C.TrainConfig(**{k: v for k, v in blob["cfg"].items()
                               if k in C.TrainConfig.__dataclass_fields__})
        net = RCQNet(cfg, n_channels=N_CH, n_scalars=N_SC,
                     n_scalar_actions=5, n_tile_heads=2).to(dev)
        net.load_state_dict(blob["policy"]); net.eval()
        res["dqn"] = run_full(make_model_policy(net, dev), episodes,
                              defense_frac, seed=1234)
        r = res["dqn"]
        print(f"{'dqn (trained)':<26s} {r['stars']:4.2f} stars  "
              f"{100*r['destruction']:5.1f}% dest  TH {100*r['th']:5.1f}%  "
              f"3star {100*r['three_star']:4.1f}%  key def {r['key_kills']:4.1f}  "
              f"AD/sweeper {r['ad_killed']:4.1f}")
        print(f"\n  star split (dqn): " + "  ".join(
            f"{k}*={100*v:.0f}%" for k, v in r["star_split"].items()))
    return res


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Evaluate the Royal Champion agent")
    p.add_argument("--model", default=None, help="path to a .pt checkpoint")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--defense-frac", type=float, default=1.0)
    p.add_argument("--spells", type=int, default=C.MAX_SPELLS)
    p.add_argument("--device", default="auto")
    p.add_argument("--all-stages", action="store_true")
    p.add_argument("--full", action="store_true",
                   help="evaluate the FULL attack (RC charge + mass dragons) "
                        "rather than the charge phase on its own")
    a = p.parse_args(argv)

    if a.full:
        compare_full(a.model, a.episodes, a.defense_frac, a.device)
        return
    if a.all_stages:
        for st in C.CURRICULUM:
            compare(a.model, a.episodes, st.defense_frac, st.spells, a.device)
    else:
        compare(a.model, a.episodes, a.defense_frac, a.spells, a.device)


if __name__ == "__main__":
    main()
