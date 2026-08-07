# Clash of Clans — RC Charge + Mass Dragons RL Agent

A deep reinforcement learning agent that plays a **complete Town Hall 15 attack**:
a Royal Champion charge under a chain of Invisibility Spells to delete the core
anti-air, then fourteen Dragons into a base that can no longer shoot them down.

![Trained agent attacking](media/trained_attack.gif)

The agent makes every decision a human makes:

1. where to deploy the Royal Champion
2. when and where to cast each of eleven Invisibility Spells — her life support
3. when to fire Seeking Shield
4. **when to stop charging and commit the air army**
5. where to drop the two funnel Baby Dragons
6. **where to bring the fourteen Dragons in from — the side the Air Sweepers are not facing**
7. where the Grand Warden and Archer Queen go
8. when to fire the Eternal Tome

Everything else — hero pathing, dragon targeting, defensive fire — is the game's
own AI.

---

## Results

10-hour training run: 5,940 episodes, 765,633 environment steps, ~800,000
gradient updates, on a laptop CPU. Evaluated over 120 attacks per policy.

| policy | stars | destruction | Town Hall | **3-star rate** |
|---|---|---|---|---|
| random | 1.12 | 62.8% | 16.7% | 0.0% |
| dragons only, no charge | 1.52 | 63.0% | 58.3% | 4.2% |
| scripted full attack | 2.39 | 91.4% | 92.5% | 46.7% |
| **DQN (trained)** | **2.72** | **95.1%** | **98.3%** | **73.3%** |

![Learning curve](media/training_curve.png)

The dip around episode 3,000 is where the last of the training wheels came off
— the supervised anchor and the scripted exploration teacher had both annealed
to zero, so the policy was briefly on its own before recovering and passing the
baseline for good around episode 4,200.

### That table is now out of date, and the reason matters

Building the recogniser meant looking hard at the generated bases, and two bugs
turned up in the generator that had been making every base easier than a real
one:

**The roster was short.** The compartment placer filled the walls first and put
the leftovers in by 120 random guesses. On a TH15 layout that is nowhere near
enough — by that point the interior has a handful of free 3×3 spots — so bases
shipped averaging 6.6 Archer Towers instead of 8 and 5.7 Cannons instead of 7.
**4.7 missing defenses per base.**

**Half the bases had no Clan Castle.** Same failure, worse consequence: the
Castle placement ran last, failed about half the time, and `cc_pos` came back
`None` — which meant no defending Clan Castle troops at all on those attacks.

Both are fixed by scanning instead of guessing, and by placing the core
buildings before the filler defenses the way a real builder does. Every base now
carries the full roster and a Clan Castle. Re-measured on the corrected
environment, 40 attacks per policy:

| policy | stars | destruction | Town Hall | 3-star |
|---|---|---|---|---|
| random | 1.50 | 70.9% | 47.5% | 7.5% |
| dragons only, no charge | 1.73 | 74.8% | 65.0% | 7.5% |
| **scripted full attack** | **2.25** | **89.9%** | 87.5% | 37.5% |
| DQN (trained on the old bases) | 1.82 | 73.8% | 65.0% | 37.5% |

The agent no longer beats the scripted baseline. It was not overfitting to noise
— it was fitting a village that was missing five defenses and, half the time,
the defending troops, and that skill does not transfer to a complete one.

The task is still clearly winnable at the new difficulty: the scripted policy
three-stars 37.5% of complete bases, so there is a real target above it.
**Retraining is one command and picks up all the fixes** — see Running it below.
Nothing about the algorithm needs to change; the environment it trains against
is simply now the right one.

### The army

Town Hall 15, four hero slots (no Barbarian King, no Minion Prince):

| | |
|---|---|
| Heroes | Royal Champion (Electro Boots + Spirit Fox), Archer Queen (**Giant Arrow**), Grand Warden (air mode, Eternal Tome), **Dragon Duke** (Fire Heart) |
| Troops | 14 Dragons + 2 Baby Dragons (302 of 320) |
| Spells | 11 Invisibility |
| Siege | **Stone Slammer** — carries the Castle troops in |
| Clan Castle | 2 Dragons + 1 Baby Dragon (50/50) and **3 more Invisibility** |
| **Total Invisibility** | **14** |

---

## What the attack actually is

A Royal Champion charge is a real Clash of Clans strategy. The Champion is sent
in alone (or ahead of the army) under a chain of Invisibility Spells, so she can
delete high-value defenses — or the Town Hall itself — while defenses cannot
target her. Guides describe the timing as roughly "cast, count to four, cast
again" as each 4.25-second spell expires.

Three mechanics make it work, and all three are modelled here:

**She targets the nearest defense.** Not the Town Hall first, not Air Defenses
first — the *nearest defense*, bypassing every non-defensive building on the way.
Since Town Hall 12, the Town Hall is itself a defense because of its Giga
Inferno, so it sits in that pool. Deploy her on the Town Hall's side of the base
and it is near the front of her queue.

**One spell does two jobs.** Cast on her, it makes her untargetable and she takes
zero damage. Cast on a *building*, it makes that building untargetable *by her*,
so she skips it and keeps walking. Protection and pathing control from the same
action. A defense that is itself invisible still fires, though, so hiding a
Scattershot to walk past it does not make it harmless.

**Invisibility resets Inferno Towers.** A single-target Inferno ramps 100 → 230 →
2,300 DPS the longer it holds a target, and loses the ramp when the target
disappears. Going invisible at the right moment is the difference between taking
100 damage a second and 2,300.

---

## The air phase

This is the part the Royal Champion charge exists to serve, and it has its own
mechanics that decide whether an attack triples.

**Dragons have no preferred target.** They fly at the *nearest building*,
collectors included. That single rule is the entire reason funnelling exists —
one uncleared hut three tiles off-axis peels the whole stack sideways and it
rings the base instead of penetrating it.

**The Baby Dragon tantrum.** A Baby Dragon with no allied air unit within 4.5
tiles gets +100% damage and +50% attack speed: 310 DPS instead of 155, nearly a
full Dragon for half the housing. So the two funnel Baby Dragons must be dropped
far from the stack *and far from each other*. Drop them together and you have
thrown away both the funnel and the damage.

**The Air Sweepers decide which side you attack from.** Two of them, each with a
120-degree cone locked to one of eight facings before the battle starts. They do
zero damage — they push the stack back four tiles and mute it for 1.2 seconds,
every five seconds. Against a three-minute clock, with no Rage and no Haste in
this army to punch through, coming in through a cone is how a triple becomes a
two-star. A competent base points them at offset quadrants so no side is fully
safe; the realistic goal is to eat one sweeper instead of two. **The sweeper
coverage map is channel 12 of the observation**, so the agent can see the cones
and learn to route around them.

**Air Defenses are the point of the whole attack.** 540 DPS each. They cannot
scratch the Champion — she is a ground unit — and they will delete fourteen
Dragons in seconds if they are still standing. "Harmless to her, lethal to the
army she is clearing for" is why the reward for killing something is deliberately
not the same as the threat it poses.

The army is the canonical list: **14 Dragons, 2 Baby Dragons, 11 Invisibility
Spells, no Rage, no Freeze**. Every spell is the Champion's life support.

### Target priority

Killing value is set by a swappable profile, because what is worth killing
depends entirely on what is coming in behind her:

| | air_support (default) | solo_snipe | ground_support |
|---|---|---|---|
| Town Hall | 15 | 20 | 12 |
| Air Defense | **6** | 0.2 | 0.1 |
| Monolith | 5 | 2 | 8 |
| Scattershot | 4 | 1.5 | 5 |
| Air Sweeper | **3** | 0.1 | 0.1 |
| Inferno Tower | 2 | 1.5 | 6 |
| Cannon | 0.2 | – | – |

Set `C.TARGET_PROFILE`. The profile is also painted into the observation as a
priority channel, so the agent sees which buildings are prizes rather than
having to infer it.


---

## The Giant Arrow Air Defense snipe

The single most interesting decision in the attack, and the one the agent has to
learn rather than be told.

**You do not aim the arrow. You aim the Queen.** From the wiki: *"the direction
the Giant Arrow travels depends on where the Archer Queen's current target is,
relative to her location at the moment the ability is used."* She will shoot at
the **nearest building** to wherever she lands, and the arrow flies along that
line, piercing everything with no falloff.

So the problem is geometric: find a deploy tile where the nearest building to it
*and* two or more Air Defenses are **collinear**. The outer building is your
gunsight. You are not aiming at the Air Defenses — you are aiming at whatever
happens to be standing in front of them.

The damage makes it worth hunting for. Reworked on 26 May 2026 (the Fandom wiki
is stale on this; Supercell's blog is current): **1,500 base, doubled against Air
Defenses = 3,000**, against 1,750 hitpoints. One arrow one-shots a max TH15 Air
Defense with 171% headroom, and because it pierces without falloff, **every Air
Defense on the line dies at once**.

The environment solves that geometry exactly and hands the answer to the agent
as **observation channel 14**: for every legal deploy cell, how many Air Defenses
an arrow fired from there would pierce. A two-Air-Defense line usually exists on
only one or two tiles of the whole map, which is what makes finding it valuable.
The Queen is deployed first after the charge and fires immediately, before she
walks a step and ruins the alignment.

## Dragon Duke and Stone Slammer

**Dragon Duke** (Hero Hall 9, level 10 at TH15) is a flying melee hero whose
Royal Rampage passive fires only while **no friendly air unit is within 6 tiles**
— +100% damage and +50% attack speed, roughly triple effective DPS. So he is
sent solo down the opposite flank. Park him inside the dragon stack and the
passive simply never turns on. Fire Heart is mandatory because Healers cannot
target a flying melee unit: +5,600 HP, 150 HP/sec regeneration, and a 3,000
damage explosion when he dies. (His trap-damage reduction has been nerfed twice
and is now **20%**, not the 50% most guides still quote.)

**Stone Slammer** targets **defenses only** until none remain, flies, splashes,
and carries the Clan Castle troops in — they pop out when it dies.


---

## Repository layout

```
src/coc/
  config.py     every game constant, cited to the wiki, plus the curriculum
  base.py       procedural TH15 base generation
  env.py        the attack simulation
  model.py      dueling fully-convolutional Q-network (~53k parameters)
  replay.py     n-step replay buffer, uint8-compressed, checkpointable
  policies.py   scripted baselines / demonstrations / exploration teacher
  train.py      Double-DQN training loop, curriculum, real resume
  evaluate.py   evaluation against baselines
  viz.py        base rendering, battle replay GIF, learning curves
  army.py       dragons, baby dragons, warden, queen, Air Sweeper cones
  full_env.py   the complete attack: charge -> funnel -> dragons
  layout.py     wall-compartment skeletons: how a real war base is built
  sprites.py    the sprite manifest and library -- what every building
                looks like at every level that matters at TH15
  vision.py     screenshot -> isometric calibration -> tile classifier
                -> 44x44 building grid -> the trained agent -> a plan
vision_data/
  sprites/      71 captured building sprites with recovered alpha
  index.json    class, level and solid-box for each one
notebooks/
  RC_Walk_Agent.ipynb        thin driver over the package
  rc_walk_single_cell.py     the whole thing flattened into one pasteable cell
scripts/
  build_single_cell.py       regenerates the flattened cell from the package
  build_notebook.py          regenerates the notebook
  crop_sprites.py            two grid screenshots -> sprite library with alpha
```

---

## How it works

### The environment

A 44×44 tile grid holding a generated TH15 base: Town Hall in the middle,
high-value defenses in the core, ordinary defenses in a ring, collectors pushed
outside, walls packing the centre so there is no gap to deploy into. Each tick
is 0.5 seconds of game time and a battle is 3 minutes (360 ticks).

Real Town Hall 15 stats throughout, at max level, sourced from the Clash of Clans
wiki. The Royal Champion is level 40 with Electro Boots: 3,910 base HP plus 2,400
from the boots, 530 DPS, 3-tile range, 3 tiles/second, a 5-tile 177 DPS damage
aura and 36 HP/second self-heal. Seeking Shield hits four targets for 1,860 each
and heals 2,600.

Two consequences of the real rules matter a lot:

*Air Defenses cannot touch her.* She is a ground unit. Air Defenses and Air
Sweepers deal her exactly zero damage. They are still defenses she will walk to
and destroy, and they count toward destruction percentage, but there is no
survival reason to prioritise them.

*The Eagle Artillery never wakes up.* It stays dormant until 200 housing space
has been deployed. A hero is 25 and a spell is 5, so a solo Champion with eight
spells is 65. It sits there as free destruction percentage.

### Observation — ten map layers plus eight scalars

| # | layer | |
|---|---|---|
| 0 | Town Hall footprint | |
| 1 | Air Defense footprint | harmless to her, but still a target |
| 2 | High-value defense footprint | X-Bow, Inferno, Scattershot, Monolith, Eagle |
| 3 | Ordinary defense footprint | |
| 4 | Non-defense building footprint | |
| 5 | Wall footprint | |
| 6 | Remaining HP fraction per tile | |
| 7 | Ground threat map | incoming DPS reaching each tile |
| 8 | Invisibility seconds remaining | |
| 9 | **Royal Champion position** | |

Scalars: her HP fraction, spells remaining, whether Seeking Shield is up, time
elapsed, whether the Town Hall is down, destruction percentage, fraction of
defenses still alive, and whether she is still waiting to be deployed.

### Action space — 486 actions

`0` wait, `1` Seeking Shield, `2…485` cast Invisibility on one of 22×22 tiles.
Spell placement is on a 2-tile stride because the spell has a 4-tile radius, so
finer resolution buys nothing and quadruples the output layer.

The first action of every episode is the **deployment**, chosen from the tiles
where the game would actually let you deploy (unoccupied, with the one-tile
buffer Clash enforces around every structure).

Illegal actions are masked rather than penalised. If she has no spells left, the
cast actions simply cannot be selected.

### Scoring

Rewards are on a unit scale — a good episode returns roughly +20 — with the Town
Hall as the prize.

| | |
|---|---|
| Town Hall destroyed | **+15** |
| 50% destruction (a star) | +5 |
| 100% destruction (the third star) | +10 |
| High-value defense killed | +1.0 |
| Ordinary defense killed | +0.4 |
| Other building killed | +0.05 |
| Damage dealt | +1 per 4,000 (×3 against the Town Hall) |
| Damage taken | −1 per 4,000 |
| Spell cast / ability used | −0.05 |
| Death | −2 |
| Timeout | −2 |
| Per tick | −0.01 |

Death is deliberately cheap. In real Clash a Champion who destroys the Town Hall
and then dies has *succeeded* — that is a star. Making death catastrophic teaches
cowardice.

### The network

Dueling and fully convolutional: three convolutions with pooling produce a
feature map at the action-grid resolution, and a 1×1 convolution turns it into
one Q-value per tile. Because tile actions are spatial, a single shared filter
evaluates all 484 of them, so "cast where she is about to be shot" generalises
across the map instead of being relearned per output unit. Two more Q-values for
wait and Seeking Shield come off a global branch that also sees the scalars.

**53,524 parameters. Checkpoints are 0.2 MB.**

### Training

Double DQN with a dueling head, 3-step returns, Huber loss, gradient-norm
clipping, a 40k–120k uint8-compressed replay buffer, and a target network synced
on a step schedule.

Destroying the Town Hall is a sparse event that uniformly random play almost
never achieves, so training starts by seeding the replay buffer with a few
hundred episodes from the scripted policy (about 85% of which succeed), and
early exploration sometimes takes the scripted action instead of a random one.
Both decay to zero — the final policy is the network's own.

Difficulty follows a curriculum that only advances when a greedy evaluation
clears the bar. It ramps the base up first and then takes her spells away, which
is the strongest difficulty knob because spells are literally how many seconds
she can be untouchable:

| stage | defenses | spells |
|---|---|---|
| s0 | 30% | 10 |
| s1 | 50% | 10 |
| s2 | 75% | 8 |
| s3 | 100% | 8 |
| s4 | 100% | 6 |
| s5 | 100% | 4 |
| s6 | 100% | 3 |

The terminal stage is a realistic army — guides recommend carrying two to four
Invisibility Spells at TH13–15.

---

## Running it

```bash
pip install torch numpy matplotlib jupyter
```

Train (CPU or CUDA is detected automatically):

```bash
python -m coc.train --full --out runs/full --hours 10 --resume --save-replay 20000
```

`--resume` picks up the newest checkpoint with the optimizer state, epsilon,
episode count, curriculum stage and RNG intact, so an interrupted run continues
rather than restarting.

Evaluate against the baselines:

```bash
python -m coc.evaluate --model runs/rc/latest.pt --episodes 200 --all-stages
```

Or work in the notebook, `notebooks/RC_Walk_Agent.ipynb`. If you prefer the
single-cell style, paste `notebooks/rc_walk_single_cell.py` into one cell —
it is generated from the package by `scripts/build_single_cell.py`, so the two
never drift.

---

## Reading a real base off the screen

`vision.py` turns a screenshot into the same 44×44 grid the simulator uses, so
the trained agent can plan on an actual war base instead of a generated one.

It does **not** touch your input. No taps, no clicks, no automation — Supercell's
terms prohibit third-party software that automates gameplay and an account that
does it gets banned. This reads a picture and tells you what to do; you play the
attack.

### What the classifier was trained on

Every building in Clash has a distinct look at every level, and that look is the
only thing a classifier can use. So the training data is built from the real
art: 71 sprites captured from the wiki — every defense at and just below its
TH15 cap, plus five Air Defense levels, four Inferno levels in both single and
multi mode, three Scattershot levels, both Monoliths, and fifteen resource
buildings — composited onto procedurally generated bases in correct isometric
projection.

The transparency is exact rather than keyed. A screenshot is flattened pixels
with no alpha channel, so each sprite grid was captured twice, once over white
and once over black, and the alpha recovered algebraically:

```
over white   Cw = a·C + (1−a)
over black   Cb = a·C
             a  = 1 − (Cw − Cb)      C = Cb / a
```

Measured disagreement between the three per-channel estimates: 0.0003. That
matters because it recovers the drop shadows and the glow around an Inferno's
beam as genuinely semi-transparent, where a background key would have turned
them into a hard halo the classifier then learns to look for.

The level labels are whatever sprite was actually drawn, so a label can never
disagree with the pixels it describes.

### Levels, and why they are readable

Supercell recolours rather than re-textures at the top end, which is lucky:
Air Defense 11 is vivid teal, 12 emerald green, 13 purple. That is a colour
histogram difference, not a subtle one. The four classes carrying a level head
are the four where the level changes the plan — Inferno Tower (and whether it
is single or multi target), Air Defense, Scattershot, Monolith.

### Reassembling buildings from tiles

The classifier labels tiles; a base is made of buildings. Every building has a
known footprint and a TH15 base has a known roster, so instead of growing
rectangles out of whatever the classifier said, the known footprint is slid over
the class votes and the best non-overlapping placements are taken, at most as
many as the roster allows. A 3×3 Cannon survives three wrong tiles out of nine.
On ground-truth labels this recovers 97.3% of buildings exactly, with every
defense at the correct roster count.

```bash
python -m coc.vision dataset --bases 45          # render training data
python -m coc.vision train --epochs 8            # train the tile classifier
python -m coc.vision check                       # score on held-out renders
python -m coc.vision preview --seed 0 --grid     # look at what it renders

# one-time, per zoom level: click the four tips of the village diamond
python -m coc.vision calibrate --top 960 120 --right 1750 520 \
    --bottom 960 920 --left 170 520 --image shot.png

python -m coc.vision read shot.png --agent runs/full/ckpt_best.pt
```

### Clash does not render 2:1, and that was the whole ballgame

Textbook isometric is 2:1 — a tile twice as wide as it is tall. This module
assumed that, and it is wrong. Measured three independent ways on a real attack
screenshot (iPhone, 2781×1280):

1. Hough over the deployment boundary and the wall runs — every grid-aligned
   line has slope exactly ±0.750, and that slope *is* the tile ratio.
2. Fitting the 44×44 boundary diamond as a whole — hw 19.20, hh 14.40.
3. Autocorrelating wall-segment spacing along a run — hw 19.0.

All three give **4:3**, and the last two agree on absolute size to within 1%.
Training at 2:1 meant every rendered base packed its buildings closer together
vertically than the game does, so the crops the classifier learned from had the
wrong neighbours in them. Worse, both calibration entry points had a sanity
check rejecting any ratio outside 1.6–2.6 — they would have refused to
calibrate a real screenshot and blamed the user's corner clicks.

Re-rendered and retrained at the correct 4:3:

| | 2:1 (wrong) | **4:3 (correct)** |
|---|---|---|
| all tiles | 89.6% | **95.9%** |
| building tiles only | 93.4% | **97.4%** |
| level, of correctly-typed tiles | 99.6% | **99.4%** |
| buildings placed on the exact tile | 77.9% | **90.3%** |
| buildings within one tile | 87.9% | **96.2%** |
| roster error per base | 2.0 | **0.0** |

### Measured

Trained on 50 rendered bases (69,289 tile crops), 12 epochs, CPU, 452k
parameters. Per-class recall at 100%: Air Defense, Archer Tower, Scattershot,
Inferno Tower, Bomb Tower. Weakest: Spell Tower 89.4% and Air Sweeper 90.3% —
both 2×2, the smallest things on the board. Every Air Defense level, every
Scattershot level and both Monoliths classify at 100%.

End to end on four bases the model has never seen — render, calibrate, crop,
classify, reassemble — the recovered roster is **exactly right**, with no
missing and no spurious buildings:

```
  Town Hall          1        Archer Tower       8
  Air Defense        4  levels 9,11,12,13        Cannon   7
  Eagle Artillery    1        Wizard Tower       5
  Monolith           1  level 1                  Mortar   4
  Scattershot        2        Hidden Tesla       5
  Inferno Tower      3  levels 6M,6S,7M          Bomb Tower 2
  X-Bow              4        Spell Tower        2
  Clan Castle        1        Air Sweeper        2

  mean footprint agreement 98%
  Infernos: 2 multi-target, 1 single-target
```

The second biggest thing that made this work was not the model. It was noticing
that the tile crop was the wrong shape. A tile is `2·hw` wide and only `hw`
tall, but a 3×3 building drawn on it is `6·hw` tall, because isometric art rises
off the ground. Sizing the crop from the tile's height gave a box reaching 36 px
above the tile centre on a building whose distinctive top was 82 px up — so the
classifier was being shown the one part of a Cannon, a Mortar and a Bomb Tower
that looks identical. Sizing it from the tile's *width* instead:

| crop (width, height ratio, top fraction) | all tiles | buildings |
|---|---|---|
| (1.6, 0.90, 0.70) — the old box | 73.7% | 78.6% |
| **(2.4, 1.25, 0.72)** | **79.4%** | **88.0%** |
| (3.0, 1.30, 0.75) | 74.3% | 88.0% |

Nine points, for showing it the building instead of the platform.

### Where this is honest about failing

**Calibration is manual, and that is deliberate.** Automatic detection of the
village diamond was tried three ways on a real screenshot and all three failed:
grass detection (most grass is under buildings), background subtraction (the
surround is ornate scenery, not flat colour), and fitting the yellow boundary
line (gold statues outside the village are the same colour, and buildings break
the line into fragments — the four fitted edges gave slopes of 0.2/0.45/1.14/0.08
where an isometric diamond must give −2/+2/+2/−2). Four clicks is a one-time
cost: your zoom and resolution do not change between attacks.

**Scenery skins repaint every building**, and a classifier trained on the
default skin will not read a base wearing one. That needs your screenshots.

**Rubble is not modelled.** The renders show intact bases, which is right for
the scouting screenshot you plan from, and wrong for anything mid-attack.

---

## Current status — measured, not claimed

Every number below came from `python -m coc.evaluate`, 60 episodes per policy.

**What the trained model scores today**, on a full TH15 base with a 10-spell
charge army:

| | DQN (short run) | scripted baseline |
|---|---|---|
| average stars | **0.18** | 0.45 |
| average destruction | **16.4%** | 20.0% |
| defenses killed | 12.1 of 51 | 15.3 of 51 |
| high-value defenses killed | 3.0 | 3.7 |
| star split | 0★ 82%, 1★ 18% | 0★ 55%, 1★ 45% |

**Neither ever gets 2 stars, and that is structural, not a shortfall.** Two
stars needs 50% destruction, and a Royal Champion only attacks defenses while
any defense stands — she walks straight past all 45 collectors and storages, so
the percentage bar physically cannot climb past roughly the defense count. This
matches the real game: a solo Champion charge is a *phase* of an attack, not a
whole attack. Judge it by which defenses died, not by the bar.

**Is the task winnable?** Yes, and this is the whole point. A hand-written
scripted policy on a *full* TH15 base with a 10-Invisibility charge army:

| policy | Town Hall destroyed | stars | destruction | mean return |
|---|---|---|---|---|
| random | 0.0% | 0.00 | 8.8% | 13.6 |
| never-cast | 0.0% | 0.00 | 6.2% | 8.1 |
| **scripted-human** | **48.3%** | **0.48** | **18.9%** | **68.1** |

The previous version of this project scored 0% here for *any* policy, including
a perfect one.

**What the hidden layer costs**, scripted policy, full TH15, 10 spells:

| | Town Hall destroyed | destruction | traps hit | defenders killed |
|---|---|---|---|---|
| base only | 46% | 19.8% | – | – |
| + traps | 46% | 19.9% | 2.1 | 2.6 |
| + Clan Castle | **26%** | 18.2% | 1.7 | 3.9 |
| + defending Queen | 26% | 18.2% | 1.5 | 3.1 |

The Clan Castle roughly halves her success rate, which is exactly what players
report — it is the top failure mode of a real RC charge, and the cost is tempo
rather than damage.

**Does the agent learn?** Yes. A short validation run (about 700 episodes on two
CPU cores) took the greedy policy from 0% to 90% Town Hall kills on stage 0 and
auto-promoted through two curriculum stages:

```
after behaviour cloning:  greedy TH 85%
EVAL @ 200:  TH 90.0% | stars 1.00 | dest 23.7%   -> promoted to s1
EVAL @ 400:  TH 53.3% | stars 0.53 | dest 17.1%
EVAL @ 600:  TH 66.7% | stars 0.67 | dest 16.5%   -> promoted to s2
```

**Has it beaten the heuristic yet?** Not yet, and this is worth being plain
about. At the end of that short run the DQN sits at 58% on stage 1 against the
scripted policy's 92%. It comfortably beats random and never-cast, it climbs,
and the curriculum advances — but 700 episodes on two cores is a smoke test, not
a training run. It needs an overnight run on real hardware to be judged.

The reason that sentence can be written at all is the baseline table. The point
of `evaluate.py` is that you never again have to wonder.

## Always check the task is winnable first

The single most useful habit this project taught: before starting a long run,
confirm a scripted policy can actually win. It takes thirty seconds.

```python
from coc.evaluate import run_policy
from coc.policies import policy_scripted_human
r = run_policy(policy_scripted_human, episodes=60, defense_frac=1.0, spells=8)
assert r["th_kill_rate"] > 0.1
```

An earlier version of this project trained for over thirty hours across roughly
8,500 episodes and destroyed the Town Hall zero times. The agent was not at
fault: the Town Hall sat last in an invented target priority behind all 24
defenses, so winning required destroying 59,000 HP with about 62 seconds of
survivable time against a need for 121. No policy could have won, so the two
largest rewards in the system never once entered the replay buffer. A thirty
second baseline check would have caught it on day one.

---

## Built with

PyTorch, NumPy and Matplotlib.

Game data from the [Clash of Clans Fandom wiki](https://clashofclans.fandom.com/)
and [clasher.us](https://www.clasher.us/), at Town Hall 15 max levels.
