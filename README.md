# 🏰 Clash of Clans — Reinforcement Learning Attack Agent

An autonomous **Reinforcement Learning** agent that learns to attack Clash of Clans bases using a **Deep Q-Network (DQN)**. The agent controls the **Royal Champion** hero, learning optimal spell placement and target prioritization to destroy the Town Hall and Air Defenses on a procedurally generated TH15 base.

![Battle Replay Demo](media/battle_replay.gif)

---

## 📋 Overview

This project simulates a simplified Clash of Clans attack scenario as a grid-based environment and trains a DQN agent to play it. The agent must learn:

- **When and where** to cast Invisibility Spells to protect the Royal Champion
- **Target prioritization** — Town Hall first, then Air Defenses, then other buildings
- **Spatial reasoning** on a 44×44 tile grid with 11 building categories

The entire pipeline — environment, neural network, training loop, and visualization — runs in a single Jupyter notebook.

---

## 🧠 How It Works

### Environment (`CoCEnv`)

- **Grid**: 44×44 tiles representing a procedurally generated TH15 base
- **Observation**: 3-channel input (Building IDs, Value Grid, Invisibility Timer)
- **Actions**: 1,937 discrete actions — 1 "wait" action + 1,936 grid locations for spell casting
- **Physics**: Simulates Royal Champion movement, attacking, HP, incoming damage, and invisibility mechanics
- **Reward shaping**: +1000 for Town Hall, +300 for Air Defenses, penalties for damage taken and bad targeting

### Neural Network Architecture

A **4-layer Deep Q-Network (DQN)** with convolutional feature extraction:

```
Input (3 × 44 × 44)
    ↓
Conv2d(3 → 32, 3×3) + ReLU
    ↓
Conv2d(32 → 64, 3×3) + ReLU
    ↓
Conv2d(64 → 64, 3×3) + ReLU
    ↓
Flatten → FC(123,904 → 512) + ReLU
    ↓
FC(512 → 1,937)  [Q-values for each action]
```

### Training

- **Algorithm**: Deep Q-Learning with Experience Replay and Target Network
- **Replay Buffer**: Stores transitions `(s, a, r, s', done)` for off-policy learning
- **Epsilon-Greedy**: Decaying exploration from ε=1.0 → ε=0.01
- **Multiple training runs**: Progressively harder configurations (`hard_v2`, `pro`, `master`, `th15`)
- **Checkpoints saved** every 100–500 episodes

---

## 📁 Project Structure

```
clashOfClansAgent/
├── README.md                           # This file
├── .gitignore                          # Excludes checkpoints & temp files
├── notebooks/
│   ├── Clash_of_Clans_Agent.ipynb      # Main notebook (env, model, training, visualization)
│   └── test.ipynb                      # Testing / experimentation notebook
├── media/
│   └── battle_replay.gif              # Animated demo of a trained agent attacking
└── checkpoints/                        # (gitignored) Model weights (~26 GB total)
    ├── rc_hard_v2_*.pth                # Hard difficulty v2 checkpoints
    ├── rc_pro_*.pth                    # Pro difficulty checkpoints
    ├── rc_master_*.pth                 # Master difficulty checkpoints
    └── rc_th15_*.pth                   # Full TH15 checkpoints (final model)
```

> **Note:** The `checkpoints/` folder contains trained model weights (`.pth` files) totaling ~26 GB. These are excluded from the repository due to GitHub's file size limits. To use pre-trained models, you would need to train them locally.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA recommended for training)
- NumPy
- Matplotlib
- Jupyter Notebook

### Installation

```bash
git clone https://github.com/steven1423/clashOfClansAgent.git
cd clashOfClansAgent
pip install torch numpy matplotlib jupyter
```

### Running

1. Open the main notebook:
   ```bash
   jupyter notebook notebooks/Clash_of_Clans_Agent.ipynb
   ```
2. Run **Cell 1** to generate and visualize a random TH15 base layout
3. Run **Cell 2+** to initialize the DQN and start training
4. Watch rewards improve over episodes as the agent learns to prioritize targets

---

## 🎮 Key Features

| Feature | Description |
|---|---|
| **Procedural Base Generation** | Random TH15 layouts with proper building placement rules |
| **Multi-Channel Observation** | Building IDs + strategic values + invisibility state |
| **Reward Shaping** | Hierarchical rewards guide the agent to prioritize high-value targets |
| **Progressive Difficulty** | Train from easy → hard → pro → master → full TH15 |
| **Battle Replay** | Animated GIF visualization of the trained agent's attack |

---

## 📊 Training Results

The agent was trained across multiple difficulty tiers with thousands of episodes per tier. Training progresses from simpler environments to the full TH15 simulation with all defenses active.

| Tier | Episodes | Description |
|---|---|---|
| `hard_v2` | 2,000 | Reduced defenses, core mechanics |
| `pro` | 1,500 | More defenses, tighter rewards |
| `master` | 1,500 | Full defense roster |
| `th15` | 5,700+ | Complete TH15 base, 10+ hour training |

---

## 🛠️ Technologies Used

- **PyTorch** — Neural network and training
- **NumPy** — Grid environment and state representation
- **Matplotlib** — Base visualization and training plots
- **Jupyter Notebook** — Interactive development environment

---

## 📄 License

This project is for educational and research purposes.
