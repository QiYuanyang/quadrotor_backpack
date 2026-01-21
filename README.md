# Alice Isaac - Humanoid Quadrotor IsaacLab

A humanoid robot with quadrotor backpack RL project using IsaacLab.

## Project Structure

```
alice_isaac/
├── assets/
│   └── humanoid_quadrotor.usd    # Robot USD file
├── envs/
│   ├── __init__.py
│   ├── quad_humanoid_env.py      # DirectRLEnv implementation
│   └── quad_humanoid_env_cfg.py  # Environment configuration
└── scripts/
    ├── test_env.py               # Test environment
    └── train_ppo.py              # Training script
```

## Setup

1. Install IsaacLab (already done in your setup)
2. Install RSL-RL for training:
   ```bash
   pip install rsl-rl
   ```

## Usage

### Test Environment

```bash
cd alice_isaac
python scripts/test_env.py
```

### Train

```bash
python scripts/train_ppo.py --num_envs 1024 --headless
```

### Training Options

- `--num_envs`: Number of parallel environments (default: 1024)
- `--headless`: Run without GUI
- `--max_iterations`: Maximum training iterations (default: 5000)
- `--seed`: Random seed (default: 42)

## Task Description

The humanoid robot with quadrotor backpack must:
- Hover at 2.5m height in Superman pose (horizontal, face-down)
- Maintain stable orientation
- Minimize drift and oscillations

## Environment Specs

- **Observations**: 39-dim (root_pos(3) + root_quat(4) + joint_pos(13) + root_lin_vel(3) + root_ang_vel(3) + joint_vel(13))
- **Actions**: 14-dim (10 actuated humanoid joints + 4 rotor thrusts)
- **Control Rate**: 50Hz
- **Episode Length**: 10 seconds
- **Parallel Envs**: 1024+ (GPU accelerated)

## Reward Structure

Dense reward shaping with milestone bonuses:
- Height tracking: -|z - 2.5|
- Orientation: Maintain Superman pose
- Velocity penalties: Minimize drift
- Control cost: Energy efficiency
- Action smoothness: Stable control
- Milestones: Airborne, oriented, hover, stable, perfect

## Reference

Based on the MuJoCo implementation in `alice_mujoco/`.
