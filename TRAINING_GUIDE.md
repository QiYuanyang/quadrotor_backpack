# Quadrotor-Humanoid Hovering Training Guide

## Implementation Summary

### ✅ Completed Fixes

1. **Observation Space**: Fixed from 59 → 71 dimensions
2. **Enhanced Reward Function**:
   - Improved orientation reward using `w²` (smoother gradients)
   - Added hover stability reward (penalize xy-drift from spawn)
   - Added joint pose regularization (encourage neutral humanoid pose)
   - Added alive bonus (+0.5 per step for staying in air)

3. **Curriculum Learning**:
   - Stage 1: Spawn at target height (2.5m + noise)
   - Stage 2: Spawn 0.5m below target (2.0m)
   - Stage 3: Spawn at default height (1.4m) - full takeoff challenge
   - Auto-advances when 80% success rate over 100 episodes

4. **Metrics & Monitoring**:
   - Real-time tracking: mean height, orientation, xy-drift, height error
   - Termination reasons: too_low, too_high, drift, timeout rates
   - Curriculum stage tracking in TensorBoard

## Network Architecture Recommendations

### Recommended: Modular Asymmetric Actor-Critic with LSTM

```
ACTOR (Policy):
├─ Shared Encoder: [256, 256] MLP
├─ Joint Control Branch: [128] MLP → 17 dims (joint torques)
├─ Rotor Control Branch: LSTM[128] → [64] MLP → 4 dims (rotor thrusts)
└─ Action: Concat [joint_actions, rotor_actions]

CRITIC (Value):
├─ Observation Encoder: [512, 256] MLP
├─ Privileged Info (optional): mass, inertia, drag coeffs
└─ Value Head: [128] MLP → 1 scalar
```

### Why This Architecture?

**For Humanoid-Quadrotor Hovering:**

1. **Modular Design**:
   - Joint control is quasi-static (slow humanoid pose adjustments)
   - Rotor control requires high-frequency reactive adjustments
   - Separate branches allow different update rates and learning speeds

2. **LSTM for Rotor Control**:
   - Captures temporal dependencies in thrust sequences
   - Better handles aerodynamic effects and momentum
   - Helps with smooth takeoff/landing trajectories
   - Memory of recent thrust history aids stability

3. **Asymmetric Critic**:
   - Critic sees privileged information (true mass, CoM, etc.)
   - Faster learning during training
   - Actor learns robust policy without privileged info
   - Standard PPO/RL technique for sim-to-real transfer

### Alternative Architectures

**Option B: Single Deep MLP** (Simpler baseline)
```
Actor: [512, 256, 128] → 21 actions
Critic: [512, 256, 128] → 1 value
```
- ✅ Simpler, faster to train initially
- ❌ May struggle with temporal dependencies
- Use if: You want quick baseline results

**Option C: Hierarchical Policy** (Most complex)
```
High-level Planner (10Hz): Target pose + altitude
Low-level Controller (60Hz): Joint torques + rotor thrusts
```
- ✅ Best for complex multi-stage tasks
- ❌ Much harder to train (needs curriculum for both levels)
- Use if: Planning to extend to locomotion + flight

## Training Configuration

### PPO Hyperparameters (Recommended Start)

```python
# In your PPO training script
ppo_cfg = {
    "num_steps_per_env": 24,  # Short episodes for hovering
    "num_epochs": 8,
    "num_mini_batches": 4,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "value_loss_coef": 1.0,
    "entropy_coef": 0.01,  # Encourage exploration initially
    "max_grad_norm": 1.0,
}

# Network architecture
policy_cfg = {
    "actor_hidden_dims": [256, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
    "use_lstm": False,  # Start without LSTM, add later if needed
}
```

### Training Schedule

**Phase 1: Baseline (No Curriculum)**
```bash
# 1000 iterations ≈ 100M steps with 4096 envs
python scripts/train_ppo.py \
    --task Template-Quad-Rotor-Direct-v0 \
    --num_envs 1024 \
    --max_iterations 1000
```
- Start with fewer envs (1024) for faster iteration
- Monitor termination reasons
- Expected: ~50% timeout rate after 500 iterations

**Phase 2: With Curriculum**
- Set `use_curriculum = True` in config
- Should see curriculum advancing every ~200-400 iterations
- Stage 1→2: Typically fast (~100 iters)
- Stage 2→3: Harder, may take 500+ iters

**Phase 3: Fine-tuning**
- Increase `num_envs` to 4096 for final training
- Reduce `entropy_coef` to 0.001 for exploitation
- Train for 2000+ total iterations

## Monitoring & Debugging

### Key Metrics to Watch

**TensorBoard Dashboard:**
```
Episode Rewards:
- Total reward: Should increase to 100-150 range
- Height reward: Should approach 10.0 (max)
- Orientation reward: Should approach 5.0 (max)

Terminations:
- Timeout rate: TARGET >80% (means hovering full 10s)
- Too_low rate: Should decrease to <5%
- Drift rate: Should be <10%

Curriculum:
- Stage: Should advance 0→1→2 over training
```

### Common Issues & Solutions

**Issue 1: Robot immediately falls (too_low = 100%)**
- Solution: Increase `rew_scale_height` to 20.0
- Solution: Start curriculum at Stage 1 (spawn at target)
- Check: Rotor thrust sufficient? (4×50N = 200N vs ~588N humanoid weight)

**Issue 2: Robot flips over (orientation_w < 0)**
- Solution: Increase `rew_scale_orientation` to 10.0
- Solution: Add orientation penalty in early termination
- Check: Are rotor torques balanced? (print differential thrust)

**Issue 3: Curriculum not advancing**
- Solution: Lower `curriculum_success_threshold` to 0.6
- Solution: Increase `curriculum_min_episodes` to 200
- Check: Are episodes actually timing out? (check timeout_rate)

**Issue 4: Robot drifts horizontally**
- Solution: Increase `rew_scale_hover` to 5.0
- Solution: Decrease `max_xy_drift` to 2.0 for stricter termination
- Check: Is xy-velocity penalized enough?

## Expected Training Timeline

With 4096 environments @ 60 FPS:
- **100 iterations** (~10 min): Robot learns to use rotors, reduces falling
- **500 iterations** (~50 min): Achieves brief hovering (2-5s episodes)
- **1000 iterations** (~1.5 hrs): Stable hovering at target height
- **2000 iterations** (~3 hrs): Curriculum Stage 3, full takeoff capability

Total GPU memory: ~8-12 GB (depends on network size)

## Next Steps: Advanced Features

### 1. Add Rotor Saturation Penalty
```python
# In compute_rewards()
rotor_saturation = (torch.abs(actions[:, 17:21]) > 0.95).float().sum(dim=1)
rew_saturation = -0.1 * rotor_saturation
```

### 2. Action Smoothing
```python
# In compute_rewards()
action_diff = actions - previous_actions
rew_smoothness = -0.01 * torch.sum(torch.square(action_diff), dim=1)
```

### 3. Domain Randomization
```python
# In _reset_idx()
mass_scale = sample_uniform(0.8, 1.2, (len(env_ids), 1), self.device)
self.robot.set_mass_scale(mass_scale, env_ids)
```

### 4. Privileged Critic Information
```python
# Extend observation with privileged info for critic only
state_space = 71 + 10  # Add mass, inertia, drag coeffs, etc.
```

## Architecture Implementation Template

If you want to implement the modular LSTM architecture, modify your PPO training script:

```python
class ModularActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_obs, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
        )
        
        # Joint control branch (feedforward)
        self.joint_branch = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 17),  # 17 joint torques
        )
        
        # Rotor control branch (with LSTM)
        self.rotor_lstm = nn.LSTM(256, 128, batch_first=True)
        self.rotor_head = nn.Sequential(
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 4),  # 4 rotor thrusts
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, obs, lstm_states=None):
        # Encode observations
        features = self.encoder(obs)
        
        # Compute actions
        joint_actions = self.joint_branch(features)
        
        # LSTM for rotor control
        rotor_features, lstm_states = self.rotor_lstm(
            features.unsqueeze(1), lstm_states
        )
        rotor_actions = self.rotor_head(rotor_features.squeeze(1))
        
        # Concatenate actions
        actions = torch.cat([joint_actions, rotor_actions], dim=-1)
        
        # Compute value
        value = self.critic(obs)
        
        return actions, value, lstm_states
```

Good luck with training! 🚁🤖
