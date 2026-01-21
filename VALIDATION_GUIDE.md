# Validation Guide for CUDA Index Fix

## What Was Fixed

### Problem
The environment code assumed 21 humanoid joints, but the USD file only contains 13 total joints:
- **10 actuated joints**: lwaist, right_thigh, left_thigh, right_upper_arm, left_upper_arm, abdomen_x, right_knee, left_knee, right_elbow, left_elbow
- **3 fixed joints**: quadrotor, right_foot, left_foot

When `set_joint_effort_target()` tried to apply torques to `joint_ids=list(range(21))`, it attempted to access joints 13-20 which don't exist, causing CUDA index out-of-bounds errors.

### Solution
1. **Dynamically identify actuated joints**: Filter out fixed joints at initialization
2. **Update action space**: Changed from 25 to 14 dimensions (10 actuated joints + 4 rotor thrusts)
3. **Update observation space**: Changed from 378 to 39 dimensions (based on actual 13 joints)
4. **Fix action application**: Only apply torques to actuated joints using filtered joint IDs

## Changes Made

### Files Modified
1. **envs/quad_humanoid_env.py**:
   - Added dynamic actuated joint identification in `__init__`
   - Updated `_pre_physics_step` to split actions based on actual joint count
   - Updated `_apply_action` to use filtered joint IDs
   - Updated observation comments to reflect actual dimensions

2. **envs/quad_humanoid_env_cfg.py**:
   - Changed `num_actions` from 25 to 14
   - Changed `num_observations` from 378 to 39
   - Updated `action_space` shape from (25,) to (14,)
   - Updated `observation_space` shape from (378,) to (39,)

3. **README.md**:
   - Updated documentation to reflect correct dimensions

## How to Validate

### 1. Check Environment Initialization
Run the test script and verify output:

```bash
cd /home/runner/work/quadrotor_backpack/quadrotor_backpack
python scripts/test_env.py --headless --num_envs 4
```

**Expected output should include**:
```
[INFO] Found 13 total joints, 10 actuated
[INFO] All joint names: ['lwaist', 'quadrotor', 'right_thigh', 'left_thigh', 'right_upper_arm', 'left_upper_arm', 'right_foot', 'left_foot', 'abdomen_x', 'right_knee', 'left_knee', 'right_elbow', 'left_elbow']
[INFO] Actuated joint IDs: [0, 2, 3, 4, 5, 8, 9, 10, 11, 12]
✓ Environment created successfully
  - Observation space: Box(-inf, inf, (4, 39), float32)
  - Action space: Box(-1.0, 1.0, (4, 14), float32)
```

### 2. Verify No CUDA Errors
The test should run without CUDA index out-of-bounds errors:

```bash
python scripts/test_env.py --headless --num_envs 4
```

**Should complete successfully** with output like:
```
Step   0: Height =  0.500m, Reward =   -5.23
Step  10: Height =  0.612m, Reward =    2.45
...
Step  90: Height =  1.234m, Reward =   15.67

✓ Test completed successfully!
```

### 3. Verify Action and Observation Shapes
Add this debug code to test_env.py after environment creation:

```python
# Verify shapes
print(f"\n=== Shape Verification ===")
print(f"Action dim: {env.action_space.shape[1]}")
print(f"Observation dim: {env.observation_space.shape[1]}")
print(f"Actuated joints: {env.num_actuated_joints}")
print(f"Total joints: {env.robot.num_joints}")

# Verify actions work
actions = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)
obs, rewards, terminated, truncated, info = env.step(actions)
print(f"Action accepted: ✓")
print(f"Observation shape: {obs['policy'].shape}")
```

**Expected output**:
```
=== Shape Verification ===
Action dim: 14
Observation dim: 39
Actuated joints: 10
Total joints: 13
Action accepted: ✓
Observation shape: torch.Size([4, 39])
```

## Key Points to Verify

### ✅ Action Space
- **Expected**: 14 dimensions (10 actuated joints + 4 rotor thrusts)
- **Verify**: `env.action_space.shape[1] == 14`

### ✅ Observation Space
- **Expected**: 39 dimensions (3 + 4 + 13 + 3 + 3 + 13)
- **Verify**: `env.observation_space.shape[1] == 39`

### ✅ Joint Application
- **Expected**: Torques applied only to 10 actuated joints
- **Verify**: No CUDA errors when stepping environment

### ✅ Actuated Joint IDs
- **Expected**: [0, 2, 3, 4, 5, 8, 9, 10, 11, 12] (indices of non-fixed joints)
- **Verify**: Check printed `[INFO] Actuated joint IDs`

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'isaaclab'"
**Solution**: This code requires IsaacLab 2.3.1+ to be installed. Follow IsaacLab installation instructions.

### Issue: "CUDA error: device-side assert triggered"
**Solution**: This should be fixed by our changes. If it persists:
1. Check that USD file hasn't changed
2. Verify `self._actuated_joint_ids` is correct
3. Run with `CUDA_LAUNCH_BLOCKING=1` for better error messages

### Issue: Observation/action shape mismatch in training
**Solution**: Restart training from scratch - old checkpoints expect 25/378 dimensions.

## Testing Checklist

Before considering the fix complete:

- [ ] Environment initializes without errors
- [ ] Correct number of actuated joints identified (10)
- [ ] Action space is 14-dimensional
- [ ] Observation space is 39-dimensional
- [ ] No CUDA index errors when stepping environment
- [ ] Environment runs for full episode (100+ steps)
- [ ] Random actions don't cause crashes
- [ ] Training script starts without errors

## Additional Notes

- The observation size changed from 378 to 39 because the MuJoCo version had more joints
- Any pre-trained models from the 378-dim version will NOT be compatible
- Training must be restarted from scratch with the new dimensions
- The reward computation still works correctly with the updated observations
