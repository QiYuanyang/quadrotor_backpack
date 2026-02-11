# Superman Flight Configuration

## Overview

This document describes the **Superman flight configuration** for the humanoid quadrotor robot. In this configuration, the humanoid flies face-first through the air (Superman pose) with a quadrotor backpack.

## Coordinate Systems

### World Frame
- **+X**: East (right)
- **+Y**: North (forward)
- **+Z**: Up (against gravity)

### Body Frames
Each body part has its own local coordinate system.

## Flight Configuration

### 1. Torso Configuration

| Property | Value |
|----------|-------|
| Local X (RED) | -Z world (DOWN) |
| Local Y (GREEN) | +Y world (forward) |
| Local Z (BLUE) | +X world (right) |

**Purpose**: Humanoid faces downward (-Z world), flying Superman-style.

```
World View:
           +Z (UP)
            |
            |
            +-- +X (EAST)
           /
          +Y (NORTH)

Humanoid Orientation:
    Head points toward -Z (down/forward)
    Feet point toward +Z (up/backward)
```

### 2. Quadrotor Body Configuration

| Property | Value |
|----------|-------|
| Local X (RED) | +X world (right) |
| Local Y (GREEN) | +Z world (up) |
| Local Z (BLUE) | -Y world (backward) |

**Purpose**: Mounted on humanoid's back, oriented for thrust.

```
Quadrotor Orientation (back view):
    Local Z (BLUE) points backward (-Y world)
    Local Y (GREEN) points up (+Z world)
    Local X (RED) points right (+X world)
```

### 3. Rotor Configuration

| Property | Value |
|----------|-------|
| Rotor Local +Z | +Z world (UP) |
| Force Applied | Local -Z (DOWN) |
| Force Value | **Always negative** (thrust) |

**Purpose**: Each rotor generates thrust in its local -Z direction.

```
Rotor Thrust:
    Rotor spins, generating force
    Force direction: Local -Z (opposite of blade rotation axis)
    World effect: Pushes robot DOWN (-Z world)
    Result: Robot pushed UP (+Z world) by reaction force
```

## Force Application

### Thrust Formula
```
F_rotor = -thrust * (rotor_local_z_in_world)
```

Where:
- `thrust` > 0 (magnitude)
- `rotor_local_z_in_world` = rotor's local Z axis expressed in world frame
- Result is always in local -Z direction

### Example
If rotor's local Z points UP (+Z world):
```
F = -thrust * (+Z) = -Z (DOWN)
Force pushes rotor DOWN
Robot moves UP
```

## Reward Design (Feb 2026)

### Reward Hierarchy (Priority Order)

| Priority | Reward | Target | Scale | Purpose |
|----------|--------|--------|-------|---------|
| 1 | **torso_face_down** | Torso Local X → -Z world | +20.0 | Superman face-down pose |
| 2 | **quadrotor_upright** | Quadrotor Local Z → +Z world | +10.0 | Thrust axis pointing up |
| 3 | **height** | root_pos[:, 2] → 2.5m | +5.0 | Reach target altitude |
| 4 | **thrust_direction** | Thrust in Local -Z only | -5.0 | Penalize wrong direction |
| 5 | **alive** | Stay above min_height | +1.0 | Stay in the air |

### Reward Formulas

#### 1. Torso Face-Down Reward
```python
# Target: Torso Local X points DOWN (-Z world)
local_x = [1, 0, 0]  # Torso's local X axis
torso_x_world = rotate(local_x, torso_quat)  # Local X in world frame

# Dot product with -Z (down)
down_target = [0, 0, -1]
alignment = dot(torso_x_world, down_target)

# Reward: +1 when perfectly aligned (X points down)
rew_torso_face_down = 20.0 * alignment
```

#### 2. Quadrotor Upright Reward
```python
# Target: Quadrotor Local Z points UP (+Z world)
# Quadrotor quaternion w component = cos(θ/2)
# w=1 when upright, w=0 when 90° tilted, w=-1 when inverted
quad_w = quadrotor_quat.w
rew_quadrotor_upright = 10.0 * quad_w²
```

#### 3. Height Reward
```python
# Target: Hover at 2.5m
rew_height = 5.0 * root_pos.z
```

#### 4. Thrust Direction Penalty
```python
# Target: Thrust ONLY in Local -Z direction
# Correct: positive action → force in Local -Z
# Wrong: negative action → force in Local +Z

# Penalize negative thrust
negative_mask = rotor_actions < 0
penalty = abs(rotor_actions) * negative_mask
rew_thrust_direction = -5.0 * mean(penalty)
```

### Force Application

**CRITICAL**: Forces are applied at each rotor's position, in the rotor's local -Z direction.

```python
# For each rotor:
force_direction = rotate([0, 0, -1], rotor_quat)  # Local -Z in world
force = force_direction * thrust_magnitude
apply_force(force, rotor_position, body_id=quadrotor_body_id)
```

### Example Training Command

```bash
cd /home/qyy/hdd/work/alice_isaac/quad_rotor
./isaaclab.sh -p source/quad_rotor/quad_rotor/scripts/train.py --task QuadrotorDirect --headless
```

## Thrust Control

### Action Space
- 4 actions (one per rotor)
- Each action: thrust magnitude (0 to max)
- Applied in local rotor -Z direction

### Implementation
```python
# Pseudocode for thrust application
for i, rotor in enumerate(rotors):
    # Get rotor's local Z axis in world frame
    rotor_z_world = rotate_vector([0, 0, 1], rotor.quaternion)
    
    # Apply force in opposite direction (local -Z)
    force = -action[i] * rotor_z_world
    
    # Apply to robot
    robot.apply_force(force, rotor.position)
```

## Summary

| Component | Local +Z Direction | Force Direction | World Effect |
|-----------|-------------------|------------------|--------------|
| Torso | +X | N/A | N/A |
| Quadrotor | -Y | N/A | N/A |
| Rotor 1 | +Z | -Z | Robot UP |
| Rotor 2 | +Z | -Z | Robot UP |
| Rotor 3 | +Z | -Z | Robot UP |
| Rotor 4 | +Z | -Z | Robot UP |

## Files

| File | Purpose |
|------|---------|
| `humanoid_quadrotor.usd` | Robot USD file |
| `FLIGHT_CONFIG.md` | This file |
| `quad_rotor_env.py` | IsaacLab environment |
| `show_frame.py` | Visual debug script |

## References

- IsaacLab Documentation: https://isaac-sim.github.io/IsaacLab/main/
- Superman Flight: Humanoid flies face-first through air
