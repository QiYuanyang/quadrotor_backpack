# Humanoid Quadrotor USD Structure

Generated: January 21, 2026

## File Location
`/home/qyy/hdd/work/alice_isaac/alice_isaac/assets/humanoid_quadrotor.usd`

## Hierarchy

```
World (Xform)
  humanoid_quadrotor (Xform)
    torso (Xform)
      torso (Xform, ARTICULATION_ROOT, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      quadrotor (Xform, RIGID_BODY)
        visuals (Xform)
        sites (Xform)
          rotor1 (Xform)
          rotor2 (Xform)
          rotor3 (Xform)
          rotor4 (Xform)
          imu_accel (Xform)
          imu_gyro (Xform)
        collisions (Xform)
      lwaist (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      pelvis (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      right_thigh (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      right_shin (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      right_foot (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      left_thigh (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      left_shin (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      left_foot (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      right_upper_arm (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      right_lower_arm (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      left_upper_arm (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
      left_lower_arm (Xform, RIGID_BODY)
        visuals (Xform)
        collisions (Xform)
    joints (Scope)
      quadrotor (PhysicsFixedJoint)
      lwaist (PhysicsJoint)
      right_thigh (PhysicsJoint)
      right_foot (PhysicsFixedJoint)
      left_thigh (PhysicsJoint)
      left_foot (PhysicsFixedJoint)
      right_upper_arm (PhysicsJoint)
      left_upper_arm (PhysicsJoint)
      abdomen_x (PhysicsRevoluteJoint)
      right_knee (PhysicsRevoluteJoint)
      left_knee (PhysicsRevoluteJoint)
      right_elbow (PhysicsRevoluteJoint)
      left_elbow (PhysicsRevoluteJoint)
```

## Articulation Root
- **Path:** `/World/humanoid_quadrotor/torso/torso`
- **Type:** Xform with ARTICULATION_ROOT API
- **Note:** Only 1 articulation root (worldBody has been disabled)

## Rigid Bodies (15 total)
1. `torso` - Main body (articulation root)
2. `quadrotor` - Quadrotor backpack
3. `lwaist` - Lower waist
4. `pelvis` - Pelvis
5. `right_thigh` - Right thigh
6. `right_shin` - Right shin
7. `right_foot` - Right foot
8. `left_thigh` - Left thigh
9. `left_shin` - Left shin
10. `left_foot` - Left foot
11. `right_upper_arm` - Right upper arm
12. `right_lower_arm` - Right lower arm
13. `left_upper_arm` - Left upper arm
14. `left_lower_arm` - Left lower arm

## Joints (13 total)

### Joint Names
`['quadrotor', 'lwaist', 'right_thigh', 'right_foot', 'left_thigh', 'left_foot', 'right_upper_arm', 'left_upper_arm', 'abdomen_x', 'right_knee', 'left_knee', 'right_elbow', 'left_elbow']`

### Joint Details

#### Fixed Joints (3)
- `quadrotor` - Fixed joint connecting quadrotor to torso
- `right_foot` - Fixed joint for right foot
- `left_foot` - Fixed joint for left foot

#### Revolute Joints with Limits (4)
- `abdomen_x` - Limits: `[-35.0°, 35.0°]` (radians: `[-0.611, 0.611]`)
- `right_knee` - Limits: `[-160.0°, -2.0°]` (radians: `[-2.793, -0.035]`)
- `left_knee` - Limits: `[-160.0°, -2.0°]` (radians: `[-2.793, -0.035]`)
- `right_elbow` - Limits: `[-90.0°, 50.0°]` (radians: `[-1.571, 0.873]`)
- `left_elbow` - Limits: `[-90.0°, 50.0°]` (radians: `[-1.571, 0.873]`)

#### Other Joints (6)
- `lwaist` - PhysicsJoint (no explicit limits shown)
- `right_thigh` - PhysicsJoint
- `left_thigh` - PhysicsJoint
- `right_upper_arm` - PhysicsJoint
- `left_upper_arm` - PhysicsJoint

## Quadrotor Sites
Located at: `/World/humanoid_quadrotor/torso/quadrotor/sites/`
- `rotor1` - First rotor position
- `rotor2` - Second rotor position
- `rotor3` - Third rotor position
- `rotor4` - Fourth rotor position
- `imu_accel` - IMU accelerometer
- `imu_gyro` - IMU gyroscope

## Important Notes for IsaacLab Configuration

### Prim Path Configuration
When spawning with `UsdFileCfg` in IsaacLab:
- Set `prim_path = "/World/envs/env_.*/Robot"`
- USD will spawn with internal structure preserved
- Articulation root will be at: `/World/envs/env_0/Robot/humanoid_quadrotor/torso/torso`

### Joint Initialization
**CRITICAL:** Knee joints CANNOT be initialized at 0.0 due to limits `[-160°, -2°]`

Recommended initialization patterns:
```python
# Option 1: Explicit joint names
joint_pos = {
    "right_knee": -1.4,  # Mid-range in radians (~-80°)
    "left_knee": -1.4,
}

# Option 2: Pattern matching (if no overlaps)
joint_pos = {
    ".*_knee": -1.4,
}
```

**Avoid overlapping patterns:** IsaacLab's string matching will reject configs like:
```python
# ❌ This will cause "Multiple matches" error
joint_pos = {
    "right_knee": -1.4,
    ".*": 0.0,  # Overlaps with "right_knee"
}
```

### Body Names for Control
For finding bodies with `find_bodies()`:
- Quadrotor body: Use pattern `".*quadrotor"` or exact name `"quadrotor"`
- Other bodies: `"torso"`, `"pelvis"`, `".*_thigh"`, `".*_shin"`, `".*_foot"`, `".*_arm"`, etc.

### Joint Limits in Radians
When setting actions or checking limits, remember USD stores degrees but IsaacLab uses radians:
- Knee: `[-2.793, -0.035]` rad
- Elbow: `[-1.571, 0.873]` rad
- Abdomen: `[-0.611, 0.611]` rad
