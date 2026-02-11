# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the humanoid quadrotor robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

HUMANOID_QUADROTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/qyy/hdd/work/alice_isaac/assets/humanoid_quadrotor/humanoid_quadrotor.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # Higher for stability with many bodies
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.4),  # Spawn at 1.4m height (standing)
        rot=(1.0, 0.0, 0.0, 0.0),  # Upright orientation (identity quaternion)
        joint_pos={
            # CRITICAL: Knee joints CANNOT be 0.0 due to limits [-160°, -2°]
            "right_knee": -1.4,  # Mid-range in radians (~-80°)
            "left_knee": -1.4,
            # Abdomen joints
            "abdomen_z": 0.0,
            "abdomen_y": 0.0,
            "abdomen_x": 0.0,
            # Hip joints (3 DOF per leg)
            "right_hip_x": 0.0,
            "right_hip_y": 0.0,
            "right_hip_z": 0.0,
            "left_hip_x": 0.0,
            "left_hip_y": 0.0,
            "left_hip_z": 0.0,
            # Shoulder joints (2 DOF per arm)
            "right_shoulder1": 0.0,
            "right_shoulder2": 0.0,
            "left_shoulder1": 0.0,
            "left_shoulder2": 0.0,
            # Elbow joints
            "right_elbow": 0.0,
            "left_elbow": 0.0,
        },
    ),
    actuators={
        # Humanoid body joints (17 actuated joints total)
        # Pattern matches: abdomen(3), hips(6), shoulders(4), knees(2), elbows(2)
        "body_joints": ImplicitActuatorCfg(
            joint_names_expr=["abdomen.*", ".*_hip_.*", ".*_shoulder.*", ".*_knee", ".*_elbow"],
            effort_limit_sim=100.0,  # Torque limit [N⋅m]
            stiffness=0.0,  # Pure torque control (no position feedback)
            damping=5.0,  # Joint damping
        ),
        # Note: Quadrotor is a fixed joint (not actuated)
        # Note: Rotor thrusts are applied via external forces in the environment
    },
)
"""Configuration for the humanoid quadrotor robot with quadrotor backpack.

This configuration defines a 15-body articulation with 17 actuated joints:
- Articulation root: torso/torso
- Fixed joints: quadrotor (backpack), right_foot, left_foot (3 joints)
- Actuated joints: 
  - Abdomen: abdomen_z, abdomen_y, abdomen_x (3 joints)
  - Hips: right/left_hip_x/y/z (6 joints)
  - Shoulders: right/left_shoulder1/2 (4 joints)
  - Knees: right/left_knee (2 joints)
  - Elbows: right/left_elbow (2 joints)
  Total: 17 actuated joints
- Bodies: torso, quadrotor, lwaist, pelvis, thighs(2), shins(2), feet(2), upper_arms(2), lower_arms(2) (15 bodies)

The quadrotor backpack provides thrust via external forces (not actuators) to lift the humanoid.
"""