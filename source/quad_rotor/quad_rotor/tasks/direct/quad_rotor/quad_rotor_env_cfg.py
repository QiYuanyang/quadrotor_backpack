# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import os
# Add assets folder to path
sys.path.insert(0, "/home/qyy/hdd/work/alice_isaac/assets")
from humanoid_quadrotor import HUMANOID_QUADROTOR_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class QuadRotorEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0  # 10s episodes = 600 steps, good balance for hovering
    # - spaces definition
    action_space = 21  # 17 joint torques + 4 rotor thrusts
    observation_space = 71  # root_pos(3) + root_quat(4) + root_lin_vel(3) + root_ang_vel(3) + 
                           # joint_pos(17) + joint_vel(17) + projected_gravity(3) + prev_actions(21) = 71
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = HUMANOID_QUADROTOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - joint pattern for finding actuated joints dynamically (17 joints total)
    actuated_joints_pattern = "(abdomen.*|.*_hip_.*|.*_shoulder.*|.*_knee|.*_elbow)"
    
    # - rotor configuration
    rotor_body_name = "quadrotor"  # Body where forces are applied
    arm_length = 0.2  # Distance from center to rotor [m]
    max_thrust_per_rotor = 300.0  # Maximum thrust per rotor [N] (4×300=1200N, T/W≈2.04 for 60kg)
    rotor_drag_coefficient = 0.01  # For yaw torque from rotor drag
    
    # - action scales
    joint_torque_scale = 0.0  # FREEZE humanoid joints - only quadrotor control!
    rotor_thrust_scale = 300.0  # Scale rotor actions from [-1,1] to [0,300]N
    
    # - reward scales
    # Superman flight rewards (Feb 2026)
    # Priority: 1) Torso face-down, 2) Quadrotor upright, 3) Height, 4) Thrust direction
    
    rew_scale_height = 5.0  # Height reward (target: 2.5m)
    rew_scale_orientation = 0.0  # DISABLED - this rewards upright humanoid, wrong for Superman
    rew_scale_hover = 0.0  # Don't care about XY position
    rew_scale_joint_pose = 0.0  # Don't care about humanoid pose
    rew_scale_lin_vel = 0.0  # Let them move freely
    rew_scale_ang_vel = -0.3  # Don't spin too much
    rew_scale_joint_vel = 0.0  # Don't care about joint movement
    rew_scale_energy = 0.0  # Don't care about energy
    rew_scale_alive = 1.0  # Staying alive = staying in air
    rew_scale_quadrotor_upright = 10.0  # CRITICAL: Backpack must point up! (Local Z → +Z)
    rew_scale_thrust_usage = 0.0  # DISABLED - use thrust_direction instead
    rew_scale_torso_face_down = 20.0  # CRITICAL: Torso Local X → -Z (face-down Superman)
    rew_scale_thrust_direction = -5.0  # PENALTY: Negative thrust (wrong direction)
    
    # - target/termination conditions
    target_height = 2.5  # Target hovering height [m]
    min_height = 0.5  # Episode terminates if below this height [m]
    max_height = 5.0  # Episode terminates if above this height [m]
    max_xy_drift = 3.0  # Episode terminates if drifts too far horizontally [m]
    
    # - curriculum learning parameters
    use_curriculum = True  # Enable curriculum learning
    curriculum_stages = 3  # Number of curriculum stages
    curriculum_success_threshold = 0.8  # Success rate to advance stage
    curriculum_min_episodes = 100  # Minimum episodes before advancing
    initial_spawn_height_offset = 0.0  # Start at target height for Stage 1
    curriculum_spawn_height_offsets = [0.0, -0.5, -1.1]  # Height offsets per stage