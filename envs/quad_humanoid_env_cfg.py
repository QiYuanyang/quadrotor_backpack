# Copyright (c) 2024
# Configuration for Humanoid Quadrotor Environment

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
import gymnasium as gym

##
# Scene Configuration
##

@configclass
class QuadHumanoidSceneCfg(InteractiveSceneCfg):
    """Configuration for the humanoid quadrotor scene."""

    # Humanoid quadrotor robot
    robot: ArticulationCfg = MISSING


##
# Environment Configuration
##

@configclass
class QuadHumanoidEnvCfg(DirectRLEnvCfg):
    """Configuration for the humanoid quadrotor environment."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,  # 200 Hz simulation (0.005s)
        render_interval=2,  # Render every 2 physics steps
        gravity=(0.0, 0.0, -9.81),
    )

    # Environment settings
    episode_length_s = 10.0  # 10 seconds per episode
    decimation = 4  # Control at 50Hz (200Hz / 4)
    num_envs = 4  # Start with small number for testing
    num_actions = 25  # 21 humanoid joints + 4 rotor thrusts
    num_observations = 378  # Full state: qpos + qvel
    
    # Scene (robot configuration set below)
    scene: QuadHumanoidSceneCfg = QuadHumanoidSceneCfg(num_envs=4, env_spacing=4.0)

    # Task settings
    target_height = 2.5  # Target hover height (m)
    action_scale_humanoid = 0.4  # Joint torque scaling
    action_scale_thrust = 50.0  # Thrust force scaling (maps [-1,1] to [0,50])
    thrust_gear = 10.0  # Gear ratio for thrust (50*10 = 500N max per rotor)
    
    # Rotor configuration (sites in USD)
    rotor_body_name = ".*quadrotor"  # Body to apply thrust forces (regex pattern)
    rotor_distance = 0.2  # Distance from quadrotor center to rotors (m)
    
    # Spaces
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(378,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(25,))
    
    def __post_init__(self):
        """Post initialization to set robot configuration."""
        super().__post_init__()
        # Set robot configuration
        self.scene.robot = HumanoidQuadrotorCfg()


##
# Humanoid Quadrotor Robot Configuration
##

@configclass 
class HumanoidQuadrotorCfg(ArticulationCfg):
    """Configuration for humanoid with quadrotor backpack."""
    
    # Spawn USD reference at Robot, articulation will be at Robot/humanoid_quadrotor/torso
    prim_path = "/World/envs/env_.*/Robot"
    
    spawn = sim_utils.UsdFileCfg(
        usd_path="/home/qyy/hdd/work/alice_isaac/alice_isaac/assets/humanoid_quadrotor.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )
    
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Start 0.5m above ground
        rot=(0.7071, 0.0, 0.7071, 0.0),  # Superman pose: 90° pitch (face down)
        joint_pos={
            "right_knee": -1.4,  # Mid-range: -80° in radians (limits: [-2.793, -0.035])
            "left_knee": -1.4,
        },
        joint_vel={
            ".*": 0.0,
        },
    )
    
    actuators = {
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,  # Pure torque control
            damping=0.0,
        ),
    }

