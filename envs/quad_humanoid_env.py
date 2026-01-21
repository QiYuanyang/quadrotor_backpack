# Copyright (c) 2024
# Humanoid Quadrotor Environment

from __future__ import annotations

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate_inverse

from .quad_humanoid_env_cfg import QuadHumanoidEnvCfg


class QuadHumanoidEnv(DirectRLEnv):
    """Environment for humanoid with quadrotor backpack to hover in Superman pose."""

    cfg: QuadHumanoidEnvCfg

    def __init__(self, cfg: QuadHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find quadrotor body for thrust application
        self._quadrotor_body_idx, _ = self.robot.find_bodies(self.cfg.rotor_body_name)
        
        # Dynamically identify actuated joints (filter out fixed joints)
        # Fixed joints in USD: quadrotor, right_foot, left_foot
        all_joint_names = self.robot.joint_names
        self._actuated_joint_ids = [
            i for i, name in enumerate(all_joint_names) 
            if not any(fixed_name in name for fixed_name in ['quadrotor', 'foot'])
        ]
        self.num_actuated_joints = len(self._actuated_joint_ids)
        
        print(f"[INFO] Found {len(all_joint_names)} total joints, {self.num_actuated_joints} actuated")
        print(f"[INFO] All joint names: {all_joint_names}")
        print(f"[INFO] Actuated joint IDs: {self._actuated_joint_ids}")
        
        # Action storage for smoothness reward
        self._previous_actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )
        
        # Individual rotor thrust storage (4 rotors)
        self._rotor_thrusts = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Cache for rewards
        self._height_errors = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """Setup the scene with robot."""
        self.robot = Articulation(self.cfg.scene.robot)
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Split actions based on actual actuated joint count
        humanoid_actions = actions[:, :self.num_actuated_joints]  # First N: humanoid joints
        thrust_actions = actions[:, self.num_actuated_joints:]  # Last 4: rotor thrusts
        
        # Scale humanoid actions (torque control)
        self._humanoid_torques = humanoid_actions * self.cfg.action_scale_humanoid
        
        # Scale thrust actions from [-1, 1] to [0, 50], then apply gear ratio
        # This gives [0, 500N] per rotor
        self._rotor_thrusts = (thrust_actions + 1.0) * 0.5 * self.cfg.action_scale_thrust * self.cfg.thrust_gear

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # Apply joint torques to humanoid (only actuated joints)
        self.robot.set_joint_effort_target(
            self._humanoid_torques,
            joint_ids=self._actuated_joint_ids
        )

        total_force = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Sum all rotor forces (all point upward in local Z)
        total_force[:, 2] = self._rotor_thrusts.sum(dim=-1)

        # Compute torques from differential thrust (X configuration)
        # Roll (X-axis):  (right side - left side) * arm
        # Pitch (Y-axis): (front - back) * arm
        # Yaw (Z-axis):   (CW - CCW) * drag coefficient
        total_torque = torch.zeros(self.num_envs, 3, device=self.device)
        arm = self.cfg.rotor_distance
        
        # Roll: F1 + F4 (right side) - F2 - F3 (left side)
        total_torque[:, 0] = (self._rotor_thrusts[:, 0] + self._rotor_thrusts[:, 3] - 
                              self._rotor_thrusts[:, 1] - self._rotor_thrusts[:, 2]) * arm
        
        # Pitch: F1 + F2 (front) - F3 - F4 (back)
        total_torque[:, 1] = (self._rotor_thrusts[:, 0] + self._rotor_thrusts[:, 1] - 
                              self._rotor_thrusts[:, 2] - self._rotor_thrusts[:, 3]) * arm
        
        # Yaw: Rotor drag creates yaw torque (CW props: 1,3; CCW props: 2,4)
        drag_coeff = 0.01  # Drag coefficient (tune based on props)
        total_torque[:, 2] = (self._rotor_thrusts[:, 0] + self._rotor_thrusts[:, 2] - 
                              self._rotor_thrusts[:, 1] - self._rotor_thrusts[:, 3]) * drag_coeff
        
        # Apply force and torque to quadrotor body
        self.robot.set_external_force_and_torque(
            total_force,
            total_torque,
            body_ids=self._quadrotor_body_idx
        )

    def _get_observations(self) -> dict:
        """Get observations matching MuJoCo's 378-dim observation (qpos + qvel)."""
        # Get robot state
        root_pos_w = self.robot.data.root_pos_w  # (num_envs, 3)
        root_quat_w = self.robot.data.root_quat_w  # (num_envs, 4)
        root_lin_vel_w = self.robot.data.root_lin_vel_w  # (num_envs, 3)
        root_ang_vel_w = self.robot.data.root_ang_vel_w  # (num_envs, 3)
        joint_pos = self.robot.data.joint_pos  # (num_envs, num_joints)
        joint_vel = self.robot.data.joint_vel  # (num_envs, num_joints)
        
        # Concatenate to match MuJoCo observation structure:
        # qpos: root_pos(3) + root_quat(4) + joint_pos
        # qvel: root_lin_vel(3) + root_ang_vel(3) + joint_vel
        obs = torch.cat([
            root_pos_w,
            root_quat_w,
            joint_pos,
            root_lin_vel_w,
            root_ang_vel_w,
            joint_vel,
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards matching MuJoCo implementation."""
        # Get robot state
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w
        
        # Extract position and velocity
        z_pos = root_pos_w[:, 2]
        xy_velocity = root_lin_vel_w[:, :2]
        z_vel = root_lin_vel_w[:, 2]
        
        # === 1. Survival bonus ===
        survival_reward = torch.ones_like(z_pos) * 10.0
        
        # === 2. Height tracking reward ===
        height_error = torch.abs(z_pos - self.cfg.target_height)
        height_reward = -height_error
        self._height_errors = height_error  # Cache for termination
        
        # === 3. Orientation tracking (Superman pose: horizontal, face down) ===
        # Target: body Z-axis should point down in world frame [0, 0, -1]
        # Rotate world down vector into body frame
        world_down = torch.zeros(self.num_envs, 3, device=self.device)
        world_down[:, 2] = -1.0
        body_z_direction = quat_rotate_inverse(root_quat_w, world_down)
        
        # Compute orientation error (L2 distance from ideal body Z axis)
        target_body_z = torch.zeros_like(body_z_direction)
        target_body_z[:, 2] = 1.0  # In body frame, Z should point along local Z
        orientation_error = torch.norm(body_z_direction - target_body_z, dim=-1)
        orientation_reward = -2.0 * orientation_error
        
        # === 4. Velocity penalties ===
        xy_velocity_penalty = -0.1 * torch.sum(xy_velocity ** 2, dim=-1)
        z_velocity_penalty = -0.5 * (z_vel ** 2)
        
        # === 5. Control cost ===
        ctrl_cost = -0.005 * torch.sum(self.actions ** 2, dim=-1)
        
        # === 6. Action smoothness ===
        action_diff = self.actions - self._previous_actions
        smoothness_reward = -0.1 * torch.sum(action_diff ** 2, dim=-1)
        self._previous_actions = self.actions.clone()
        
        # === 7. Thrust usage bonus (encourage using thrust) ===
        max_thrust = self.cfg.action_scale_thrust * self.cfg.thrust_gear
        mean_thrust = torch.mean(self._rotor_thrusts / max_thrust, dim=-1)
        thrust_bonus = 0.5 * mean_thrust
        
        # === Milestone bonuses ===
        milestone_reward = torch.zeros_like(z_pos)
        
        # Milestone 1: Stay airborne
        milestone_reward += torch.where(z_pos > 2.0, 10.0, 0.0)
        
        # Milestone 2: Good orientation
        milestone_reward += torch.where(orientation_error < 0.3, 5.0, 0.0)
        
        # Milestone 3: Near target height
        milestone_reward += torch.where(height_error < 1.0, 10.0, 0.0)
        
        # Milestone 4: Stable hover
        stable_hover = (height_error < 0.5) & (torch.abs(z_vel) < 0.2)
        milestone_reward += torch.where(stable_hover, 20.0, 0.0)
        
        # Milestone 5: Perfect hover
        xy_speed = torch.norm(xy_velocity, dim=-1)
        perfect_hover = (height_error < 0.2) & (torch.abs(z_vel) < 0.1) & (xy_speed < 0.1)
        milestone_reward += torch.where(perfect_hover, 30.0, 0.0)
        
        # === Total reward ===
        total_reward = (
            survival_reward +
            height_reward +
            orientation_reward +
            xy_velocity_penalty +
            z_velocity_penalty +
            ctrl_cost +
            smoothness_reward +
            thrust_bonus +
            milestone_reward
        )
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        root_pos_w = self.robot.data.root_pos_w
        
        # Terminated if too high or drifted too far
        too_high = root_pos_w[:, 2] > 5.0
        too_far_x = torch.abs(root_pos_w[:, 0]) > 3.0
        too_far_y = torch.abs(root_pos_w[:, 1]) > 3.0
        
        terminated = too_high | too_far_x | too_far_y
        
        # Time limit truncation
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Get default states
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        
        # Reset robot state to Superman pose with noise
        # Position: [0, 0, 0.5] + noise
        default_root_state[:, 0] = 0.0
        default_root_state[:, 1] = 0.0
        default_root_state[:, 2] = 0.5
        default_root_state[:, :3] += torch.randn_like(default_root_state[:, :3]) * 0.05
        
        # Orientation: 90° pitch (Superman) 
        # quat for 90° pitch = [w, x, y, z] = [0.7071, 0, 0.7071, 0]
        default_root_state[:, 3] = 0.7071  # w
        default_root_state[:, 4] = 0.0     # x
        default_root_state[:, 5] = 0.7071  # y (pitch)
        default_root_state[:, 6] = 0.0     # z
        
        # Add environment origins
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Joint positions: zero + small noise
        joint_pos += torch.randn_like(joint_pos) * 0.05
        
        # Joint velocities: zero + small noise
        joint_vel += torch.randn_like(joint_vel) * 0.05
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset action history
        self._previous_actions[env_ids] = 0.0
        self._rotor_thrusts[env_ids] = 0.0
