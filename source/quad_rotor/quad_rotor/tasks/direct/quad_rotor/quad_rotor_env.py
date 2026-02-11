# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import numpy as np
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_apply_inverse, quat_apply

from .quad_rotor_env_cfg import QuadRotorEnvCfg

# Headless frame capture
try:
    import carb
    import omni.usd
    from pxr import Usd, UsdGeom, Gf, UsdRender
    HEADLESS_CAPTURE_AVAILABLE = True
except ImportError:
    HEADLESS_CAPTURE_AVAILABLE = False


class QuadRotorEnv(DirectRLEnv):
    cfg: QuadRotorEnvCfg

    def __init__(self, cfg: QuadRotorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find actuated joint indices dynamically using regex pattern
        self._actuated_joint_ids, _ = self.robot.find_joints(self.cfg.actuated_joints_pattern)
        
        # Find quadrotor body for applying rotor forces
        self._quadrotor_body_ids, _ = self.robot.find_bodies(self.cfg.rotor_body_name)
        
        # Storage for actions
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # Storage for spawn positions (for hover reward)
        self._spawn_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Curriculum learning tracking
        self._curriculum_stage = 0
        self._curriculum_episode_count = 0
        self._curriculum_success_count = 0
        
        # Termination reason tracking
        self._termination_reasons = {
            "too_low": torch.zeros(self.num_envs, device=self.device),
            "too_high": torch.zeros(self.num_envs, device=self.device),
            "drift": torch.zeros(self.num_envs, device=self.device),
            "timeout": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Setup rotor thrust visualization markers with color-coded arrows
        # Match XML site colors: Rotor 1: Red, Rotor 2: Green, Rotor 3: Blue, Rotor 4: Yellow
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ThrustArrows",
            markers={
                "rotor1_red": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.05, 0.05, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "rotor2_green": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.05, 0.05, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "rotor3_blue": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.05, 0.05, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                "rotor4_yellow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.05, 0.05, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            }
        )
        self._thrust_markers = VisualizationMarkers(marker_cfg)
        
        # Define rotor positions in body frame (X-configuration)
        arm = self.cfg.arm_length
        self._rotor_offsets = torch.tensor([
            [ arm,  arm, 0.0],  # Rotor 1: front-right
            [-arm,  arm, 0.0],  # Rotor 2: front-left
            [-arm, -arm, 0.0],  # Rotor 3: back-left
            [ arm, -arm, 0.0],  # Rotor 4: back-right
        ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)  # (num_envs, 4, 3)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Split actions: first 17 are joint torques, last 4 are rotor thrusts
        joint_torques = self.actions[:, :17] * self.cfg.joint_torque_scale
        rotor_actions = self.actions[:, 17:21]  # No clamping - purely penalty-based!
        
        # Apply joint torques
        self.robot.set_joint_effort_target(joint_torques, joint_ids=self._actuated_joint_ids)
        
        # Convert action to thrust (allow negative for penalty to work)
        rotor_thrusts = rotor_actions * self.cfg.max_thrust_per_rotor

        # DEBUG: Print rotor info every 300 steps (10 seconds)
        if self.common_step_counter % 300 == 0 and self.num_envs > 0:
            quadrotor_quat = self.robot.data.body_quat_w[0, self._quadrotor_body_ids[0], :]
            quat_w = quadrotor_quat[3].item()
            num_negative = (rotor_actions < 0.0).sum().item()
            mean_negative = rotor_actions[rotor_actions < 0.0].mean().item() if num_negative > 0 else 0.0
            print(f"\n[Step {self.common_step_counter}] Env 0 Debug:")
            print(f"  Rotor actions: {rotor_actions[0].cpu().numpy()}")
            print(f"  Rotor thrusts [N]: {rotor_thrusts[0].cpu().numpy()}")
            print(f"  Total thrust: {rotor_thrusts[0].sum():.1f}N (need ~588N)")
            print(f"  Negative rotors: {num_negative}/{self.num_envs * 4} (avg: {mean_negative:.3f})")
            print(f"  Quadrotor w: {quat_w:.3f} (1.0=upright)")

        # Get quadrotor body state for applying per-rotor forces
        body_pos = self.robot.data.body_pos_w[:, self._quadrotor_body_ids[0], :]  # (num_envs, 3)
        body_quat = self.robot.data.body_quat_w[:, self._quadrotor_body_ids[0], :]  # (num_envs, 4)
        
        # Transform rotor offsets from body frame to world frame
        # _rotor_offsets: (num_envs, 4, 3)
        rotor_positions_world = body_pos.unsqueeze(1) + quat_apply(
            body_quat.unsqueeze(1).repeat(1, 4, 1),
            self._rotor_offsets
        )
        
        # Get rotor quaternions in world frame (same as quadrotor body quaternion)
        rotor_quats_world = body_quat.unsqueeze(1).repeat(1, 4, 1)  # (num_envs, 4, 4)
        
        # Compute force direction for each rotor: Local -Z in world frame
        # Local -Z = rotate [0, 0, -1] by rotor quaternion
        # Isaac quaternion format: [x, y, z, w]
        # We need to apply force in rotor's local -Z direction
        # Using quat_apply with [0, 0, -1] gives us the local -Z in world
        local_neg_z = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).unsqueeze(0)
        force_dirs_world = quat_apply(rotor_quats_world, local_neg_z)  # (num_envs, 4, 3)
        
        # Compute per-rotor forces: thrust * direction
        # thrust can be negative (penalized), direction is always local -Z
        rotor_forces = force_dirs_world * rotor_thrusts.unsqueeze(-1)  # (num_envs, 4, 3)
        
        # DEBUG: Check force directions
        if self.common_step_counter % 300 == 0 and self.num_envs > 0:
            quad_z_world = force_dirs_world[0, 0, :].cpu().numpy()
            print(f"  Rotor 1 force dir (local -Z in world): [{quad_z_world[0]:+.3f}, {quad_z_world[1]:+.3f}, {quad_z_world[2]:+.3f}]")

        # Apply forces at each rotor position (SUPERMAN: thrust pushes in local -Z = robot UP)
        # Shape: forces (num_envs, 4, 3), positions (num_envs, 4, 3)
        # set_external_force_and_torque expects shape (num_envs, num_bodies, 3)
        # We have 1 body (quadrotor) but need to apply forces at different points
        
        # Apply force for each environment
        for env_idx in range(self.num_envs):
            for rotor_idx in range(4):
                force = rotor_forces[env_idx, rotor_idx, :].unsqueeze(0)  # (1, 3)
                pos = rotor_positions_world[env_idx, rotor_idx, :].unsqueeze(0)  # (1, 3)
                body_id = self._quadrotor_body_ids[env_idx] if env_idx < len(self._quadrotor_body_ids) else self._quadrotor_body_ids[0]
                self.robot.set_external_force(
                    force, 
                    pos, 
                    body_ids=torch.tensor([body_id], device=self.device)
                )
        
        # Update thrust visualization
        if self.sim.has_gui():
            self._update_thrust_visualization(rotor_thrusts)

        # Headless frame capture: save tmp.png every 1 second (60 steps at 60 Hz)
        if not self.sim.has_gui() and self.common_step_counter % 60 == 0:
            self._capture_frame()

    def _update_thrust_visualization(self, rotor_thrusts: torch.Tensor) -> None:
        """Update visual markers showing rotor thrust forces as arrows."""
        # Get quadrotor body state
        body_pos = self.robot.data.body_pos_w[:, self._quadrotor_body_ids[0], :]
        body_quat = self.robot.data.body_quat_w[:, self._quadrotor_body_ids[0], :]
        
        # Transform rotor offsets from body frame to world frame
        rotor_positions = body_pos.unsqueeze(1) + quat_apply(
            body_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            self._rotor_offsets.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        
        # Arrow orientation: rotate body quaternion to align arrow with thrust direction
        # Arrow mesh points along +X by default, we need to rotate it to point along body +Z (thrust up)
        # This is a -90-degree rotation around Y-axis to point arrow upward
        from isaaclab.utils.math import quat_mul
        
        # Rotation to align arrow (+X) with thrust direction (+Z): -90° around Y
        # Isaac uses [x,y,z,w] quaternion format: 90° around -Y = [0, -0.7071, 0, 0.7071]
        arrow_to_thrust_quat = torch.tensor([0.0, -0.7071, 0.0, 0.7071], device=self.device)  # [x,y,z,w]
        
        # Combine: first rotate arrow to align with body Z, then apply body orientation
        marker_orientations = quat_mul(
            body_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            arrow_to_thrust_quat.unsqueeze(0).repeat(self.num_envs * 4, 1)
        )
        
        # Scale arrows by thrust magnitude
        thrust_scales = (rotor_thrusts / self.cfg.max_thrust_per_rotor).unsqueeze(-1).repeat(1, 1, 3)
        marker_scales = thrust_scales.reshape(-1, 3) * 1.0  # Scale for visibility
        
        # Create marker indices to select color prototype for each rotor
        # Prototype 0=red, 1=green, 2=blue, 3=yellow (matching XML site colors)
        marker_indices = torch.arange(4, device=self.device).repeat(self.num_envs)
        
        # Update markers with color-coding
        self._thrust_markers.visualize(
            translations=rotor_positions.reshape(-1, 3),
            orientations=marker_orientations,
            scales=marker_scales,
            marker_indices=marker_indices,
        )

    def _capture_frame(self) -> None:
        """Capture current viewport to tmp.png in project root (headless only).

        Saves to: /home/qyy/hdd/work/alice_isaac/tmp.png
        Called every 60 steps (1 second at 60 Hz simulation).
        """
        if not HEADLESS_CAPTURE_AVAILABLE:
            return

        try:
            output_path = Path("/home/qyy/hdd/work/alice_isaac/tmp.png")

            viewport = omni.usd.get_context().get_stage()
            if viewport is None:
                return

            render_product_path = "/World/RenderProduct"
            render_product_prim = viewport.GetPrimAtPath(render_product_path)

            if render_product_prim:
                import omni.kit.viewport.utility

                active_viewport = omni.kit.viewport.utility.get_active_viewport()
                if active_viewport is None:
                    return

                width = active_viewport.resolution.x
                height = active_viewport.resolution.y

                if width > 0 and height > 0:
                    texture = active_viewport.get_texture()
                    if texture is not None:
                        texture_data = texture.read_rgba(width, height)
                        if texture_data is not None:
                            img_array = np.frombuffer(texture_data, dtype=np.uint8)
                            img_array = img_array.reshape((height, width, 4))
                            img_array = np.flipud(img_array)
                            img_array = img_array[:, :, :3]
                            from PIL import Image
                            img = Image.fromarray(img_array, mode="RGB")
                            img.save(output_path, "PNG")

        except Exception:
            pass

    def _get_observations(self) -> dict:
        # Get robot state
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        joint_pos = self.robot.data.joint_pos[:, self._actuated_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._actuated_joint_ids]
        
        # Compute projected gravity in body frame
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        projected_gravity = quat_apply_inverse(root_quat, gravity_vec)
        
        # Concatenate all observations
        obs = torch.cat(
            [
                root_pos,  # 3
                root_quat,  # 4
                root_lin_vel,  # 3
                root_ang_vel,  # 3
                joint_pos,  # 17
                joint_vel,  # 17
                projected_gravity,  # 3
                self._previous_actions,  # 21
            ],
            dim=-1,
        )
        
        # Update previous actions
        self._previous_actions = self.actions.clone()
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        # Get torso quaternion (root of humanoid)
        torso_quat = self.robot.data.root_quat_w
        # Get quadrotor body quaternion (separate from root/pelvis)
        quadrotor_quat = self.robot.data.body_quat_w[:, self._quadrotor_body_ids[0], :]
        return compute_rewards(
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_orientation,
            self.cfg.rew_scale_hover,
            self.cfg.rew_scale_joint_pose,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_energy,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_quadrotor_upright,
            self.cfg.rew_scale_thrust_usage,
            self.cfg.rew_scale_torso_face_down,
            self.cfg.rew_scale_thrust_direction,
            self.cfg.target_height,
            root_pos,
            self._spawn_positions,
            torso_quat,
            quadrotor_quat,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w,
            self.robot.data.joint_pos[:, self._actuated_joint_ids],
            self.robot.data.joint_vel[:, self._actuated_joint_ids],
            self.actions,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        
        # Termination conditions
        too_low = root_pos[:, 2] < self.cfg.min_height
        too_high = root_pos[:, 2] > self.cfg.max_height
        drifted = torch.norm(root_pos[:, :2], dim=1) > self.cfg.max_xy_drift
        
        terminated = too_low | too_high | drifted
        
        # Track termination reasons (accumulate over all envs)
        self._termination_reasons["too_low"] += too_low.float()
        self._termination_reasons["too_high"] += too_high.float()
        self._termination_reasons["drift"] += drifted.float()
        self._termination_reasons["timeout"] += time_out.float()
        
        # Add metrics to extras for logging
        self.extras["metrics/mean_height"] = root_pos[:, 2].mean()
        self.extras["metrics/mean_orientation_w"] = self.robot.data.root_quat_w[:, 0].mean()
        self.extras["metrics/mean_xy_drift"] = torch.norm(root_pos[:, :2] - self._spawn_positions[:, :2], dim=1).mean()
        self.extras["metrics/height_error"] = torch.abs(root_pos[:, 2] - self.cfg.target_height).mean()
        
        # Track negative rotor actions
        rotor_actions = self.actions[:, 17:21]
        num_negative = (rotor_actions < 0.0).float().sum()
        total_rotors = self.num_envs * 4
        self.extras["metrics/negative_rotor_ratio"] = num_negative / total_rotors
        self.extras["metrics/mean_rotor_action"] = rotor_actions.mean()
        
        # Termination reason statistics (normalize by number of terminated envs this step)
        num_terminated = (terminated | time_out).sum()
        if num_terminated > 0:
            self.extras["terminations/too_low_rate"] = too_low.sum().float() / num_terminated
            self.extras["terminations/too_high_rate"] = too_high.sum().float() / num_terminated
            self.extras["terminations/drift_rate"] = drifted.sum().float() / num_terminated
            self.extras["terminations/timeout_rate"] = time_out.sum().float() / num_terminated
        
        # Curriculum tracking
        if self.cfg.use_curriculum:
            self.extras["curriculum/stage"] = float(self._curriculum_stage)
            
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset joint positions and velocities to defaults
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Determine spawn height based on curriculum stage
        default_root_state = self.robot.data.default_root_state[env_ids]
        
        if self.cfg.use_curriculum and self._curriculum_stage < self.cfg.curriculum_stages:
            # Get height offset for current curriculum stage
            spawn_offset = self.cfg.curriculum_spawn_height_offsets[self._curriculum_stage]
            # Start at target height + curriculum offset
            spawn_height = self.cfg.target_height + spawn_offset
        else:
            # Default spawn height from asset config (1.4m)
            spawn_height = default_root_state[:, 2].clone()
        
        # Add small randomization
        height_noise = sample_uniform(-0.2, 0.2, (len(env_ids),), self.device)
        default_root_state[:, 2] = spawn_height + height_noise
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Store spawn positions for hover reward
        self._spawn_positions[env_ids] = default_root_state[:, :3].clone()

        # Write states to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset previous actions
        self._previous_actions[env_ids] = 0.0
        
        # Update curriculum tracking (count successful episodes that timed out)
        if self.cfg.use_curriculum and len(env_ids) > 0:
            # Count timeouts as successful episodes
            timed_out_envs = self.episode_length_buf[env_ids] >= self.max_episode_length - 1
            self._curriculum_success_count += timed_out_envs.sum().item()
            self._curriculum_episode_count += len(env_ids)
            
            # Check if we should advance curriculum stage
            if self._curriculum_episode_count >= self.cfg.curriculum_min_episodes:
                success_rate = self._curriculum_success_count / self._curriculum_episode_count
                if success_rate >= self.cfg.curriculum_success_threshold:
                    if self._curriculum_stage < self.cfg.curriculum_stages - 1:
                        self._curriculum_stage += 1
                        print(f"\n🎓 Curriculum advanced to Stage {self._curriculum_stage + 1}/{self.cfg.curriculum_stages}")
                        print(f"   Success rate: {success_rate:.2%} (threshold: {self.cfg.curriculum_success_threshold:.2%})")
                        print(f"   New spawn height offset: {self.cfg.curriculum_spawn_height_offsets[self._curriculum_stage]:.2f}m\n")
                    # Reset counters
                    self._curriculum_episode_count = 0
                    self._curriculum_success_count = 0


@torch.jit.script
def compute_rewards(
    rew_scale_height: float,
    rew_scale_orientation: float,
    rew_scale_hover: float,
    rew_scale_joint_pose: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_joint_vel: float,
    rew_scale_energy: float,
    rew_scale_alive: float,
    rew_scale_quadrotor_upright: float,
    rew_scale_thrust_usage: float,
    rew_scale_torso_face_down: float,
    rew_scale_thrust_direction: float,
    target_height: float,
    root_pos: torch.Tensor,
    spawn_positions: torch.Tensor,
    torso_quat: torch.Tensor,
    quadrotor_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
):
    from isaaclab.utils.math import quat_apply
    
    # Height reward - the higher the better!
    # Linear reward proportional to height above ground
    rew_height = rew_scale_height * root_pos[:, 2]
    
    # Orientation reward (stay upright: maximize w² component)
    # w=1 when upright, w→0 when tilted, w=-1 when inverted
    # NOTE: This rewards humanoid standing upright, which is WRONG for Superman
    # Use torso_face_down instead for Superman flight
    quat_w = torso_quat[:, 0]
    rew_orientation = rew_scale_orientation * torch.square(quat_w)
    
    # Hover stability reward (stay near spawn position in xy-plane)
    xy_drift = torch.norm(root_pos[:, :2] - spawn_positions[:, :2], dim=1)
    rew_hover = rew_scale_hover * torch.exp(-xy_drift / 1.0)
    
    # Joint pose regularization (encourage neutral humanoid pose)
    rew_joint_pose = rew_scale_joint_pose * torch.sum(torch.square(joint_pos), dim=1)
    
    # Velocity penalties (scales are already negative)
    rew_lin_vel = rew_scale_lin_vel * torch.sum(torch.square(root_lin_vel), dim=1)
    rew_ang_vel = rew_scale_ang_vel * torch.sum(torch.square(root_ang_vel), dim=1)
    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=1)
    
    # Energy penalty (scale is already negative)
    rew_energy = rew_scale_energy * torch.sum(torch.square(actions), dim=1)
    
    # Alive bonus (encourage staying in the air)
    rew_alive = rew_scale_alive
    
    # Quadrotor orientation reward
    # Isaac format is [x,y,z,w], so w is at index 3
    quadrotor_quat_w = quadrotor_quat[:, 3]  # w component
    rew_quadrotor_upright = rew_scale_quadrotor_upright * torch.square(quadrotor_quat_w)
    
    # Thrust usage reward - encourage using thrust (any direction)
    rotor_actions = actions[:, 17:21]
    avg_thrust_magnitude = torch.abs(rotor_actions).mean(dim=1)
    rew_thrust_usage = rew_scale_thrust_usage * avg_thrust_magnitude
    
    # ============================================================
    # SUPERMAN FLIGHT REWARDS (Feb 2026)
    # ============================================================
    
    # 1. Torso Face-Down Reward
    # Target: Torso Local X should point DOWN (-Z world) for Superman flight
    # Superman: Humanoid flies face-first through air, torso X points -Z
    # Get torso local X axis in world frame using quat_apply
    
    # local_x = [1, 0, 0] in body frame
    local_x = torch.tensor([1.0, 0.0, 0.0], device=torso_quat.device).unsqueeze(0).repeat(torso_quat.shape[0], 1)
    torso_x_world = quat_apply(torso_quat, local_x)
    
    # Target: torso X should point DOWN (-Z world)
    down_target = torch.tensor([0.0, 0.0, -1.0], device=torso_quat.device).unsqueeze(0).repeat(torso_quat.shape[0], 1)
    # Dot product: +1 means aligned with down, -1 means aligned with up
    face_down_alignment = torch.sum(torso_x_world * down_target, dim=-1)
    
    # Reward: positive when X points down (face-down Superman pose)
    rew_torso_face_down = rew_scale_torso_face_down * face_down_alignment
    
    # 2. Thrust Direction Penalty
    # Target: Thrust should be in rotor's local -Z direction
    # We penalize negative rotor actions (wrong direction)
    # Correct thrust: positive action = force in local -Z = robot UP
    # Wrong thrust: negative action = force in local +Z = robot DOWN
    
    # Penalize negative thrust (wrong direction)
    negative_thrust_mask = rotor_actions < 0
    negative_thrust_penalty = torch.abs(rotor_actions) * negative_thrust_mask.float()
    avg_negative_thrust = negative_thrust_penalty.mean(dim=1)
    
    # Apply penalty (negative scale = penalty)
    rew_thrust_direction = rew_scale_thrust_direction * avg_negative_thrust
    
    # ============================================================
    
    total_reward = (
        rew_height + rew_orientation + rew_hover + rew_joint_pose + 
        rew_lin_vel + rew_ang_vel + rew_joint_vel + rew_energy + rew_alive +
        rew_quadrotor_upright + rew_thrust_usage + rew_torso_face_down + rew_thrust_direction
    )
    return total_reward