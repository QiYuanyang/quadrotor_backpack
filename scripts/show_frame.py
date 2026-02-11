#!/usr/bin/env python3
"""
Visualize body frames (RGB = Local X, Y, Z axes) on torso, quadrotor, and 4 rotors.

Usage:
    cd /home/qyy/hdd/work/alice_isaac/quad_rotor
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate alice_isaac
    ./isaaclab.sh -p source/quad_rotor/quad_rotor/scripts/show_frame.py

Legend (RGB Frame):
    - RED   = Local +X axis
    - GREEN = Local +Y axis
    - BLUE  = Local +Z axis
    - Each frame shows the body's actual orientation in world space

Bodies with frames:
    - Torso (humanoid root)
    - Quadrotor (thrust axis)
    - 4 Rotors (attached to quadrotor)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize Z-axis on robot bodies.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Add source to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "source")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

# Import after launching
import torch
import numpy as np
import time

from quad_rotor.quad_rotor.tasks.direct.quad_rotor.quad_rotor_env import QuadRotorEnv
from quad_rotor.quad_rotor.tasks.direct.quad_rotor.quad_rotor_env_cfg import QuadRotorEnvCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),     2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ])
    return R


def main():
    print("\n" + "=" * 70)
    print("BODY FRAME VISUALIZATION (RGB = Local X, Y, Z)")
    print("=" * 70)
    print("\nBodies with frames:")
    print("  - Torso (humanoid root)")
    print("  - Quadrotor (thrust axis)")
    print("  - 4 Rotors (attached to quadrotor)")
    print("\nFrame Legend:")
    print("  RED   = Local +X axis")
    print("  GREEN = Local +Y axis")
    print("  BLUE  = Local +Z axis")
    print("\nSuperman Configuration:")
    print("  - Torso frame: Z points DOWN (humanoid faces ground)")
    print("  - Quadrotor frame: Z points horizontally (backward)")
    print("=" * 70 + "\n")

    # Create environment
    env_cfg = QuadRotorEnvCfg()
    env_cfg.scene.num_envs = 1
    env = QuadRotorEnv(cfg=env_cfg)
    env.reset()

    # Get robot
    robot = env.scene.articulations["robot"]

    # Find bodies
    body_map = {}
    body_map["torso"] = 0  # Root
    quad_ids, _ = robot.find_bodies("quadrotor")
    body_map["quadrotor"] = quad_ids[0]

    # Rotor offsets (approximate positions on quadrotor)
    arm = 0.2
    rotor_offsets = {
        "rotor1": np.array([ arm,  arm, 0.0]),
        "rotor2": np.array([-arm,  arm, 0.0]),
        "rotor3": np.array([-arm, -arm, 0.0]),
        "rotor4": np.array([ arm, -arm, 0.0]),
    }

    # Store initial root state to fix robot in place
    initial_root_state = robot.data.root_state_w.clone()
    robot.write_root_state_to_sim(initial_root_state)

    # Create Z-axis marker using frame_prim (shows RGB frame: X=red, Y=green, Z=blue)
    z_arrow_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ZAxisArrows",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.15, 0.15, 0.15),
            ),
        }
    )
    z_arrow_marker = VisualizationMarkers(z_arrow_cfg)

    # For frame_prim, no rotation needed - it shows the body's actual orientation
    # Red = Local X, Green = Local Y, Blue = Local Z

    print("Visualizing body frames (RGB = Local X, Y, Z)...")
    print("Robot is fixed at spawn position.\n")

    last_print = time.time()

    try:
        while simulation_app.is_running():
            # Apply zero actions
            actions = torch.zeros(env.num_envs, env.cfg.action_space, device=robot.device)
            env.step(actions)

            # Keep robot locked
            robot.write_root_state_to_sim(initial_root_state)

            # Get quadrotor data for rotor positions
            quad_pos = robot.data.body_pos_w[0, quad_ids[0], :].cpu().numpy()
            quad_quat = robot.data.body_quat_w[0, quad_ids[0], :].cpu().numpy()

            # Convert quaternion to rotation matrix
            qx, qy, qz, qw = quad_quat
            R_quad = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),     2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw),     1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
            ])

            # Collect all marker data
            translations = []
            orientations = []
            marker_indices = []

            # Add frame for torso and quadrotor
            for body_name, body_id in body_map.items():
                pos = robot.data.body_pos_w[0, body_id, :].cpu().numpy()
                quat = robot.data.body_quat_w[0, body_id, :].cpu().numpy()

                translations.append(torch.tensor([pos[0], pos[1], pos[2]], device=robot.device))
                orientations.append(torch.tensor([quat[0], quat[1], quat[2], quat[3]], device=robot.device))
                marker_indices.append(torch.tensor([0]))

            # Add frame for each rotor
            for rotor_name, offset in rotor_offsets.items():
                rotor_pos = quad_pos + R_quad @ offset

                translations.append(torch.tensor([rotor_pos[0], rotor_pos[1], rotor_pos[2]], device=robot.device))
                orientations.append(torch.tensor([quad_quat[0], quad_quat[1], quad_quat[2], quad_quat[3]], device=robot.device))
                marker_indices.append(torch.tensor([0]))

            # Visualize all frames
            if translations:
                z_arrow_marker.visualize(
                    translations=torch.stack(translations),
                    orientations=torch.stack(orientations),
                    marker_indices=torch.cat(marker_indices),
                )

            # Print status every second
            if time.time() - last_print >= 1.0:
                last_print = time.time()

                quad_z = R_quad[:, 2]

                print(f"[{time.strftime('%H:%M:%S')}] Quadrotor Local Z: [{quad_z[0]:+.2f}, {quad_z[1]:+.2f}, {quad_z[2]:+.2f}]")

                if abs(quad_z[2]) < 0.1 and abs(quad_z[0]) > 0.9:
                    print("  -> Points HORIZONTALLY (Superman configuration)")
                elif quad_z[2] > 0.8:
                    print("  -> Points UP")
                elif quad_z[2] < -0.8:
                    print("  -> Points DOWN")
                else:
                    print("  -> DIAGONAL")
                print()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
