"""
Debug script to visualize thrust direction.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from envs.quad_humanoid_env import QuadHumanoidEnv

def debug_thrust_direction():
    """Debug thrust direction in world frame."""
    env = QuadHumanoidEnv(render_mode="human")
    obs, info = env.reset()
    
    print("Analyzing thrust direction...")
    print("=" * 60)
    
    # Get humanoid orientation
    torso_quat = env.data.body("torso").xquat  # [w, x, y, z]
    torso_mat = env.data.body("torso").xmat.reshape(3, 3)
    print(f"Torso quaternion: {torso_quat}")
    print(f"Torso rotation matrix:\n{torso_mat}")
    
    # Get quadrotor orientation
    quad_quat = env.data.body("quadrotor").xquat
    quad_mat = env.data.body("quadrotor").xmat.reshape(3, 3)
    print(f"\nQuadrotor quaternion: {quad_quat}")
    print(f"Quadrotor rotation matrix:\n{quad_mat}")
    
    # Thrust in quadrotor's local frame (should be Z-axis = [0, 0, 1])
    local_thrust = np.array([0, 0, 1])
    
    # Transform to world frame
    world_thrust = quad_mat @ local_thrust
    print(f"\nThrust direction in world frame: {world_thrust}")
    print(f"World Z-axis (desired up): [0, 0, 1]")
    print(f"Dot product with world Z: {np.dot(world_thrust, [0, 0, 1]):.3f}")
    print(f"  (1.0 = straight up, 0.0 = horizontal, -1.0 = straight down)")
    
    # Weight vs max thrust
    print(f"\n{'='*60}")
    print(f"Weight: ~700N (70kg × 10 m/s²)")
    print(f"Max thrust: 2000N (4 × 500N)")
    print(f"Effective upward thrust: {2000 * np.dot(world_thrust, [0, 0, 1]):.1f}N")
    print(f"{'='*60}")
    
    if np.dot(world_thrust, [0, 0, 1]) < 0.5:
        print("\n⚠️  WARNING: Thrust is NOT pointing upward!")
        print("The robot cannot lift off with this configuration.")
        print(f"Suggestion: Thrust needs to point more upward (dot product > 0.5)")
    
    env.close()

if __name__ == "__main__":
    debug_thrust_direction()
