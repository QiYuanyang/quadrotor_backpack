"""
Test script to verify thrust forces are working correctly.
Applies maximum thrust to all rotors to see if robot can hover.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from envs.quad_humanoid_env import QuadHumanoidEnv

def test_max_thrust():
    """Test maximum thrust configuration."""
    env = QuadHumanoidEnv(render_mode="human")
    obs, info = env.reset()
    
    print("Testing maximum thrust on all rotors...")
    print("Robot mass: ~70kg, Weight: ~700N")
    print("Max thrust per rotor: 500N (action=50, gear=10)")
    print("Total max thrust: 2000N")
    print("\nApplying maximum thrust to all rotors...\n")
    
    for step in range(200):
        # Create action: zeros for humanoid joints, max thrust for rotors
        action = np.zeros(21)
        action[17:21] = 50.0  # Maximum thrust on all 4 rotors
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            z_pos = env.data.body("torso").xpos[2]
            z_vel = env.data.qvel[2]
            if 'force1' in info:
                total_force = sum([info[f'force{i}'] for i in range(1, 5)])
                print(f"Step {step:3d}: Height={z_pos:.2f}m, Vel={z_vel:+.2f}m/s, "
                      f"Forces: R1={info['force1']:.0f}N R2={info['force2']:.0f}N "
                      f"R3={info['force3']:.0f}N R4={info['force4']:.0f}N | Total={total_force:.0f}N")
        
        env.render()
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    env.close()
    print("\nTest complete!")

if __name__ == "__main__":
    test_max_thrust()
