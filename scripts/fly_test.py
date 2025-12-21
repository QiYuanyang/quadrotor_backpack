import sys
import os
import numpy as np
import gymnasium as gym
import time

# Add the project root to path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quad_humanoid_env import QuadHumanoidEnv

def main():
    # Create env with render mode human to visualize if possible, or just run logic
    # Note: In a headless environment 'human' might fail or open a window you can't see.
    try:
        env = QuadHumanoidEnv(render_mode="human")
    except Exception as e:
        print(f"Could not initialize with render_mode='human': {e}")
        env = QuadHumanoidEnv(render_mode=None)

    obs, info = env.reset()
    print("Simulation started. Testing thrust...")

    # Simulation loop
    for i in range(500):
        # Action space: 17 humanoid joints + 4 rotors
        # Indices 17, 18, 19, 20 are the rotors
        action = np.zeros(env.action_space.shape)
        
        # Apply 40% thrust (Control range 0-50, so 20. Gear=10 -> 200N per rotor -> 800N total)
        # Humanoid weight is approx 700N. This should lift it slowly.
        thrust_ctrl = 20.0 
        action[17:21] = thrust_ctrl
        
        # Keep legs somewhat stiff? 
        # The humanoid default controller expects actions in range -0.4 to 0.4 for joints
        # We leave them at 0 (damped)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            z_pos = info['z_pos']
            vz = info.get('z_velocity', 0) # We didn't put z_vel in info yet, but let's check z_pos
            print(f"Step {i}: Height={z_pos:.3f} m")
            
        if terminated or truncated:
            print("Terminated/Truncated")
            break
            
        # time.sleep(0.01) # Slow down for visualization if running locally

    env.close()

if __name__ == "__main__":
    main()
