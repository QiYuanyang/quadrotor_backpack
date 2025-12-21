import sys
import os
import numpy as np
import gymnasium as gym

# Add the project root to path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quad_humanoid_env import QuadHumanoidEnv

def main():
    env = QuadHumanoidEnv(render_mode=None)
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    
    obs, info = env.reset()
    print("Initial Observation:", obs.shape)
    
    # Run for 100 steps
    for i in range(100):
        # Random action
        action = env.action_space.sample()
        
        # Try to apply upward thrust to rotors (last 4 actions)
        # Action space is usually normalized -1 to 1 or similar depending on definition.
        # In XML we defined ctrlrange.
        # If we want to fly, we should bias the last 4 actions to be positive.
        
        # Humanoid has ~17 joints + 4 rotors = 21 actions?
        # Let's check action space size.
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: z_pos={info['z_pos']:.4f}, reward={reward:.4f}")
            
        if terminated or truncated:
            obs, info = env.reset()
            
    print("Test finished successfully.")

if __name__ == "__main__":
    main()
