import sys
import os
import numpy as np
import time

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quad_humanoid_env import QuadHumanoidEnv

def main():
    # Create env with visualization
    try:
        env = QuadHumanoidEnv(render_mode="human")
    except Exception as e:
        print(f"Could not initialize with render_mode='human': {e}")
        env = QuadHumanoidEnv(render_mode=None)

    obs, info = env.reset()
    print("Simulation started. Testing Superman pose with zero actions...")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Simulation loop - let it fall to see the initial pose
    for i in range(200):
        # Zero action (no control)
        action = np.zeros(env.action_space.shape)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: Height={info['z_pos']:.3f}m, "
                  f"Orient_err={info['orientation_error']:.3f}, "
                  f"Reward={reward:.2f}")
            
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
            break
            
        time.sleep(0.01)

    env.close()
    print("\nTest completed. The robot should have spawned face-down in mid-air.")

if __name__ == "__main__":
    main()
