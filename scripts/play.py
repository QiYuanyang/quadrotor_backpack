"""
Play script to visualize a trained policy.
Loads a checkpoint and renders the humanoid-quadrotor attempting to hover.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from stable_baselines3 import PPO
from envs.quad_humanoid_env import QuadHumanoidEnv

def play(model_path, episodes=5, render=True):
    """
    Load and visualize a trained policy.
    
    Args:
        model_path: Path to the trained model checkpoint
        episodes: Number of episodes to play
        render: Whether to render the environment
    """
    print(f"Loading model from: {model_path}")
    
    # Create environment
    env = QuadHumanoidEnv(render_mode="human" if render else None)
    
    # Load trained model
    model = PPO.load(model_path, env=env)
    
    print(f"\nPlaying {episodes} episodes...\n")
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        print(f"Episode {episode + 1}/{episodes}")
        
        while not done:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            # Print thrust forces every 10 steps
            if render and step_count % 10 == 0:
                if 'force1' in info:
                    print(f"  Step {step_count}: Forces = "
                          f"R1:{info['force1']:6.1f}N  "
                          f"R2:{info['force2']:6.1f}N  "
                          f"R3:{info['force3']:6.1f}N  "
                          f"R4:{info['force4']:6.1f}N  "
                          f"| Total:{sum([info[f'force{i}'] for i in range(1,5)]):7.1f}N")
            
            if render:
                env.render()
        
        print(f"  Steps: {step_count}, Total Reward: {episode_reward:.2f}")
        if 'height' in info:
            print(f"  Final Height: {info['height']:.2f}m")
        if 'orientation_error' in info:
            print(f"  Final Orientation Error: {info['orientation_error']:.3f}")
        print()
    
    env.close()
    print("Playback complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Play a trained PPO policy")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.zip",
        help="Path to model checkpoint (default: checkpoints/best_model.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (useful for headless systems)"
    )
    
    args = parser.parse_args()
    
    play(args.model, args.episodes, render=not args.no_render)
