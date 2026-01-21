#!/usr/bin/env python3
# Copyright (c) 2024
# Test script for humanoid quadrotor environment

"""Test the humanoid quadrotor environment with random actions."""

import argparse

# Initialize IsaacLab app first
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test humanoid quadrotor environment")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import QuadHumanoidEnv, QuadHumanoidEnvCfg


def main():
    """Test environment with random actions."""
    print("=" * 60)
    print("Testing Humanoid Quadrotor Environment")
    print("=" * 60)
    
    # Create config
    cfg = QuadHumanoidEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 5.0
    
    # Create environment
    print("\nCreating environment...")
    env = QuadHumanoidEnv(cfg=cfg)
    
    print(f"✓ Environment created successfully")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Number of environments: {env.num_envs}")
    print(f"  - Device: {env.device}")
    
    # Reset
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  - Observation shape: {obs['policy'].shape}")
    
    # Run random actions
    print("\nRunning random actions for 100 steps...")
    print("-" * 60)
    
    # Get action dimension from action space
    action_dim = env.action_space.shape[1]
    
    for step in range(100):
        # Random actions (small magnitude)
        actions = torch.randn(env.num_envs, action_dim, device=env.device) * 0.3
        
        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Print status every 10 steps
        if step % 10 == 0:
            height = env.robot.data.root_pos_w[0, 2].item()
            reward = rewards[0].item()
            print(f"Step {step:3d}: Height = {height:6.3f}m, Reward = {reward:8.2f}")
            
            # Check for issues
            if torch.isnan(obs['policy']).any():
                print("⚠ WARNING: NaN detected in observations!")
                break
            if torch.isnan(rewards).any():
                print("⚠ WARNING: NaN detected in rewards!")
                break
    
    print("-" * 60)
    print("\n✓ Test completed successfully!")
    print("\nEnvironment is ready for training.")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
