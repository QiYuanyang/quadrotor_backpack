#!/usr/bin/env python3
# Copyright (c) 2024
# Training script for humanoid quadrotor using RSL-RL

"""Train humanoid quadrotor with RSL-RL PPO."""

import argparse
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import QuadHumanoidEnv, QuadHumanoidEnvCfg


def main():
    """Train with RSL-RL PPO."""
    parser = argparse.ArgumentParser(description="Train humanoid quadrotor")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Max training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Humanoid Quadrotor with RSL-RL PPO")
    print("=" * 60)
    print(f"Num environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Headless: {args.headless}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    
    # Create environment configuration
    env_cfg = QuadHumanoidEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Create environment
    print("\nCreating environment...")
    env = QuadHumanoidEnv(cfg=env_cfg)
    print("✓ Environment created successfully")
    
    # Import RSL-RL (check if available)
    try:
        from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
        from rsl_rl.runners import OnPolicyRunner
        print("✓ RSL-RL imported successfully")
    except ImportError:
        print("✗ RSL-RL not found. Please install it:")
        print("  pip install rsl-rl")
        return
    
    # Wrap environment
    env = RslRlVecEnvWrapper(env)
    
    # PPO configuration
    runner_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        save_interval=100,
        experiment_name="quad_humanoid_ppo",
        empirical_normalization=False,
        policy=RslRlOnPolicyRunnerCfg.PolicyCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
            activation="elu",
        ),
        algorithm=RslRlOnPolicyRunnerCfg.AlgorithmCfg(
            value_loss_coef=0.5,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=10,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
    
    # Create runner and train
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\nStarting training...")
    print(f"Logs will be saved to: {log_dir}")
    
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=env_cfg.sim.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("\n✓ Training completed!")
    env.close()


if __name__ == "__main__":
    main()
