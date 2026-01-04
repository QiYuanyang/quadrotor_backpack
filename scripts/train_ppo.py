import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quad_humanoid_env import QuadHumanoidEnv

def make_env():
    """Create and wrap the environment."""
    def _init():
        env = QuadHumanoidEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init

def main():
    # Configuration
    num_envs = 16  # Run 16 parallel environments for faster training
    total_timesteps = 5_000_000  # 5 million steps
    n_steps = 2048  # Number of steps per update (per environment)
    batch_size = 256  # Larger batch size for better GPU utilization
    n_epochs = 10
    learning_rate = 3e-4
    
    # Create log directories
    log_dir = "./logs/"
    checkpoint_dir = "./checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Creating {num_envs} parallel environments...")
    # Create parallel vectorized environments using SubprocVecEnv for true parallelism
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    print("Creating PPO model...")
    # Create PPO model
    # With larger batch sizes and parallel envs, GPU becomes beneficial
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Parallel environments: {num_envs}")
    print(f"Batch size: {batch_size}")
    print(f"Total rollout steps per update: {n_steps * num_envs}")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Add exploration bonus
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )
    
    print(f"\nTraining will be ~{num_envs}x faster with parallel environments!")
    print(f"Expected wall-clock time: ~{total_timesteps / (num_envs * 5000):.1f} hours\n")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_envs,  # Adjust frequency for parallel envs
        save_path=checkpoint_dir,
        name_prefix="quad_humanoid_ppo"
    )
    
    eval_env = SubprocVecEnv([make_env() for _ in range(4)])  # Use 4 envs for evaluation
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=10000 // num_envs,  # Adjust frequency for parallel envs
        n_eval_episodes=5,
        deterministic=True
    )
    
    print("Starting training...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Tensorboard logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("\nTo monitor training, run:")
    print(f"  tensorboard --logdir {log_dir}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        print("\nTraining completed!")
        print("Saving final model...")
        model.save(os.path.join(checkpoint_dir, "quad_humanoid_ppo_final"))
        env.save(os.path.join(checkpoint_dir, "vec_normalize.pkl"))
        print("Model saved successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model...")
        model.save(os.path.join(checkpoint_dir, "quad_humanoid_ppo_interrupted"))
        env.save(os.path.join(checkpoint_dir, "vec_normalize_interrupted.pkl"))
        print("Model saved!")
    
    env.close()

if __name__ == "__main__":
    main()
