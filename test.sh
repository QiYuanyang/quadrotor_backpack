#!/bin/bash
# Quick test script for humanoid quadrotor environment

cd "$(dirname "$0")"

echo "Testing Humanoid Quadrotor Environment..."
python scripts/test_env.py --num_envs 4 --headless
