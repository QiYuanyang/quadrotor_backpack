#!/bin/bash
# Script to clean up old training logs and checkpoints

echo "Current disk usage:"
du -sh logs/ checkpoints/ 2>/dev/null

echo ""
echo "Available cleanup options:"
echo "1. Keep only the latest PPO run in logs/"
echo "2. Delete all logs (CAREFUL!)"
echo "3. Delete all checkpoints except best_model"
echo "4. Delete everything except latest run"
echo "5. Cancel"

read -p "Choose option (1-5): " choice

case $choice in
  1)
    echo "Keeping only the latest PPO run..."
    cd logs/
    latest=$(ls -t | head -1)
    echo "Keeping: $latest"
    ls -t | tail -n +2 | xargs rm -rf
    cd ..
    echo "Done!"
    ;;
  2)
    read -p "Are you sure you want to delete ALL logs? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      rm -rf logs/*
      echo "All logs deleted!"
    fi
    ;;
  3)
    echo "Keeping only best_model.zip..."
    cd checkpoints/
    find . -type f ! -name 'best_model.zip' ! -name 'vec_normalize.pkl' -delete
    cd ..
    echo "Done!"
    ;;
  4)
    echo "Cleaning everything except latest run..."
    cd logs/
    latest=$(ls -t | head -1)
    ls -t | tail -n +2 | xargs rm -rf
    cd ../checkpoints/
    find . -type f -mtime +1 ! -name 'best_model.zip' -delete
    cd ..
    echo "Done!"
    ;;
  5)
    echo "Cancelled"
    ;;
  *)
    echo "Invalid option"
    ;;
esac

echo ""
echo "Current disk usage after cleanup:"
du -sh logs/ checkpoints/ 2>/dev/null
