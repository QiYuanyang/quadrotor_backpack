# Humanoid Quadrotor Backpack Simulation

This repository contains a MuJoCo simulation of a humanoid robot equipped with a quadrotor "backpack". The goal is to study the dynamics and control of a humanoid that can take off, fly, and land using the attached propulsion system.

## Project Structure

- **`assets/`**: Contains the MuJoCo XML models.
  - `humanoid_quadrotor.xml`: The main model file combining the standard humanoid with a rigid quadrotor attachment.
- **`envs/`**: Contains the Gymnasium environment wrapper.
  - `quad_humanoid_env.py`: Defines the `QuadHumanoidEnv` class, inheriting from `MujocoEnv`.
- **`scripts/`**: Python scripts for testing and training.
  - `fly_test.py`: A simple script to run the simulation with a dummy policy (zero actions) or simple thrust commands to verify physics.
  - `test_env.py`: A script to verify the Gymnasium environment API and observation/action spaces.
- **`requirements.txt`**: List of Python dependencies.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/QiYuanyang/quadrotor_backpack.git
    cd quadrotor_backpack
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n alice python=3.10
    conda activate alice
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # If you plan to use Stable Baselines3 for training:
    pip install stable-baselines3 torch
    ```

## Usage

### Running the Simulation (Physics Verification)

To visualize the robot and test the physics (currently runs a dummy policy where the robot falls):

```bash
python scripts/fly_test.py
```

### Testing the Environment

To check the Gymnasium environment setup:

```bash
python scripts/test_env.py
```

## Simulation Details

- **Model:** The quadrotor is attached rigidly to the humanoid's torso, rotated 90 degrees (vertical configuration).
- **Actuators:**
  - 17 Humanoid Joints (Hips, Knees, Ankles, Shoulders, Elbows).
  - 4 Rotor Thrusts (Modeled as motors applying force along the local Z-axis).
- **Sensors:**
  - Standard Humanoid sensors (Joint positions/velocities).
  - Quadrotor IMU (Accelerometer, Gyro).
  - Global Frame Position/Velocity/Orientation.
