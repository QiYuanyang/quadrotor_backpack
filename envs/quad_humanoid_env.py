import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class QuadHumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, **kwargs):
        # Get the absolute path to the XML file
        xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "humanoid_quadrotor.xml")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64) # Placeholder shape, will be auto-calculated usually but MujocoEnv needs it? 
        # Actually MujocoEnv calculates it if we don't pass it, but we need to define _get_obs
        
        utils.EzPickle.__init__(self, **kwargs)
        
        # frame_skip=5 is standard for humanoid (dt=0.003 * 5 = 0.015s)
        MujocoEnv.__init__(
            self,
            model_path=xml_path,
            frame_skip=5,
            observation_space=None,
            default_camera_config={
                "trackbodyid": 1,
                "distance": 4.0,
                "lookat": np.array((0.0, 0.0, 2.0)),
                "elevation": -20.0,
            },
            **kwargs
        )
        
        # Ensure observation_space is set
        if not hasattr(self, "observation_space") or self.observation_space is None:
            obs = self._get_obs()
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)
        
        # Track previous action for smoothness reward
        self.previous_action = None

    def step(self, action):
        xy_position_before = self.data.body("torso").xpos[:2].copy()
        
        self.do_simulation(action, self.frame_skip)
        
        xy_position_after = self.data.body("torso").xpos[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        
        # Observations
        observation = self._get_obs()
        
        # Get current state
        z_pos = self.data.body("torso").xpos[2]
        z_vel = self.data.qvel[2]
        
        # Get torso orientation (quaternion)
        quat = self.data.body("torso").xquat  # [w, x, y, z]
        
        # Compute the z-axis of the body frame in world coordinates
        # For Superman pose (face down), z-axis should point down: [0, 0, -1]
        w, x, y, z = quat
        z_axis_world = np.array([
            2*(x*z + w*y),
            2*(y*z - w*x),
            1 - 2*(x**2 + y**2)
        ])
        
        # === DENSE REWARD SHAPING ===
        
        # 1. Base survival bonus (always positive if alive)
        reward = 5.0
        
        # 2. Height tracking (dense, continuous feedback)
        target_height = 2.0  # Lower target for easier task starting from ground
        height_error = abs(z_pos - target_height)
        height_reward = -2.0 * height_error
        
        # 3. Orientation tracking (dense, continuous feedback)
        orientation_error = np.linalg.norm(z_axis_world - np.array([0, 0, -1]))
        orientation_reward = -5.0 * orientation_error
        
        # 4. Velocity penalties (encourage stability)
        xy_velocity_penalty = -0.1 * (xy_velocity[0]**2 + xy_velocity[1]**2)
        z_velocity_penalty = -0.5 * z_vel**2
        
        # 5. Control cost (penalize large actions)
        ctrl_cost = -0.005 * np.square(action).sum()
        
        # 6. Action smoothness (encourage stable control)
        if self.previous_action is not None:
            smoothness_reward = -0.1 * np.sum((action - self.previous_action)**2)
        else:
            smoothness_reward = 0.0
        self.previous_action = action.copy()
        
        # 7. Thrust usage bonus (encourage learning to use rotors)
        thrust_actions = action[17:21]  # Last 4 actions are thrust
        thrust_bonus = 0.5 * np.mean(np.clip(thrust_actions, 0, 50)) / 50.0
        
        # === MILESTONE BONUSES (Intermediate goals) ===
        
        # Milestone 1: Stay airborne (above 2m)
        if z_pos > 2.0:
            reward += 10.0
        
        # Milestone 2: Good orientation (face down)
        if orientation_error < 0.3:
            reward += 5.0
        
        # Milestone 3: Near target height (within 1m)
        if abs(height_error) < 1.0:
            reward += 10.0
        
        # Milestone 4: Stable hover (close to target, low velocity)
        if abs(height_error) < 0.5 and abs(z_vel) < 0.2:
            reward += 20.0
        
        # Milestone 5: Perfect hover (very close, minimal drift)
        if abs(height_error) < 0.2 and abs(z_vel) < 0.1 and np.linalg.norm(xy_velocity) < 0.1:
            reward += 30.0
        
        # === TOTAL REWARD ===
        reward += (height_reward + orientation_reward + xy_velocity_penalty + 
                   z_velocity_penalty + ctrl_cost + smoothness_reward + thrust_bonus)
        
        # Termination conditions
        terminated = False
        if z_pos < 0.2:  # Hit ground
            terminated = True
            reward -= 10.0  # Smaller crash penalty
        if z_pos > 5.0:  # Too high
            terminated = True
        if abs(xy_position_after[0]) > 3.0 or abs(xy_position_after[1]) > 3.0:  # Drifted too far
            terminated = True

        info = {
            "z_pos": z_pos,
            "z_vel": z_vel,
            "x_velocity": xy_velocity[0],
            "y_velocity": xy_velocity[1],
            "height_reward": height_reward,
            "orientation_reward": orientation_reward,
            "orientation_error": orientation_error,
            "height_error": height_error,
            "smoothness_reward": smoothness_reward,
            "thrust_bonus": thrust_bonus,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        
        # Add sensor data if needed, but qpos/qvel usually covers it for RL
        # We might want to add the quadrotor specific sensors explicitly if they aren't in qpos/qvel
        # But qpos includes the root joint (free joint) which has pos and quat.
        
        # Let's include the center of mass inertia etc if standard humanoid does, 
        # but for now simple concatenation is enough.
        
        # Standard humanoid excludes the first 2 qpos (x, y) to make it translation invariant?
        # For takeoff/landing, Z is important. X,Y might be important if we want to hover in place.
        # Let's keep everything for now.
        
        return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -0.05
        noise_high = 0.05

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        
        # Superman pose: spawn close to ground, horizontal (face down, lying)
        # qpos[0:3] = x, y, z position
        # qpos[3:7] = quaternion (w, x, y, z) for orientation
        
        # Set position: 0.5 meters up (close to ground, lying position)
        qpos[0] = 0.0  # x
        qpos[1] = 0.0  # y
        qpos[2] = 0.5  # z height - just above ground
        
        # Set orientation: Pitch down 90 degrees (face ground, lying horizontally)
        # Quaternion for 90 degree pitch (rotation around Y-axis)
        angle = np.pi / 2  # 90 degrees
        qpos[3] = np.cos(angle / 2)  # w
        qpos[4] = 0.0                 # x
        qpos[5] = np.sin(angle / 2)  # y (pitch axis)
        qpos[6] = 0.0                 # z
        
        # Set initial velocity to zero
        qvel[:6] = 0.0
        
        # Reset action tracking
        self.previous_action = None
        
        self.set_state(qpos, qvel)
        return self._get_obs()
