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

    def step(self, action):
        xy_position_before = self.data.body("torso").xpos[:2].copy()
        
        self.do_simulation(action, self.frame_skip)
        
        xy_position_after = self.data.body("torso").xpos[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        
        # Observations
        observation = self._get_obs()
        
        # Rewards
        # 1. Height reward (Takeoff)
        z_pos = self.data.body("torso").xpos[2]
        target_height = 2.0
        height_reward = -10.0 * (z_pos - target_height) ** 2
        if z_pos > 0.5: # Bonus for being off ground
            height_reward += 10.0
            
        # 2. Stability reward (Upright)
        # Get torso quaternion
        quat = self.data.body("torso").xquat
        # Upright is [1, 0, 0, 0] (w, x, y, z) roughly
        # Simple check: z-axis of torso frame should be close to global z
        # But let's just use a small penalty for angular velocity and deviation from upright
        
        # 3. Control cost
        ctrl_cost = 0.1 * np.square(action).sum()
        
        reward = height_reward - ctrl_cost + 10.0 # Survival bonus
        
        # Termination
        terminated = False
        if z_pos < 0.3: # Fell down
            # terminated = True # Don't terminate immediately for landing training?
            pass
        if z_pos > 5.0: # Too high
            terminated = True

        info = {
            "z_pos": z_pos,
            "x_velocity": xy_velocity[0],
            "y_velocity": xy_velocity[1],
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
        noise_low = -0.1
        noise_high = 0.1

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        
        # Ensure it starts on the ground (z height)
        # qpos[2] is usually z height of root
        qpos[2] = 1.4 # Reset to standing height
        
        self.set_state(qpos, qvel)
        return self._get_obs()
