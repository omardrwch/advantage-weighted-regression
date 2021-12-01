import numpy as np
import utils
from gym.spaces import Box, Discrete
from rlberry.envs import Wrapper


class AWREnvWrapper(Wrapper):
    """
    Wraps the observation and action spaces of environments for AWRAgent.
    """
    def __init__(self, env):
        Wrapper.__init__(self, env)
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        assert isinstance(obs_space, Box) or isinstance(obs_space, Discrete)
        assert isinstance(action_space, Box) or isinstance(action_space, Discrete)
        self.make_one_hot_obs = isinstance(obs_space, Discrete)
        self.continuous_actions = isinstance(action_space, Box)
        
        #
        # Set new spaces
        #
        if self.make_one_hot_obs:
            self.observation_space = Box(low=0.0, high=1.0, shape=(obs_space.n,), dtype=np.float32)
        else:
            self.observation_space = obs_space

        # If actions continuous, normalize actions to [-1, 1] range
        if self.continuous_actions:
            self.action_space = Box(low=-1.0, high=1.0, shape=action_space.shape, dtype=np.float32)

    def process_obs(self, obs):
        if self.make_one_hot_obs:
            one_hot_obs = np.zeros(self.env.observation_space.n, dtype=np.float32)
            one_hot_obs[obs] = 1.0
            return one_hot_obs
        else:
            return obs
    
    def process_action(self, action):
        if self.continuous_actions:
            # input action is in [-1, 1], map to [low, high]
            return utils.unscale_action(self.env.action_space, action)
        else:
            return action

    def reset(self):
        obs = self.env.reset()
        return self.process_obs(obs)

    def step(self, action):
        action = self.process_action(action)
        observation, reward, done, info = self.env.step(action)
        observation = self.process_obs(observation)
        return observation, reward, done, info
