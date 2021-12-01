import numpy as np
import collections


ReplayData = collections.namedtuple(
    'ReplayData',
    ['observations',
     'actions',
     'rewards',
     'discounts'])


class ReplayBuffer:
    def __init__(self, batch_size, chunk_size, max_replay_size, rng):
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._rng = rng
        self._max_replay_size = max_replay_size
        self._observations = []
        self._actions = []
        self._rewards = []
        self._discounts = []

    @property
    def data(self):
        observations = self._observations
        actions = self._actions
        rewards = self._rewards
        discounts = self._discounts
        return ReplayData(
            observations,
            actions,
            rewards,
            discounts
        )

    def _sample_one_trajectory(self):
        current_size = len(self)
        start_index = self._rng.choice(current_size - self._chunk_size)
        end_index = start_index + self._chunk_size
        observations = np.array(self._observations[start_index:end_index])
        actions = np.array(self._actions[start_index:end_index])
        rewards = np.array(self._rewards[start_index:end_index])
        discounts = np.array(self._discounts[start_index:end_index])
        return observations, actions, rewards, discounts
    
    def __len__(self):
        return len(self._rewards)

    def append(self, obs, action, reward, discount):
        self._observations.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._discounts.append(discount)
        if len(self) > self._max_replay_size:
            self._observations.pop(0)
            self._actions.pop(0)
            self._rewards.pop(0)
            self._discounts.pop(0)

    def sample(self):
        if len(self) <= self._chunk_size:
            return None
        batch = dict()
        obs_trajectories = []
        action_trajectories = []
        reward_trajectories = []
        discount_trajectories = []
        for _ in range(self._batch_size):
            observations, actions, rewards, discounts = self._sample_one_trajectory()
            obs_trajectories.append(observations)
            action_trajectories.append(actions)
            reward_trajectories.append(rewards)
            discount_trajectories.append(discounts)
        batch['observations'] = np.array(obs_trajectories)
        batch['actions'] = np.array(action_trajectories)
        batch['rewards'] = np.array(reward_trajectories)
        batch['discounts'] = np.array(discount_trajectories)
        return batch