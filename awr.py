import gym
import gym.spaces
import torch
import numpy as np
import utils
from awr_env_wrapper import AWREnvWrapper
from replay_buffer import ReplayBuffer
from rlberry import types
from rlberry.agents import Agent
from typing import Callable, Optional


class AWRAgent(Agent):
    """
    Implementation of Advantage-Weighted Regression: Simple and Scalable Off-Policy RL
    (https://arxiv.org/pdf/1910.00177.pdf).

    Parameters
    ----------
    value_net_constructor : callable
        Constructor for V network. Takes an env as argument: v_net = value_net_constructor(env)
    policy_net_constructor : callable
        Constructor for policy network. Takes an env as argument: pi_net = policy_net_constructor(env)
    gamma : float
        Discount factor.
    lambda_ : float
       Lambda returns parameter.
    beta : float
        Temperatrue, advantage weighting parameter.
    action_noise: float
        Standard deviation of Gaussian policy (for continuous action spaces).
    random_exploration_eps: float
        Probability of taking a random action.
    max_weight : float
        Maximum value of weights.
    batch_size : int
        Batch size.
    max_replay_size : int
        Maximum number of transitions in the replay buffer.
    max_episode_length : int
        Maximum length of an episode. If None, episodes will only end if `done = True`
        is returned by env.step().
    learning_starts : int
        How many steps of the model to collect transitions for before learning starts.
    update_interval : int
        Update the model every ``update_interval`` steps.
    value_steps_per_update : int
        Number of gradient steps in value net per update.
    policy_steps_per_update : int
        Number of gradient steps in policy net per update.
    value_steps_per_update : float
        Learning rate for value optimization.
    policy_learning_rate : float
        Learning rate for policy optimization.
    normalize_advantages : bool
        If true, normalize advantages.
    max_episode_length : int
        Maximum length of an episode. If None, episodes will only end if `done = True`
        is returned by env.step().
    eval_interval : int
        Interval (in number of transitions) between agent evaluations in fit().
        If None, never evaluate.
    """

    name = "AWR"

    ADV_EPS = 1e-5
    
    def __init__(
        self,
        env: types.Env,
        value_net_constructor: Callable[[gym.Env], torch.nn.Module],
        policy_net_constructor: Callable[[gym.Env], torch.nn.Module],
        gamma: float = 0.99,
        lambda_: float = 0.95,
        beta: float = 1.0,
        action_noise: float = 0.4,
        random_exploration_eps: float = 0.1,
        max_weight: float = 20.0,
        batch_size: int = 256,
        max_replay_size: int = 50_000,
        learning_starts: int = 2048,
        update_interval: int = 2048,
        value_steps_per_update: int = 200,
        policy_steps_per_update: int = 1000,
        value_learning_rate: float = 1e-2,
        policy_learning_rate: float = 5e-5,
        normalize_advantages: bool = True,
        max_episode_length: Optional[int] = None,
        eval_interval: Optional[int] = 500,
        device: str="cpu",    # try also "cuda:best"
        **kwargs):
        Agent.__init__(self, env, **kwargs)

        # Wrap environment
        # * Discrete observations -> one-hot encoding
        # * Continuous actions -> normalized to [-1, 1]
        self.env = AWREnvWrapper(self.env)
        self.eval_env = AWREnvWrapper(self.eval_env)
        env = self.env

        #
        # Function to process outputs of policy_net
        #
        self.continuous_actions = isinstance(env.action_space, gym.spaces.Box)

        #
        # Params
        #
        self._gamma = gamma
        self._lambda = lambda_
        self._beta = beta
        self._action_noise = action_noise
        self._random_exploration_eps = random_exploration_eps
        self._max_weight = max_weight
        self._batch_size = batch_size
        self._learning_starts = learning_starts
        self._update_interval = update_interval
        self._value_steps_per_update = value_steps_per_update
        self._policy_steps_per_update = policy_steps_per_update
        self._normalize_advantages = normalize_advantages
        self._max_episode_length = max_episode_length or np.inf
        self._eval_interval = eval_interval

        #
        # Replay
        #
        self.replay = ReplayBuffer(1, 1, max_replay_size, self.rng)

        # Torch device
        self.device = utils.choose_device(device)

        #
        # Networks/optimizers/loss
        #
        self.value_net = value_net_constructor(env).to(self.device)
        self.policy_net = policy_net_constructor(env).to(self.device)
        self.value_optimizer = torch.optim.Adam(params=self.value_net.parameters(), lr=value_learning_rate)
        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=policy_learning_rate)
        self.value_loss_fn = torch.nn.MSELoss()

        #
        # Counters
        #
        self._total_timesteps = 0
        self._total_episodes = 0
        self._total_value_updates = 0
        self._total_policy_updates = 0
        self._timesteps_since_last_update = 0

    @property
    def total_timesteps(self):
        return self._total_timesteps

    def update(self):
        # Preprocessing
        replay_data = self.replay.data
        buffer_size = len(replay_data.rewards)
        all_observations =  torch.tensor(np.array(replay_data.observations, dtype=np.float32)).to(self.device)
        all_actions = np.array(replay_data.actions)
        all_rewards = np.array(replay_data.rewards, dtype=np.float32)
        all_discounts = np.array(replay_data.discounts, dtype=np.float32)

        all_values = self.value_net(all_observations).detach().squeeze(1).cpu().numpy()
    
        all_lambda_returns = utils.lambda_returns_no_batch(
            all_rewards[:-1],
            all_discounts[:-1],
            all_values[1:],    # values at t+1
            np.array(self._lambda, dtype=np.float32)
        )

        #
        # Update critic
        #
        for _ in range(self._value_steps_per_update):
            # increment counter
            self._total_value_updates += 1

            # sample a batch
            batch_indices = self.rng.choice(buffer_size - 1, size=(self._batch_size,))
            batch_observations = all_observations[batch_indices]
            batch_lambda_returns = all_lambda_returns[batch_indices]

            # value loss
            values = self.value_net(batch_observations).squeeze(dim=-1)
            targets = torch.tensor(batch_lambda_returns).to(self.device)
            assert values.shape == targets.shape
            value_loss = self.value_loss_fn(values, targets)

            # update
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # log info
            if self.writer:
                self.writer.add_scalar('losses/value_loss', value_loss.item(), self.total_timesteps)
                self.writer.add_scalar('counters/total_value_updates', self._total_value_updates, self.total_timesteps)

        #
        # Delete old variables, recompute with updated net
        #
        del all_values
        del all_lambda_returns        
        all_values = self.value_net(all_observations).detach().squeeze(1).cpu().numpy()

        all_lambda_returns = utils.lambda_returns_no_batch(
            all_rewards[:-1],
            all_discounts[:-1],
            all_values[1:],
            np.array(self._lambda, dtype=np.float32)
        )
        all_advantages = all_lambda_returns - all_values[:-1]

        # normalize advantages
        if self._normalize_advantages:
            adv_mean = all_advantages.mean()
            adv_std = all_advantages.std()
            all_advantages = (all_advantages - adv_mean) / (self.ADV_EPS + adv_std)

        #
        # Update actor
        #
        for _ in range(self._policy_steps_per_update):
           # increment counter
            self._total_policy_updates += 1

            # sample a batch
            batch_indices = self.rng.choice(buffer_size - 1, size=(self._batch_size,))
            batch_observations = all_observations[batch_indices]
            batch_actions = torch.tensor(all_actions[batch_indices]).to(self.device)

            # compute advantages and weights
            batch_advantages = all_advantages[batch_indices]
            batch_advantages = torch.tensor(batch_advantages).to(self.device)
            weights = torch.exp(batch_advantages / self._beta)
            weights_clipped = torch.clamp(weights, min=0.0, max=self._max_weight)

            # logprobs
            policy_net_out = self.policy_net(batch_observations)

            if self.continuous_actions:
                # apply tanh to map to [-1, 1]
                mean_actions = torch.tanh(policy_net_out)
                log_probs = -0.5 * torch.square(
                    mean_actions - batch_actions
                ).sum(dim=-1) / (self._action_noise ** 2.0)
            else:
                log_probs = torch.log_softmax(policy_net_out, dim=-1)
                log_probs = torch.gather(log_probs, dim=-1, index=batch_actions[:, None]).squeeze(dim=-1)

            # policy loss
            assert log_probs.shape == weights_clipped.shape
            policy_loss = -(log_probs * weights_clipped).mean()

            # update
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
            # log info
            if self.writer:
                weights_np = weights.detach().numpy()
                self.writer.add_scalar('stats/weights/weights_mean', weights_np.mean(), self.total_timesteps)
                self.writer.add_scalar('stats/weights/weights_std', weights_np.std(), self.total_timesteps)
                self.writer.add_scalar('stats/weights/weights_min', weights_np.min(), self.total_timesteps)
                self.writer.add_scalar('stats/weights/weights_max', weights_np.max(), self.total_timesteps)
                self.writer.add_scalar('losses/policy_loss', policy_loss.item(), self.total_timesteps)
                self.writer.add_scalar('counters/total_policy_updates', self._total_policy_updates, self.total_timesteps)

    def compute_state(self, next_obs, state=None, action=None):
        """
        state = embedding of [obs(0), action(0), ..., obs(t-1), action(t-1), obs(t)]
        """
        state = next_obs
        return state

    def must_update(self, is_end_of_episode):
        """Returns true if the model must be updated in the current timestep,
        and the number of gradient steps to take"""
        del is_end_of_episode
        total_timesteps = self._total_timesteps
        if total_timesteps < self._learning_starts:
            return False
        run_update = total_timesteps % self._update_interval == 0
        return run_update

    def fit(self, budget: int, **kwargs):
        del kwargs
        timesteps_counter = 0
        episode_rewards = 0.0
        episode_timesteps = 0
        observation = self.env.reset()
        state = self.compute_state(observation)
        while timesteps_counter < budget:
            if self.total_timesteps < self._learning_starts:
                action = self.env.action_space.sample()
            else:
                self._timesteps_since_last_update += 1
                action = self.policy(state, evaluation=False)
            next_obs, reward, done, _ = self.env.step(action)

            # update state
            state = self.compute_state(next_obs, state, action)

            # check max episode length
            done = done and (episode_timesteps < self._max_episode_length)

            # store data
            episode_rewards += reward
            self.replay.append(
                obs=observation, action=action, reward=reward, discount=self._gamma * (1.0 - done))

            # counters and next obs
            self._total_timesteps += 1
            timesteps_counter += 1
            episode_timesteps += 1
            observation = next_obs

            # update
            run_update = self.must_update(done)
            if run_update:
                self.update()

            # eval
            total_timesteps = self._total_timesteps
            if self._eval_interval is not None and total_timesteps % self._eval_interval == 0:
                eval_rewards = self.eval(
                    eval_horizon=self._max_episode_length,
                    gamma=1.0)
                eval_rewards_deterministic = self.eval(
                    eval_horizon=self._max_episode_length,
                    gamma=1.0,
                    deterministic=True)
                if self.writer:
                    buffer_size = len(self.replay)
                    self.writer.add_scalar('rewards/eval_rewards', eval_rewards, total_timesteps)
                    self.writer.add_scalar('rewards/eval_rewards_deterministic', eval_rewards_deterministic, total_timesteps)
                    self.writer.add_scalar('counters/buffer_size', buffer_size, total_timesteps)


            # check if episode ended
            if done:
                self._total_episodes += 1
                if self.writer:
                    self.writer.add_scalar('rewards/episode_rewards', episode_rewards, total_timesteps)
                    self.writer.add_scalar('counters/total_episodes', self._total_episodes, total_timesteps)
                episode_rewards = 0.0
                episode_timesteps = 0
                observation = self.env.reset()


    def policy(self, state, evaluation=False, deterministic=False):
        """
        state = embedding of [obs(0), action(0), ..., obs(t-1), action(t-1), obs(t)]
        """
        # random exploration
        epsilon = self._random_exploration_eps
        if (not evaluation) and self.rng.uniform() < epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_net_out = self.policy_net(tensor_state)[0, :]

        if self.continuous_actions:
            # apply tanh to map to [-1, 1]
            action = torch.tanh(policy_net_out)
            if not evaluation:
                noise = torch.zeros_like(action).normal_(0, self._action_noise).to(self.device)
                action = torch.clamp(action + noise, min=-1.0, max=1.0)
            action = action.numpy()
        else:
            probs = torch.softmax(policy_net_out, dim=-1).numpy()
            if not deterministic:
                action = self.rng.choice(self.env.action_space.n, p=probs)
            else:
                action = np.argmax(probs)
        return action

    def eval(self, eval_horizon=10 ** 5, n_simimulations=5, gamma=1.0, deterministic=False, **kwargs):
        del kwargs  # unused
        episode_rewards = np.zeros(n_simimulations)
        for sim in range(n_simimulations):
            observation = self.eval_env.reset()
            state = self.compute_state(observation)
            tt = 0
            while tt < eval_horizon:
                action = self.policy(state, evaluation=True, deterministic=deterministic)
                next_obs, reward, done, _ = self.eval_env.step(action)
                state = self.compute_state(next_obs, state, action)
                episode_rewards[sim] += reward * np.power(gamma, tt)
                tt += 1
                if done:
                    break
        return episode_rewards.mean()
