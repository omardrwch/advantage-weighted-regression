import argparse
import gym
import gym.spaces
import nets
from rlberry.envs import gym_make
from rlberry.manager import AgentManager
from awr import AWRAgent


parser = argparse.ArgumentParser()


# ------------------
# Parameters
# ------------------
parser.add_argument("--env_id", type=str, default="CartPole-v0", help="ID for gym.make")
parser.add_argument("--num_timesteps", type=int, default=50_000, help="Number of env timesteps for training.")
parser.add_argument("--n_fit", type=int, default=4, help="Number of agent instances to train.")
parser.add_argument("--seed", type=int, default=24216574, help="Seed for AgentManager.")


def value_net_constructor(env):
    return nets.MLP(
        input_dim=env.observation_space.shape[0],
        output_dim=1,
        hidden_sizes=(128, 64),
    )


def policy_net_constructor(env):
    if isinstance(env.action_space, gym.spaces.Box):
        output_dim = env.action_space.shape[0]
    else:
        output_dim = env.action_space.n
    return nets.MLP(
        input_dim=env.observation_space.shape[0],
        output_dim=output_dim,
        hidden_sizes=(128, 64),
    )


if __name__ == '__main__':

    args = parser.parse_args()

    env = (gym_make, dict(id=args.env_id))

    params = dict(
        value_net_constructor=value_net_constructor,
        policy_net_constructor=policy_net_constructor,
    )

    fit_kwargs = dict(
        fit_budget=args.num_timesteps,
    )

    manager = AgentManager(
        AWRAgent,
        train_env=env,
        fit_kwargs=fit_kwargs,
        init_kwargs=params,
        n_fit=args.n_fit,
        output_dir=f'temp/{args.env_id}',
        parallelization='process',
        seed=args.seed,
        enable_tensorboard=True,
    )
    print(f"\nOutput dir = {manager.output_dir}\n")
    manager.fit()
    manager.save()
