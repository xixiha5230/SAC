import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from Algorithm.SACIR import SACIR
from tensorboardX import SummaryWriter
from Memory.ReplayMemoryIR import ReplayMemoryIR
from Envwrapper.UnityEnv import UnityWrapper

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument(
    "--env-name",
    default="venv_605_easy",
    help="Mujoco Gym environment (default: HalfCheetah-v2)",
)
parser.add_argument(
    "--policy",
    default="Gaussian",
    help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
)
parser.add_argument(
    "--eval",
    type=bool,
    default=True,
    help="Evaluates a policy a policy every 10 episode (default: True)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    metavar="G",
    help="target smoothing coefficient(τ) (default: 0.005)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0003,
    metavar="G",
    help="learning rate (default: 0.0003)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.1,
    metavar="G",
    help="Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)",
)
parser.add_argument(
    "--automatic_entropy_tuning",
    type=bool,
    default=False,
    metavar="G",
    help="Automaically adjust α (default: False)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123456,
    metavar="N",
    help="random seed (default: 123456)",
)
parser.add_argument(
    "--batch_size", type=int, default=256, metavar="N", help="batch size (default: 256)"
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1000001,
    metavar="N",
    help="maximum number of steps (default: 1000000)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    metavar="N",
    help="hidden size (default: 256)",
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=1,
    metavar="N",
    help="model updates per simulator step (default: 1)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=10000,
    metavar="N",
    help="Steps sampling random actions (default: 10000)",
)
parser.add_argument(
    "--eval_per_steps",
    type=int,
    default=50,
    metavar="N",
    help="Steps eval reward(default: 100)",
)
parser.add_argument(
    "--eval_times",
    type=int,
    default=5,
    metavar="N",
    help="episode per eval(default: 5)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    metavar="N",
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=200000,
    metavar="N",
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument(
    "--cuda", type=bool, default=True, help="run on CUDA (default: False)"
)
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name, continuous=True)
env = UnityWrapper(args.env_name, seed=args.seed)
env.reset(seed=args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(torch.get_num_threads())

# Agent
agent = SACIR(env.observation_space, env.action_space, args)

# Tesnorboard
writer = SummaryWriter(
    "result/{}/{}_SAC_{}{}".format(
        args.env_name,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.policy,
        "_autotune" if args.automatic_entropy_tuning else "",
    )
)

# Memory
memory = ReplayMemoryIR(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
latest_avg_reward = float("-inf")

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (
                    critic_1_loss,
                    critic_2_loss,
                    policy_loss,
                    ent_loss,
                    alpha,
                ) = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                writer.add_scalar("entropy_temprature/alpha", alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(
            state, action, reward, next_state, mask
        )  # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar("reward/train", episode_reward, i_episode)
    print(
        "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
        )
    )

    if i_episode % args.eval_per_steps == 0 and args.eval is True:
        avg_reward = 0.0
        episodes = args.eval_times
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar("avg_reward/test", avg_reward, i_episode)

        print("----------------------------------------")
        print(
            "Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2))
        )
        print("----------------------------------------")

        if latest_avg_reward <= avg_reward or i_episode % 100 == 0:
            agent.save_checkpoint(args.env_name, str(round(avg_reward, 2)) + ".ckpt")
            if latest_avg_reward < avg_reward:
                latest_avg_reward = avg_reward

env.close()
