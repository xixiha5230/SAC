import argparse
import os
from time import sleep
import gym
import glob
from Algorithm.SACIR import SACIR
from Envwrapper.UnityEnv import UnityWrapper

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument(
    "--env-name",
    default="venv_605_easy",
    help="Mujoco Gym environment (default: LunarLander-v2)",
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
    "--target_update_interval",
    type=int,
    default=1,
    metavar="N",
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=1000000,
    metavar="N",
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")
args = parser.parse_args()


env = UnityWrapper(args.env_name, worker_id=1, seed=args.seed)
# env = UnityWrapper(None, worker_id=123)
env.reset(seed=args.seed)
env.action_space.seed(args.seed)

agent = SACIR(env.observation_space, env.action_space, args)
list_of_files = glob.glob("result/{}/checkpoints/*".format(args.env_name))
# latest_file = max(list_of_files, key=os.path.getctime)
latest_file = "result/{}/checkpoints/10.0.ckpt".format(args.env_name)

print("Test on : ", latest_file)
agent.load_checkpoint(latest_file)

while True:
    done = False
    obs = env.reset(seed=args.seed)
    total_reward = 0
    step = 0
    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, done, _ = env.step(action)
        env.render()
        sleep(0.01)
        total_reward += reward
        step += 1
    print("Reward: ", total_reward, " Step: ", step)
