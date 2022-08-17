from time import sleep, time
import torch
from Network.ModelIR import StateNetwork, GaussianPolicyIR
from Envwrapper.UnityEnv import UnityWrapper

env_name = "venv_605_middle"
seed = 12345
device = "cuda" if torch.cuda.is_available else "cpu"


def load_checkpoint(obs_space, num_action, action_space, ckpt_path):
    state_net = StateNetwork(obs_space, 256, 64).to(device)
    policy = GaussianPolicyIR(64, num_action, 256, action_space).to(device)
    print("Loading models from {}".format(ckpt_path))
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        state_net.load_state_dict(checkpoint["state_net_state_dict"])
        policy.load_state_dict(checkpoint["policy_state_dict"])
        state_net.eval()
        policy.eval()
    return state_net, policy


def select_action(policy, state_net, state):
    state_tmp = [
        torch.FloatTensor(state[0]).to(device).unsqueeze(0),
        torch.FloatTensor(state[1]).to(device).unsqueeze(0),
    ]
    eval = True
    if eval:
        _, _, action = policy.sample(state_net(state_tmp))
    else:
        action, _, _ = policy.sample(state_net(state_tmp))
    return action.detach().cpu().numpy()[0]


env = UnityWrapper(None,  seed=seed)
# env = UnityWrapper("result/{}/env".format(env_name), worker_id=2, seed=seed)
env.reset(seed=seed)
env.action_space.seed(seed)

# list_of_files = glob.glob("result/{}/checkpoints/*".format(env_name))
# latest_file = max(list_of_files, key=os.path.getctime)
ckpt = "2022-08-16_11-16-R-6.0.ckpt"
latest_file = "result/{}/checkpoints/{}".format(env_name, ckpt)
print("Test on : ", latest_file)
state_net, policy_net = load_checkpoint(
    env.observation_space, env.action_size, env.action_space, latest_file)

TIMES = 100
times = TIMES
success_counter = 0
collision_counter = 0
while times > 0:
    done = False
    obs = env.reset(seed=seed)
    total_reward = 0
    step = 0
    while not done:
        action = select_action(policy_net, state_net, obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        sleep(0.01)
        total_reward += reward
        step += 1
    if total_reward > 0:
        success_counter += 1
    if total_reward < 0:
        collision_counter += 1
    times -= 1
    print("Reward: ", total_reward, " Step: ", step)
print("Sucess: ",  success_counter, " collision: ",
      collision_counter, " Total: ", TIMES)
