from ctypes import Union
import torch
import torch.nn as nn
from typing import Tuple, Union
from Network.Model import GaussianPolicy, weights_init_


def conv1d_output_size(
    length: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    from math import floor

    l_out = floor(
        ((length + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )
    return l_out


def conv2d_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a convolution layer.
    kernel_size, stride, padding and dilation correspond to the inputs of the
    torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    :param h_w: The height and width of the input.
    :param kernel_size: The size of the kernel of the convolution (can be an int or a
    tuple [width, height])
    :param stride: The stride of the convolution
    :param padding: The padding of the convolution
    :param dilation: The dilation of the convolution
    """
    from math import floor

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(
        ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return x


class Conv2d(nn.Module):
    def __init__(self, shape, hidden_dim, out_dim):
        super(Conv2d, self).__init__()
        conv_1_hw = conv2d_output_shape((shape[0], shape[1]), 8, 4)
        conv_2_hw = conv2d_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv2d_output_shape(conv_2_hw, 3, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(shape[2], 32, [8, 8], [4, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, [4, 4], [2, 2]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, [3, 3], [1, 1]),
            nn.LeakyReLU(),
        )
        self.fc_w = 64 * conv_3_hw[0] * conv_3_hw[1]
        self.fc = nn.Sequential(
            nn.Linear(self.fc_w, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    # shape(1,84,84,4)
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.shape[0], self.fc_w)
        x = self.fc(x)
        return x


class Conv1d(nn.Module):
    def __init__(self, length, channel, hidden_dim, out_dim):
        super(Conv1d, self).__init__()

        conv_1_l = conv1d_output_size(length, 8, 4)
        conv_2_l = conv1d_output_size(conv_1_l, 4, 2)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, 16, 8, 4),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, 4, 2),
            nn.LeakyReLU(),
        )
        self.fc_input = conv_2_l * 32
        self.fc = nn.Sequential(nn.Linear(self.fc_input, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor):
        batch = x.shape[-2]
        x = x.reshape(x.shape[-2], 2, x.shape[-1] // 2)
        hidden = self.conv(x)
        hidden = hidden.reshape(batch, self.fc_input)
        x = self.fc(hidden)
        return x


class StateNetwork(nn.Module):
    def __init__(self, obs_shape, hidden_dim=256, out_dim=64):
        assert obs_shape[0].shape == (84, 84, 4)
        assert obs_shape[1].shape == (202,)
        super(StateNetwork, self).__init__()

        self.conv2d = Conv2d(obs_shape[0].shape, hidden_dim, 64)
        self.conv1d = Conv1d(obs_shape[1].shape[-1] // 2, 2, hidden_dim, 64)
        self.fc_ir = nn.Sequential(nn.Linear(64 + 64, out_dim), nn.ReLU())
        self.apply(weights_init_)

    def forward(self, state):
        img_batch = state[0]
        ray_batch = state[1]

        img = self.conv2d(img_batch)
        ray = self.conv1d(ray_batch)
        fc = self.fc_ir(torch.cat([img, ray], dim=-1))
        return fc


class QNetworkIR(nn.Module):
    def __init__(self, obs_shape, num_actions, hidden_dim=256):
        super(QNetworkIR, self).__init__()
        self.state = StateNetwork(obs_shape, 256, 64)
        # Q1 architecture
        self.q_1 = Linear(64 + num_actions, hidden_dim, 1)
        # Q2 architecture
        self.q_2 = Linear(64 + num_actions, hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        state = self.state(state)
        xu = torch.cat([state, action], dim=-1)
        q_1 = self.q_1(xu)
        q_2 = self.q_2(xu)
        return q_1, q_2


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicyIR(GaussianPolicy):
    def __init__(self, obs_shape, num_actions, hidden_dim=256, action_space=None):
        super(GaussianPolicyIR, self).__init__()
        self.state = StateNetwork(obs_shape, 256, 64)
        self.mean_linear = Linear(64, hidden_dim, num_actions)
        self.log_std_linear = Linear(64, hidden_dim, num_actions)
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )
        self.apply(weights_init_)

    def forward(self, state):
        state = self.state(state)
        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyIR, self).to(device)


# length is (ray per direction * 2 + 1)
def generate_ray_index(length):
    bool_index = []
    for i in reversed(range(1, length)):
        if i % 2 == 0:
            bool_index.append(i)
    bool_index.append(0)
    for i in range(1, length):
        if i % 2 != 0:
            bool_index.append(i)
    bool_index = [i * 2 for i in bool_index]
    dis_index = [i + 1 for i in bool_index]
    index = dis_index + bool_index
    return index


if __name__ == "__main__":
    import numpy as np
    import sys
    from PIL import Image as im

    sys.path.append(sys.path[0] + "/../")
    from Envwrapper.UnityEnv import UnityWrapper
    from Network.Model import GaussianPolicy

    def cov2d_test():
        def show_imae(array):
            array *= 255
            b = np.hstack(i for i in np.dsplit(array, array.shape[-1]))
            b = np.squeeze(b)
            data = im.fromarray(b)
            data.show()

        cov = Conv2d(4, 256, 64)
        env = UnityWrapper("venv_605", seed=0)
        obs = env.reset()
        done = False
        while not done:
            obs, r, done, _ = env.step(env._action_space.sample())
            array = np.array(obs[0])
            c = []
            c.append(array)
            c.append(array)

            b = cov(torch.Tensor(c))
            print(b)
            show_imae(array)

    def cov1d_test():
        cov = Conv1d(101, 2, 256, 64)
        env = UnityWrapper(None, seed=0)
        obs = env.reset()
        # （50 + 50 + 1）* 2
        # 0 1 2 3 4 5 6 7
        ray = obs[1]
        a = generate_ray_index(101)
        ray = ray[a]
        a = ray * 2
        c = []
        c.append(ray)
        c.append(a)

        c = torch.tensor(c)
        y = cov(c)

        print(y)

    def gsam():
        env = UnityWrapper("venv_605", seed=0)
        gs = GaussianPolicyIR(
            env.observation_space, env.action_space.shape[0], 64, env.action_space
        )
        obs = env.reset()
        ray = obs[1]
        a = generate_ray_index(101)
        ray = ray[a]
        nr = []
        nr.append(ray)
        nr.append(ray * 2)
        nr = torch.Tensor(nr)

        img = np.array(obs[0])
        ni = []
        ni.append(img)
        ni.append(img)
        ni = torch.tensor(ni)
        m, l = gs([ni, nr])

        print(m, l)

    gsam()
