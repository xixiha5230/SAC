import cv2
import numpy as np
import torch
import torch.nn as nn
from Envwrapper.UnityEnv import label_img
from Network.ModelIR import Conv1d, weights_init_


class YoloPretrainNet():

    def __init__(self, device):
        self.device = device
        with torch.no_grad():
            self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path='./Pretrain/venv_605_middle.pt')

    def predict(self, img, show_img=False):
        t0 = t1 = None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        res = self.model(img)
        if show_img:
            cv2.imshow("view", label_img(img, res.xyxyn[0]))
            cv2.waitKey(10)
        t0 = t1 = None
        for j in res.xyxyn[0]:
            if j[5] == 0:
                t0 = j
            elif j[5] == 1:
                t1 = j
        if t0 == None:
            t0 = torch.Tensor([0, 0, 0, 0, 0, 0]).to(self.device)
        if t1 == None:
            t1 = torch.Tensor([0, 0, 0, 0, 0, 1]).to(self.device)
        res = torch.cat([t0, t1], dim=-1).cpu().numpy()
        return res


class StateNetworkDR(nn.Module):
    def __init__(self, obs_space, hidden_dim=256, out_dim=64, device='gpu'):
        assert obs_space[0].shape == (84, 84, 3)
        assert obs_space[1].shape == (1602,)
        super(StateNetworkDR, self).__init__()

        self.conv1d = Conv1d(
            (obs_space[1].shape[-1]-2) // 2, 2, hidden_dim, 64)
        self.yolo_fc = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU())
        self.fc_ir = nn.Sequential(nn.Linear(64 + 64, out_dim), nn.ReLU())
        self.apply(weights_init_)

    def forward(self, state):
        img_batch = state[0]
        ray_batch = state[1]

        img = self.yolo_fc(img_batch)
        ray = self.conv1d(ray_batch)
        fc = self.fc_ir(torch.cat([img, ray], dim=-1))
        return fc
