import numpy as np
from typing import Optional, List, Union
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


class UnityWrapper(UnityToGymWrapper):
    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        additional_args: Optional[List[str]] = None,
        side_channels: Optional[List[SideChannel]] = None,
        log_folder: Optional[str] = None,
        num_areas: int = 1,
    ):
        u_env = UnityEnvironment(
            file_name,
            worker_id,
            base_port,
            seed,
            no_graphics,
            timeout_wait,
            additional_args,
            side_channels,
            log_folder,
            num_areas,
        )
        super().__init__(u_env, allow_multiple_obs=True, action_space_seed=seed)
        self._max_episode_steps = 500

    def reset(self, seed=0) -> Union[List[np.ndarray], np.ndarray]:
        return super().reset()


def laybelimg(img, xyxyns, pixel=1):
    global img_counter
    colors = [(255, 233, 200), (23, 66, 122),
              (0, 0, 123), (233, 200, 255), (23, 66, 1220)]

    x = img.shape[0]
    y = img.shape[1]
    xyxyns = xyxyns.cpu().detach().numpy()
    for i in xyxyns:
        if i[4] > 0.5:
            img = cv2.rectangle(img, (int(i[0] * x), int(i[1] * y)),
                                (int(i[2] * x), int(i[3] * y)), colors[int(i[5])], pixel)
    if len(xyxyns) == 0:
        # cv2.imwrite("cache/{}.jpg".format(img_counter), img)
        print("no object")
        img_counter += 1
    return img


if __name__ == "__main__":
    import cv2
    import torch
    import random
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./DONE1.pt')
    env = UnityWrapper(
        "venv_605_middle",
        seed=random.randint(0, 9999),
    )
    env.reset(0)
    obs = env.reset()
    img = cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    fig, ax = plt.subplots()
    img_array = []
    img_array.append([ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))])

    img_counter = 0

    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            img = cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            results = model(img)
            img = laybelimg(img, results.xyxyn[0])
            img_array.append(
                [ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), animated=True)]
            )
            cv2.imshow("detaction", img)
            cv2.waitKey(10)
            obs, _, done, _ = env.step(env.action_space.sample())

    ani = animation.ArtistAnimation(fig, img_array, interval=1, blit=True,
                                    repeat_delay=0)
    ani.save("a.gif", fps=30)
    exit(0)
