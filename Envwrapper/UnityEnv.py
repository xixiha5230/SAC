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


if __name__ == "__main__":
    env = UnityWrapper(
        "venv_605",
        seed=0,
    )
    env.reset(0)
    a = env.reset()

    print(a)
