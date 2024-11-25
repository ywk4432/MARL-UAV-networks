import argparse
from typing import Callable

import numpy as np
import tqdm
import yaml

from src.envs.env_1 import Env as Env1
from src.envs.env_2 import Env as Env2


def main(
    config_file: str, env_name: str, save_path: str, reconfig: Callable = None
) -> None:
    with open(f"src/config/envs/{config_file}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    if reconfig is not None:
        reconfig(config)
    env_type = {"env1": Env1, "env2": Env2}[env_name]
    env = env_type(**config)
    env.reset()
    for _ in tqdm.trange(config["slot_num"], desc="Time Slot"):
        action = []
        for i in range(env.uav_num):
            rewards = []
            for act in range(config["uav"]["action_size"]):
                acts = [1] * env.uav_num  # 假定其他无人机均不动
                acts[i] = act
                reward, _, _, _ = env.step(acts, dry_run=True)
                rewards.append(reward[i])
            action.append(np.argmax(rewards))
        env.step(action)
    env.record(path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_type")
    parser.add_argument("config")
    parser.add_argument("run_id")
    args = parser.parse_args()
    main(args.config, args.env_type, f"record/greedy/{args.run_id}")
