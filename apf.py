import argparse
import math
import os
from typing import Callable

import numpy as np
import tqdm
import yaml

from src.envs.env_3 import LUAVEnv


def act_id_trans(act):
    if act == 0:  # 0.0
        i = 0
        j = 0
        obs_id = 4
    elif act == 1:  # pi / 4
        i = 1
        j = 1
        obs_id = 8
    elif act == 2:  # pi / 2
        i = 0
        j = 1
        obs_id = 5
    elif act == 3:  # 3pi /4
        i = -1
        j = 1
        obs_id = 2
    elif act == 4:  # pi
        i = -1
        j = 0
        obs_id = 1
    elif act == 5:  # 5pi / 4
        i = -1
        j = -1
        obs_id = 0
    elif act == 6:  # 3pi / 2
        i = 0
        j = -1
        obs_id = 3
    elif act == 7:  # 7pi / 4
        i = 1
        j = -1
        obs_id = 6
    elif act == 8:  # 2pi
        i = 1
        j = 0
        obs_id = 7
    return i, j, obs_id


def get_luav_actions(env):
    l_action = []

    l_action_num = env.get_l_actions()
    for luav in env.luav_list:
        l_apf = []
        for act in range(l_action_num):
            luavaction = env.laction_convert([act])
            step_dir, step_dis = luavaction[0], luavaction[1]
            l_pos_x = round(luav.pos[0] + math.cos(step_dir) * step_dis)
            l_pos_y = round(luav.pos[1] + math.sin(step_dir) * step_dis)
            l_fg = env.cell_map[l_pos_x][l_pos_y].apf
            l_apf.append(l_fg)
            # print(f"act:{act} l_fg:{l_fg}")

        l_action.append(np.argmax(l_apf))

    return l_action


def get_fuav_actions(env, config):
    action = []
    f_obs = env.get_fuav_obs()
    f_obs_size = (env.fuav_obs_size) ** 2
    f_action_num = env.get_f_actions()
    l = 2 * config["luav_connect_dis"] + config["slot_step_num"] + config["dis"][-1]
    for fuav in env.fuav_list:
        apf = []
        i = fuav.id
        pos_x, pos_y = fuav.pos_abs[0], fuav.pos_abs[1]
        f_obs_i = f_obs[i]
        f_obs_uav = f_obs_i[f_obs_size : 2 * f_obs_size]
        f_rela_x = f_obs_i[-2]
        f_rela_y = f_obs_i[-1]
        for act in range(f_action_num):
            x, y, obs_id = act_id_trans(act)
            fg = env.cell_map[pos_x + x][pos_y + y].apf
            if act != 0:
                # 避障 斥力
                fg -= f_obs_uav[obs_id]
            # 编队 引力
            dis_rela = math.sqrt((f_rela_x * l + x) ** 2 + (f_rela_y * l + y) ** 2)
            if dis_rela != 0:
                fg += 2 / dis_rela
            else:
                fg += 4

            apf.append(fg)

        action.append(np.argmax(apf))
    return action


def main(config_file: str, save_path: str, reconfig: Callable = None) -> None:
    with open(f"src/config/envs/{config_file}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    if reconfig is not None:
        reconfig(config)

    env = LUAVEnv(**config)
    env.reset()

    for _ in tqdm.trange(1, desc="Time Slot"):
        # 时间尺度？
        slot_step = 0
        t = 0
        while True:
            if slot_step == 0:
                # 确定领导者无人机的动作

                l_actions = get_luav_actions(env)
                env.step(slot_step, "l", l_actions)

                slot_step += 1

            elif slot_step == config["slot_step_num"]:
                f_actions = get_fuav_actions(env, config)
                env.step(slot_step, "f", f_actions)
                t += 1
                if t > config["episode_limit"]:
                    break
                slot_step = 0

    env.record(t, path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")  # formation(_subn)
    # parser.add_argument("run_id")
    args = parser.parse_args()
    main(args.config, f"record/formation/apf/{os.getpid()}")
