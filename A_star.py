import argparse
import math
import os
from typing import Callable

import numpy as np
import tqdm
import yaml

from src.envs.env_3 import LUAVEnv
from src.envs.env_3 import supple


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


def act_id_get(i, j):
    if i == 0 and j == 0:
        id = 0
    elif i == 1 and j == 1:
        id = 1
    elif i == 0 and j == 1:
        id = 2
    elif i == -1 and j == 1:
        id = 3
    elif i == -1 and j == 0:
        id = 4
    elif i == -1 and j == -1:
        id = 5
    elif i == 0 and j == -1:
        id = 6
    elif i == 1 and j == -1:
        id = 7
    elif i == 1 and j == 0:
        id = 8
    return id


def act_check(action, pos, env, config):
    """
    只负责检查领航无人机动作的合法性，不执行
    """
    act_legal = True

    dir, dis = action[0], action[1]
    env_cell_map = env.cell_map

    new_end = (
        round(pos[0] + math.cos(dir) * dis),
        round(pos[1] + math.sin(dir) * dis),
    )
    pos_x, pos_y = new_end[0], new_end[1]

    if (
        pos_x < config["safe_dis"]
        or pos_x >= env.map_length - config["safe_dis"]
        or pos_y < config["safe_dis"]
        or pos_y >= env.map_width - config["safe_dis"]
    ):
        # print("luav即将飞出边界 -> illegal action 0")
        act_legal = False

    else:
        if env_cell_map[pos_x][pos_y].obs == 1:
            # print("luav即将前往区域存在障碍物 -> illegal action 3")
            act_legal = False

    new_poses = supple.ac_detect(pos, action)

    for new_pos in new_poses:
        # pos_x, pos_y, pos_z = new_pos[0], new_pos[1], new_pos[2]
        pos_x, pos_y = round(new_pos[0]), round(new_pos[1])

        if pos_x < 0 or pos_x >= env.map_length or pos_y < 0 or pos_y >= env.map_width:
            # print("luav即将飞出边界 -> illegal action 0")
            act_legal = False

        else:
            if env_cell_map[pos_x][pos_y].obs == 1:
                # print("luav即将前往区域存在障碍物 -> illegal action 3")
                act_legal = False
        posdtes = [
            (pos_x + 1, pos_y),
            (pos_x - 1, pos_y),
            (pos_x, pos_y + 1),
            (pos_x, pos_y - 1),
        ]
        for posdt in posdtes:
            pos_dtx, pos_dty = posdt[0], posdt[1]
            if (
                pos_dtx < 0
                or pos_dtx >= env.map_length
                or pos_dty < 0
                or pos_dty >= env.map_width
            ):
                # print("luav即将飞出边界 -> illegal action 0")
                act_legal = False

            else:
                if env_cell_map[pos_dtx][pos_dty].obs == 1:
                    # print("luav即将前往区域存在障碍物 -> illegal action 3")
                    act_legal = False
    return act_legal


def get_luav_actions(env, config):
    l_action = []
    l_action_num = env.get_l_actions()
    ue_center = env.ue_cluster_center_list[0]
    tx, ty = ue_center.pos[0], ue_center.pos[1]

    for luav in env.luav_list:
        l_f = []
        for act in range(l_action_num):
            luavaction = env.laction_convert([act])
            step_dir, step_dis = luavaction[0], luavaction[1]
            l_pos_x = round(luav.pos[0] + math.cos(step_dir) * step_dis)
            l_pos_y = round(luav.pos[1] + math.sin(step_dir) * step_dis)
            g = step_dis
            dx = abs(tx - l_pos_x)
            dy = abs(ty - l_pos_y)
            dmin = dx if dx <= dy else dy
            dmax = dy if dx <= dy else dx
            h = math.sqrt(5) * dmin + math.sqrt(2) * dmax
            f = g + h
            # 动作合法性
            if not act_check(luavaction, luav.pos, env, config):
                f += 100
            l_f.append(f)
            # print(f"act:{act} l_fg:{l_fg}")

        l_action.append(np.argmin(l_f))

    return l_action


def get_fuav_actions(env, config, lact, pr):
    # luav绝对动作
    luavaction = env.laction_convert([lact])
    dir, dis = luavaction[0], luavaction[1] / config["slot_step_num"]
    x = math.cos(dir) * dis
    y = math.sin(dir) * dis

    action = []
    f_obs = env.get_fuav_obs()
    f_obs_size = (env.fuav_obs_size) ** 2
    f_action_num = env.get_f_actions()
    l = 2 * config["luav_connect_dis"] + config["slot_step_num"] + config["dis"][-1]
    for fuav in env.fuav_list:

        i = fuav.id
        pos_x, pos_y = fuav.pos_abs[0], fuav.pos_abs[1]
        f_obs_i = f_obs[i]
        # f_obs_obs = f_obs_i[0:f_obs_size]
        f_obs_uav = f_obs_i[f_obs_size : 2 * f_obs_size]
        f_rela_x = f_obs_i[-4]
        f_rela_y = f_obs_i[-3]
        ue_x = f_obs_i[-2] * config["map_length"]
        ue_y = f_obs_i[-1] * config["map_width"]
        if pr:
            print(f"ue_x:{ue_x} ue_y:{ue_y}")
        # if pr:
        #     print(f"pos_abs:{pos_x} {pos_y}  act_rela:{fuav.act_rela}")

        # 编队
        fd = -(f_rela_x**2) - f_rela_y**2
        w = 3 - (math.e) ** fd
        act_x = x - w * f_rela_x
        act_y = y - w * f_rela_y

        # 避障
        tx = fuav.act_rela[0]
        ty = fuav.act_rela[1]
        # tx = ue_x
        # ty = ue_y
        f_dwa = np.zeros(f_action_num)
        for act in range(f_action_num):
            dx, dy, obs_id = act_id_trans(act)
            df_obs = env.cell_map[pos_x + dx][pos_y + dy].obs
            df_uav = env.cell_map[pos_x + dx][pos_y + dy].uav_inf[1]

            if act != 0:
                if df_obs or df_uav:
                    f_dwa[act] = -1
                else:
                    dis = math.sqrt((tx - pos_x - dx) ** 2 + (ty - pos_y - dy) ** 2)
                    if dis != 0:
                        f_dwa[act] = 5 / dis
                    else:
                        f_dwa[act] = 7
            elif act == 0:
                dis = math.sqrt((tx - pos_x) ** 2 + (ty - pos_y) ** 2)
                if dis != 0:
                    f_dwa[act] = 5 / dis
                else:
                    f_dwa[act] = 7

        # if pr:
        #     print(f"f_dwa_list:{f_dwa}")
        dwa_act = np.argmax(f_dwa)
        # if pr:
        #     print(f"f_dwa_act:{dwa_act}")
        dwa_x, dwa_y, dwa_id = act_id_trans(dwa_act)
        a = 0.5
        act_x += dwa_x * a
        act_y += dwa_y * a

        # 转换
        if pr:
            print(f"l:{x} rela:{w * f_rela_x} dwa:{dwa_x*a} act_x:{act_x}")
            print(f"l:{y} rela:{w * f_rela_y} dwa:{dwa_y*a} act_y:{act_y}")
        act_x = round(act_x)
        act_y = round(act_y)
        if act_x > 1:
            act_x = 1
        elif act_x < -1:
            act_x = -1
        if act_y > 1:
            act_y = 1
        elif act_y < -1:
            act_y = -1
        act_id = act_id_get(act_x, act_y)
        if pr:
            print(f"real act: act_x:{act_x} act_y:{act_y} act:{act_id}")

        # 调整

        action.append(act_id)
        # action.append(dwa_act)

    return action


def main(config_file: str, save_path: str, reconfig: Callable = None) -> None:
    with open(f"src/config/envs/{config_file}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    if reconfig is not None:
        reconfig(config)

    env = LUAVEnv(**config)
    env.reset()

    for _ in tqdm.trange(1, desc="Time Slot"):
        slot_step = 0
        t = 0
        slot = 0
        # while True:
        #     if slot_step == 0:
        #         # 确定领导者无人机的动作

        #         l_actions = get_luav_actions(env,config)
        #         env.step(slot_step,'l',l_actions)
        #         t += 1
        #         if t>config["episode_limit"]:
        #             break
        #         slot_step = 0
        l = [34, 34, 33, 33, 34, 33, 26, 35, 35, 35, 0, 0, 0, 0, 0, 0]
        while True:
            if slot_step == 0:
                # 确定领导者无人机的动作

                l_actions = l[slot]
                env.step_unselect(slot_step, "l", [l_actions])
                print(slot)
                slot_step += 1

            elif slot_step <= config["slot_step_num"]:
                if slot <= 4:
                    pr = 1
                else:
                    pr = 0
                f_actions = get_fuav_actions(env, config, l_actions, pr)
                env.step_unselect(slot_step, "f", f_actions)
                t += 1
                if t > (config["episode_limit"] * config["slot_step_num"]):
                    break
                slot_step += 1
                if slot_step == (config["slot_step_num"] + 1):
                    slot_step = 0
                    slot += 1

    env.record(t, path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")  # formation(_subn)
    # parser.add_argument("run_id")
    args = parser.parse_args()
    main(args.config, f"record/formation/A*/{os.getpid()}")
