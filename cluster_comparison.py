import argparse
import math
import os
from typing import Callable

import numpy as np
import tqdm
import yaml
import random

from src.envs.env_4 import SNEnv
from src.envs.env_4.match_algorithms import *



def get_luav_actions(env):
    uav_pos = []
    uav_energy = []
    for uav in env.fuav_list:
        uav_pos.append(uav.pos_abs)
        uav_energy.append(max(0, uav.energy))

    greedy_matching_result, greedy_total_cost = greedy_matching(
        uav_pos, env.cluster_center
    )

    return greedy_matching_result

def get_fuav_actions(env, config):
    fuav_action = []
    for fuav in env.fuav_list:
        act_id = greedy_dis_fuav(fuav,env,config)  # 调用映射到的函数
        fuav_action.append(act_id)
    # print(f"fuav_action is {fuav_action}")
    assert len(fuav_action) == config["fuav_num"]
    return fuav_action

def greedy_dis_fuav(fuav,env,config):
    
    # sn_data_list = []
    # for sn_id in fuav.sn_list:
    #     sn = env.sn_list[sn_id]
    #     sn_data_list.append(sn.packet[0])
    # print(f"sn_data_list:{sn_data_list}")
    
    
    sn_dis_list = []
    act = 0
    for sn_id in fuav.sn_list:
        sn = env.sn_list[sn_id]
        dis = math.sqrt(
                (fuav.pos_abs[0] - sn.pos[0]) ** 2
                + (fuav.pos_abs[1] - sn.pos[1]) ** 2
            )
        sn_dis_list.append(dis)
    # print(f"sn_dis_list:{sn_dis_list}")
    sorted_indices = sorted(range(len(sn_dis_list)), key=lambda x: sn_dis_list[x])
    # print(f"sorted_indices:{sorted_indices}")
    for idx in sorted_indices:
        # print(f"idx:{idx}")
        sn_id = fuav.sn_list[idx]
        sn = env.sn_list[sn_id]
        if sn.packet[0]>0:
            act = idx
            break
    # print(act)
    return act


def main(config_file: str, save_path: str = None, reconfig: Callable = None) -> None:
    with open(f"src/config/envs/{config_file}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
        
    if reconfig is not None:
        reconfig(config)

    env = SNEnv(**config)
    
    # algo = ["bkm","kmeans"]
    algo = ["bkm","kmeans","gmm"]
    
    for i in range(len(algo)):
    # for i in range(1):
        print(f"this is turn {i}  algo is {algo[i]}")
        save_path = f"record/TPRA/cluster/{os.getpid()}/{i}_{algo[i]}"
        
        env.reset()

        for _ in tqdm.trange(1, desc="Time Slot"):
            slot_step = 0
            t = 0
            slot = 0
            while True:
                if slot_step == 0:
                    if slot >= config["episode_limit"]:
                        break
                    
                    # 聚类！
                    if i == 0:
                        env.bkm()
                    elif i == 1:
                        env.kmeans()
                    elif i == 2:
                        env.gmm()
                    else:
                        print("algo_id error")
                    
                    # 匹配  luav act
                    l_actions = get_luav_actions(env)
                    # print(f"luav_action is {l_actions}")

                    env.step(slot_step, [l_actions])

                    slot_step += 1

                elif slot_step <= config["slot_step_num"]:
                    # 不同对比算法 fuav act
                    f_actions = get_fuav_actions(env, config)
                    env.step(slot_step, [f_actions])
                    
                    t += 1
                    
                    slot_step += 1
                    if slot_step == (config["slot_step_num"] + 1):
                        slot_step = 0
                        slot += 1

        env.record(t, path=save_path)

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")  # env_arg
    # parser.add_argument("run_id")
    args = parser.parse_args()
    main(args.config)
