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


def get_fuav_actions(env, config, algo_id):
    # fuavactions 
    function_map = {
        0: random_fuav,
        1: greedy_data_fuav,
        2: greedy_aoi_fuav
    }
    fuav_action = []
    # 根据序号 i 调用相应的函数
    #print(algo_id)
    if algo_id in function_map:
        for fuav in env.fuav_list:
            act_id = function_map[algo_id](fuav,env,config)  # 调用映射到的函数
            fuav_action.append(act_id)
            
    else:
        print("Invalid function number!")
    # print(f"fuav_action is {fuav_action}")
    assert len(fuav_action) == config["fuav_num"]
    return fuav_action


def random_fuav(fuav,env,config):
    # 生成随机动作序列
    x = int(config["sn_num"] / config["fuav_num"])
    return  random.randrange(0, x)


def greedy_data_fuav(fuav,env,config):
    sn_data_list = []
    for sn_id in fuav.sn_list:
        sn = env.sn_list[sn_id]
        sn_data_list.append(sn.packet[0])
    return sn_data_list.index(max(sn_data_list))
    
 
def greedy_aoi_fuav(fuav,env,config):
    sn_aoi_list = []
    for sn_id in fuav.sn_list:
        sn = env.sn_list[sn_id]
        sn_aoi_list.append(sn.packet[1])
    return sn_aoi_list.index(max(sn_aoi_list))


def main(config_file: str, save_path: str = None, reconfig: Callable = None) -> None:
    with open(f"src/config/envs/{config_file}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
        
    if reconfig is not None:
        reconfig(config)

    
    env = SNEnv(**config)
    
    algo = ["random","greedy_data","greedy_aoi"]
    for i in range(len(algo)):
    # for i in range(1):
        print(f"this is turn {i}  algo is {algo[i]}")
        save_path = f"record/TPRA/RA/{os.getpid()}/{i}_{algo[i]}"
        env.reset()

        for _ in tqdm.trange(1, desc="Time Slot"):
            slot_step = 0
            t = 0
            slot = 0
            while True:
                if slot_step == 0:
                    if slot >= config["episode_limit"]:
                        break
                    
                    # 聚类
                    env.bkm()
                    
                    # 匹配  luav act
                    l_actions = get_luav_actions(env)
                    # print(f"luav_action is {l_actions}")

                    env.step(slot_step, [l_actions])

                    slot_step += 1

                elif slot_step <= config["slot_step_num"]:
                    # 不同对比算法 fuav act
                    f_actions = get_fuav_actions(env, config,i)
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
