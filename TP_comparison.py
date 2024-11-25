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



def get_luav_actions(env,config,algo_id):
    function_map = {
        0: greedy_matching,
        1: hungarian_matching,
        2: sorted_matching
    }
    
    uav_pos = []
    uav_energy = []
    for uav in env.fuav_list:
        uav_pos.append(uav.pos_abs)
        uav_energy.append(max(0, uav.energy))
        
   
    # 根据序号 i 调用相应的函数
    #print(algo_id)
    if algo_id in function_map:
        if algo_id < 2:
            matching_result,total_cost  = function_map[algo_id](uav_pos, env.cluster_center)  # 调用映射到的函数
        else:
            matching_result,total_cost  = function_map[algo_id]( uav_energy, env.data_list, uav_pos, env.cluster_center)  # 调用映射到的函数
                  
    else:
        print("Invalid function number!")
    # print(f"fuav_action is {fuav_action}")
    assert len(matching_result) == config["fuav_num"]
    return matching_result

def get_fuav_actions(env, config):
    fuav_action = []
    for fuav in env.fuav_list:
        act_id = greedy_data_fuav(fuav,env,config)  # 调用映射到的函数
        fuav_action.append(act_id)
    # print(f"fuav_action is {fuav_action}")
    assert len(fuav_action) == config["fuav_num"]
    return fuav_action


def greedy_data_fuav(fuav,env,config):
    sn_data_list = []
    for sn_id in fuav.sn_list:
        sn = env.sn_list[sn_id]
        sn_data_list.append(sn.packet[0])
    return sn_data_list.index(max(sn_data_list))
    


def main(config_file: str, save_path: str = None, reconfig: Callable = None) -> None:
    with open(f"src/config/envs/{config_file}.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
        
    if reconfig is not None:
        reconfig(config)

    env = SNEnv(**config)
    
    algo = ["greedy","hungarian","sorted"]
    for i in range(len(algo)):
    # for i in range(1):
        print(f"this is turn {i}  algo is {algo[i]}")
        save_path = f"record/TPRA/TP/{os.getpid()}/{i}_{algo[i]}"
        
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
                    env.gmm()
                    
                    # 匹配  luav act
                    l_actions = get_luav_actions(env,config,i)
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
