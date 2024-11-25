"""
 # @ Author: Wenke
 # @ Create Time: 2023-09-18 11:34:09
 # @ Modified by: Wenke
 # @ Modified time: 2023-09-19 06:47:49
 # @ Description: 系统中各类元素：用户集群、障碍物、地面cell格、领航无人机、跟随无人机
 """

import copy
import math

import numpy as np
import pandas as pd



PI = math.pi

"""
 # @ Author: Wenke
 # @ Create Time: 2023-09-18 11:34:09
 # @ Modified by: Wenke
 # @ Modified time: 2023-09-19 06:47:49
 # @ Description: 系统中各类元素：用户集群、障碍物、地面cell格、领航无人机、跟随无人机
 """

import copy
import math

import numpy as np
import pandas as pd



PI = math.pi


class SensorNode:
    def __init__(self, id, init_pos, collect=False, packet=[0,0]) -> None:
        self.id = id
        self.pos = tuple(init_pos)
        self.collect = collect # 收集状态：繁忙/空闲
        self.packet = packet # [数据量，信息年龄]
        self.l_aoi = []
        
    def data_add(self, env_uav_data,slot,step,e):
        if slot == e and self.packet[1] > 0 :
            self.l_aoi.append(self.packet[1])
        uav_name = f"sn{self.id}"
        data = pd.DataFrame(
            {
                "ID": [self.id],
                "slot": [slot],
                "step": [step],
                "pos": [self.pos],
                "collect": [self.collect],
                "data": [self.packet[0]],
                "aoi": [self.packet[1]],
                "last_aoi":[copy.deepcopy(self.l_aoi)]
            }
        )
        
        if uav_name in env_uav_data:
            env_uav_data[uav_name] = pd.concat(
                [env_uav_data[uav_name], data], ignore_index=True
            )
        else:
            env_uav_data[uav_name] = data




class LUAVNode:
    def __init__(
        self, id=-1, init_pos=(0, 0, 0), env_cfg=None, action=None, target_pos=(0, 0)
    ):
        if action is None:
            action = (0, 0)
        self.id = id
        self.pos = init_pos
        self.action = action  # (方向，距离)
        self.act_id = 0
        self.alive = True
        self.act_legal = True
        self.slot = 0
        self.step = 0
        self.slot_step_num = env_cfg.slot_step_num  # 单个slot内的step数
        # self.energy = env_cfg.luav_init_energy

        # self.luav_connect_dis = env_cfg.luav_connect_dis
        # self.fuav_observation_size = env_cfg.fuav_observation_size
        self.fuav_list = []  # 只保存每个slot保持连接的跟随无人机的ID
        self.fuav_num = 0  # 每个slot所连接跟随无人机的数量

       
        self.env_cfg = env_cfg
        self.agent = None
        self.dis_total = []

    def get_observation(self, env_cell_map):
        self.observation_obs[:] = 0
        self.observation_uav[:] = 0
        self.observation_ue[:] = 0
        for i in range(-self.luav_observation_size, self.luav_observation_size):
            for j in range(-self.luav_observation_size, self.luav_observation_size):
                index_i = i + self.luav_observation_size
                index_j = j + self.luav_observation_size
                if (
                    self.pos[0] + i < 0
                    or self.pos[0] + i >= self.env_cfg.map_length
                    or self.pos[1] + j < 0
                    or self.pos[1] + j >= self.env_cfg.map_width
                ):
                    self.observation_obs[index_i][index_j] = -1
                    self.observation_uav[index_i][index_j] = -1
                    self.observation_ue[index_i][index_j] = -1
                else:
                    self.observation_obs[index_i][index_j] = env_cell_map[
                        self.pos[0] + i
                    ][self.pos[1] + j].obs
                    self.observation_uav[index_i][index_j] = env_cell_map[
                        self.pos[0] + i
                    ][self.pos[1] + j].uav_inf[1]
                    self.observation_ue[index_i][index_j] = env_cell_map[
                        self.pos[0] + i
                    ][self.pos[1] + j].ue_num

    def clear(self):
        # self.fuav_list.clear()
        self.action = (0, 0)
        # self.fuav_absact_kill_num = 0
        # self.slot_reward = 0
        # self.reward = 0
        self.act_id = 0
        self.dis_total = []
            

    def step_run(self):
        self.step += 1
        step_dir, step_dis = self.action[0], self.action[1]
        # 不取整?

        new_pos = (
            round(self.pos[0] + math.cos(step_dir) * step_dis),
            round(self.pos[1] + math.sin(step_dir) * step_dis),
            self.pos[2],
        )
        self.pos = new_pos

    def data_add(self, env_uav_data):
        uav_name = f"luav{self.id}"
        data = pd.DataFrame(
            {
                "ID": [self.id],
                "slot": [self.slot],
                "step": [self.step],
                "pos": [self.pos],
                "action": [self.action],
                "act_id": [self.act_id],
                # "act_legal": [self.act_legal],
                # "slot_reward": [self.slot_reward],
                # "reward": [self.reward],
                # "sub_reward": [copy.deepcopy(self.sub_reward)],
                # "fuav_list": [copy.deepcopy(self.fuav_list)],
                # "fuav_num": [self.fuav_num],
                # "fuav_absact_kill_num": [self.fuav_absact_kill_num],
            }
        )
        if uav_name in env_uav_data:
            env_uav_data[uav_name] = pd.concat(
                [env_uav_data[uav_name], data], ignore_index=True
            )
        else:
            env_uav_data[uav_name] = data

class FUAVNode:
    def __init__(
        self,
        id=-1,
        luav_id=-1,
        init_pos_abs=(0, 0),
        env_cfg=None,
        env_acts=None,
        target_pos=(0, 0)
    ):
        if env_acts is None:
            env_acts = []
        self.id = id
        self.alive = True
        self.pos_abs = init_pos_abs
        
        
        self.slot = 0  # slot数
        self.step = 0  # step数

     

        self.luav_connect = True
        self.luav_id = luav_id

        # self.luav_pos_abs = (0, 0)  # 领航无人机的目标点
        # self.act_rela = (0, 0)  # 跟随无人机相对位置目标点
        # self.act_abs_legal = True  # 领航无人机动作的合法性
        self.act = []  # 无人机的动作
        # self.act_legal = True  # 无人机的动作检查
        
        self.energy = env_cfg.fuav_init_energy  # 无人机可用储能
        self.step_ecost = 0.0  # 单个step内耗能

        self.state_size = env_cfg.lagent_state_size
        self.obs_size = env_cfg.lagent_obs_size
        self.act_dim = env_cfg.lagent_act_dim

     
        self.state = np.array([0.0 for _ in range(self.state_size)])
        self.next_state = np.array([0.0 for _ in range(self.state_size)])
        self.step_reward = 0  # 无人机在单个step内的动作合法奖励值
        self.reward_slot_end = 0  # 无人机在slot中是否抵达目标位置的奖励值

        self.env_cfg = env_cfg  # 存储系统性参数
        self.agent = None

       

        self.l_reward = 0
        self.reward = 0.0
        self.sub_reward = []
        # self.target_pos = (target_pos[0] + init_pos_rela[0], target_pos[1] + init_pos_rela[1])
        self.sn_list = []
        self.cluster = -1
        self.sn_data = []
        
        self.work = False
        self.data_v = 0 # 收集数据量
        
        

    def data_add(self, env_uav_data):
        uav_name = f"fuav{self.id}"
        data = pd.DataFrame(
            {
                "ID": [self.id],
                "slot": [self.slot],
                "step": [self.step],
                "pos_abs": [self.pos_abs],
                "cluster":[self.cluster],
                "sn_list":[self.sn_list],
                "sn_data":[copy.deepcopy(self.sn_data)],
                "energy": [self.energy],
                "action": [self.act],
                "collect": [self.work],
                "dist_or_data":[self.data_v],
                # "reward_total": [self.reward_total],
                "flight_reward":[self.l_reward],
                "collect_reward": [self.reward],
                "sub_reward": [copy.deepcopy(self.sub_reward)],
                
            }
        )
        
        if uav_name in env_uav_data:
            env_uav_data[uav_name] = pd.concat(
                [env_uav_data[uav_name], data], ignore_index=True
            )
        else:
            env_uav_data[uav_name] = data

    def get_observation(self,env_sn_list):
        # 无人机位置，能量
        # SN位置，数据量，aoi，aipv
        self.observation_uav=[]
        self.observation_sn=[]
        
        self.observation_uav.append(self.pos_abs[0])
        self.observation_uav.append(self.pos_abs[1])
        self.observation_uav.append(self.energy)
        
        for sn_id in self.sn_list:
            sn = env_sn_list[sn_id]
            self.observation_sn.append(sn.pos[0])
            self.observation_sn.append(sn.pos[1])
            self.observation_sn.append(sn.packet[0])
            self.observation_sn.append(sn.packet[1])
            self.observation_sn.append(sn.packet[0]* math.exp(sn.packet[1]))
            
        self.observation_uav.extend(self.observation_sn)   
        return self.observation_uav

    def clear(self):
        """
        每个step开始都要进行信息清理
        """
        self.act = []
        self.act_legal = True
        self.step_ecost = 0.0
        self.step_reward = 0.0
        self.sub_reward = []
        self.reward = 0
        self.l_reward = 0
        self.work = False #是否收集数据
        self.data_v = 0 #飞行距离or收集数据量
        self.sn_data = []
        

    def energy_update(self):
        """更新无人机悬停能耗"""
        self.energy = self.energy - self.env_cfg.hover_cost

    def step_run(self, env_cell_map, env_luav_list, act, env_uav_data):
        """
        无人机执行动作
        """
        if self.alive:
            # 清理上个step的状态
            self.clear()
            if act is not None:
                # 接收输入的动作
                self.act = act
            else:
                print("No Action Received.")

            # print(f"fuav {self.id} step {self.step}")
            self.step += 1
            self.act_execute_2(env_cell_map, env_luav_list)

            if self.step % self.env_cfg.slot_step_num == 1:
                self.slot += 1
            # self.data_add(env_uav_data=env_uav_data)

    def frame_run(self, act, env_uav_data,env_sn_list):
        """
        无人机执行动作
        """
        self.clear()
        if self.energy>0:
            # 清理上个step的状态
            
            if act is not None:
                # 接收输入的动作
                self.act = act
                assert len(self.act) == len(self.sn_list)
            else:
                print("No Action Received.")

            # print(f"fuav {self.id} step {self.step}")
            
            for id in self.sn_list:
                sn = env_sn_list[id]
                data = sn.packet[0]
                self.sn_data.append(data)
            assert len(self.sn_data) == len(self.sn_list)
            # 收集数据
            self.collect_data(env_sn_list)
            # 更新悬停能量
            self.energy_update()
            
            # self.data_add(env_uav_data=env_uav_data)
        self.step += 1    
        if self.step % self.env_cfg.slot_step_num == 1:
            self.slot += 1
    
    def collect_data(self, env_sn_list):
        """
        无人机收集数据和更新传输能量
        """
        for i in range(len(self.act)):
            if self.act[i]:
                q_id = self.sn_list[i]
                q_sn = env_sn_list[q_id]
                q_data = q_sn.packet[0]
                if not(q_sn.collect or q_sn.packet[0] or q_sn.packet[1]):
                    # 没有要传输的
                    # print("The SN node doesn't have packet!")
                    pass
                elif q_sn.collect and (q_sn.packet[0]>0):
                    # 有要传输的
                    self.data_v = q_sn.packet[0]
                    # 收集数据量
                    d = math.sqrt(
                            (self.pos_abs[0] - q_sn.pos[0]) ** 2
                            + (self.pos_abs[1] - q_sn.pos[1]) ** 2
                            + self.env_cfg.h
                        )
                    if d == 0:
                        d = 0.001
                    data = self.env_cfg.step_time*max(math.log(self.env_cfg.collect_speed /d),0)
                    # if data <= 0:
                    #     print(d)
                    
                    if data > 0:
                        self.work = True
                        q_sn.packet[0] -=  data
                        
                    if q_sn.packet[0]<=0:
                        self.energy -= (self.env_cfg.transmit_cost *(1+q_sn.packet[0]/data))
                        q_sn.packet[0] = 0
                        last_data = q_sn.packet[1]
                        q_sn.l_aoi.append(last_data)
                        q_sn.packet[1] = 0
                        q_sn.collect = False   
                        
                    else:
                        self.data_v = data
                        # if self.data_v <= 0:
                        #     print(self.data_v)
                        self.energy -= self.env_cfg.transmit_cost
                else:
                    print("sn packet error!")
                
        
