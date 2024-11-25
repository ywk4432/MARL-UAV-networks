#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
from types import SimpleNamespace

from .element import *
from .match_algorithms import *

from src.clustering.my_kmeans import *
from ..multiagentenv import MultiAgentEnv
from datetime import datetime

# from data_save import *
from . import data_save
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import csv

uav_color_list = [
    "y",
    "g",
    "teal",
    "m",
    "hotpink",
    "c",
    "b",
    "r",
    "orange",
    "purple",
    "indigo",
    "tan",
    "royalblue",
    "w",
    "k",
]


class SNEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        cfg = SimpleNamespace(**kwargs)

        random.seed(cfg.seed)
        self.name = cfg.env_name
        self.luav_num = cfg.luav_num
        self.fuav_num = cfg.fuav_num

        self.l_n_agents = cfg.luav_num
        self.f_n_agents = cfg.fuav_num

        #
        self.map_length = cfg.map_length  # 行数
        self.map_width = cfg.map_width  # 列数

        # 地面SN分布生成
        # self.sn_list = [
        #     SensorNode(id=index, init_pos=sn)
        #     for index, sn in enumerate(cfg.sn_pos_list)
        # ]
        self.sn_list = [
            SensorNode(
                id=index,
                init_pos=[
                    random.randint(0, self.map_length),
                    random.randint(0, self.map_width),
                ],
            )
            for index in range(cfg.sn_num)
        ]
        self.sn_threshold = cfg.sn_threshold

        self.slot = -1
        self.step_c = -1
        self.slot_step_num = cfg.slot_step_num

        self.luav_list = []
        self.fuav_list = []

        self.luav_init_pos_list = cfg.luav_init_pos_list
        self.fuav_init_pos_list = cfg.fuav_init_pos_list

        #
        # self.fuav_acts = cfg.fuav_acts  # 9维

        self.init_energy = cfg.fuav_init_energy
        self.flight_cost = cfg.flight_cost  # 飞行能耗
        self.hover_cost = cfg.hover_cost  # 悬停能耗
        self.transmit_cost = cfg.transmit_cost  # 传输能耗

        #
        # self.cell_map = []

        self.cfg = cfg
        self.uav_data = {}  # 存储所有无人机在各时隙的信息

        self.done = False

        self.episode_limit = cfg.episode_limit
        self.state = []

        self.cluster_center = []
        self.cluster_list = []
        self.bkm_clusters = []
        self.data_list = []
        self.kl_list = []
        self.r_tp_list = []

    def env_nodes_clear(self, l, f):
        """
        清理环境中内所有节点上一时隙不必要状态
        """
        if l:
            # print("清理领航无人机信息")
            for i, luav in enumerate(self.luav_list):
                luav.clear()
        if f:
            # print("清理跟随无人机信息")
            for i, fuav in enumerate(self.fuav_list):
                fuav.clear()

    def env_slot_clear(self, l=True, f=True):
        """
        清空环境中上一时隙的无用信息 —— 大时间尺度
        """
        self.env_nodes_clear(l=l, f=f)

        self.step_c = 0

    def env_step_clear(self, l=False, f=True):
        """
        清空环境中上个step的信息 —— 小时间尺度
        """
        self.env_nodes_clear(l=l, f=f)

    def sn_update(self, id):
        """
        SN节点随机产生状态包
        """

        sn_p = random.random()
        if sn_p >= self.sn_threshold:
            if not (
                self.sn_list[id].collect
                or self.sn_list[id].packet[0]
                or self.sn_list[id].packet[1]
            ):
                self.sn_list[id].collect = True
                self.sn_list[id].packet = [sn_p * 5 + 5, 0]
            elif self.sn_list[id].collect and (self.sn_list[id].packet[0] > 0):
                pass
                # print("The SN node already have packet!")
            else:
                print("sn packet error!")

    def sn_slot_update(self):
        for sn in self.sn_list:
            if sn.collect and (sn.packet[0] > 0):
                sn.packet[1] += 1
            elif not (sn.collect or sn.packet[0] or sn.packet[1]):
                self.sn_update(sn.id)
                # print(sn.id)
                # print(sn.packet[0])
            else:
                print("sn packet error!")

    def kmeans(self):
        p_list = []
        self.cluster_center = np.array([])
        self.cluster_list = np.array([])
        self.data_list = np.array([])
        # self.bkm_clusters.clear()

        for sn_obj in self.sn_list:
            point_obj = Point(
                p_id=sn_obj.id,
                pos=sn_obj.pos,
                data=sn_obj.packet[0],
                aoi=sn_obj.packet[1],
            )
            p_list.append(point_obj)

        bkm = BalancedKMeans(
            cluster_num=len(self.fuav_list),
            max_iter=self.cfg.MAX_ITERATIONS,
            p_list=p_list,
        )

        K = self.fuav_num
        pos_list = [sn.pos for sn in self.sn_list]
        locations = np.array(pos_list)
        centroids, kmeans_data_list, kmeans_clusters = kmeans_cluster(K, locations, bkm)

        self.cluster_center = centroids
        self.cluster_list = kmeans_clusters
        self.data_list = kmeans_data_list

    def gmm(self):
        p_list = []
        self.cluster_center = np.array([])
        self.cluster_list = np.array([])
        self.data_list = np.array([])
        # self.bkm_clusters.clear()

        for sn_obj in self.sn_list:
            point_obj = Point(
                p_id=sn_obj.id,
                pos=sn_obj.pos,
                data=sn_obj.packet[0],
                aoi=sn_obj.packet[1],
            )
            p_list.append(point_obj)

        bkm = BalancedKMeans(
            cluster_num=len(self.fuav_list),
            max_iter=self.cfg.MAX_ITERATIONS,
            p_list=p_list,
        )

        K = self.fuav_num
        pos_list = [sn.pos for sn in self.sn_list]
        locations = np.array(pos_list)
        centroids, gmm_data_list, gmm_clusters = gmm_cluster(K, locations, bkm)

        self.cluster_center = centroids
        self.cluster_list = gmm_clusters
        self.data_list = gmm_data_list

    def bkm(self):
        p_list = []
        self.cluster_center.clear()
        self.cluster_list.clear()
        self.data_list.clear()
        self.bkm_clusters.clear()

        for sn_obj in self.sn_list:
            point_obj = Point(
                p_id=sn_obj.id,
                pos=sn_obj.pos,
                data=sn_obj.packet[0],
                aoi=sn_obj.packet[1],
            )
            p_list.append(point_obj)

        bkm = BalancedKMeans(
            cluster_num=len(self.fuav_list),
            max_iter=self.cfg.MAX_ITERATIONS,
            p_list=p_list,
        )
        bkm.fit()
        # bkm.show()
        self.bkm_clusters = bkm.clusters

        for cluster in bkm.clusters:
            self.cluster_center.append(cluster.center)
            self.cluster_list.append(cluster.points)
            c_data = cluster.get_data()
            self.data_list.append(c_data)

    def match(self, luav_act):
        # uav cluster energy cluster_data
        uav_pos = []
        uav_energy = []
        for uav in self.fuav_list:
            uav_pos.append(uav.pos_abs)
            uav_energy.append(max(0, uav.energy))

        greedy_matching_result, greedy_total_cost = greedy_matching(
            uav_pos, self.cluster_center
        )
        hungarian_matching_result, hungarian_total_cost = hungarian_matching(
            uav_pos, self.cluster_center
        )
        sorted_matching_result, sorted_total_cost = sorted_matching(
            uav_energy, self.data_list, uav_pos, self.cluster_center
        )
        greedy_total_cost, greedy_kl = self.match_dis_cal(greedy_matching_result)
        hungarian_total_cost, hungarian_kl = self.match_dis_cal(
            hungarian_matching_result
        )
        sorted_total_cost, sorted_kl = self.match_dis_cal(sorted_matching_result)
        pointer_total_cost, pointer_kl = self.match_dis_cal(luav_act)
        # 计算kl散度

        uav_name = f"match"
        data = pd.DataFrame(
            {
                "greedy_match": [greedy_matching_result],
                "greedy_cost": [sum(greedy_total_cost)],
                "greedy_kl": [greedy_kl],
                "hungarian_match": [hungarian_matching_result],
                "hungarian_cost": [sum(hungarian_total_cost)],
                "hungarian_kl": [hungarian_kl],
                "sorted_match": [sorted_matching_result],
                "sorted_cost": [sum(sorted_total_cost)],
                "sorted_kl": [sorted_kl],
                "pointer_match": [luav_act],
                "pointer_cost": [sum(pointer_total_cost)],
                "pointer_kl": [pointer_kl],
            }
        )

        if uav_name in self.uav_data:
            self.uav_data[uav_name] = pd.concat(
                [self.uav_data[uav_name], data], ignore_index=True
            )
        else:
            self.uav_data[uav_name] = data

    def reset(self):
        """Returns initial observations and states"""
        """ env_init """
        """
        环境初始化: 环境中luav, fuav列表、sn列表
        """
        # 清理环境在上一回合的历史信息（如果有的话）
        self.luav_list.clear()
        self.fuav_list.clear()
        self.kl_list.clear()
        self.r_tp_list.clear()
        self.slot = 0
        self.step_c = 0
        self.uav_data = {}
        random.seed(self.cfg.seed)

        # 清除sn的数据包信息
        for sn in self.sn_list:
            sn.collect = False
            sn.packet = [0, 0]
            sn.l_aoi.clear()

        # 环境中的luav、fuav列表初始化
        for i in range(self.luav_num):
            init_pos = self.luav_init_pos_list[i]
            new_luav = LUAVNode(id=i, env_cfg=self.cfg, init_pos=init_pos)

            for j in range(self.fuav_num):
                init_pos_f = self.fuav_init_pos_list[j]
                new_fuav = FUAVNode(
                    id=j, luav_id=i, env_cfg=self.cfg, init_pos_abs=init_pos_f
                )
                new_fuav.data_add(self.uav_data)
                self.fuav_list.append(new_fuav)
                new_luav.fuav_list.append(new_fuav.id)

            new_luav.fuav_num = len(new_luav.fuav_list)
            # new_luav.data_add(self.uav_data)
            self.luav_list.append(new_luav)

        for sn in self.sn_list:
            self.sn_update(sn.id)
            sn.data_add(self.uav_data, self.slot, self.step_c, self.episode_limit)

    # -------------- INTERACTION METHODS --------------
    def faction_convert(self, id, l):
        # sn = int(self.cfg.sn_num / self.fuav_num)
        # act_list = np.zeros(sn)
        # assert id < self.get_f_actions()
        # act_list[id] = 1

        act_list = np.zeros(l)
        assert id < l
        act_list[id] = 1
        return act_list

    def match_dis_cal(self, match_result):
        assert len(match_result) == self.fuav_num
        energy_list = []

        # kl散度
        dtotal = 0
        E_total = 0
        kl_DE = 0
        for sn in self.sn_list:
            dtotal += sn.packet[0]
        # for uav in self.fuav_list:
        #     E_total += max(0.01, uav.energy)

        for i, c_id in enumerate(match_result):
            dis = math.sqrt(
                (self.fuav_list[i].pos_abs[0] - self.cluster_center[c_id][0]) ** 2
                + (self.fuav_list[i].pos_abs[1] - self.cluster_center[c_id][1]) ** 2
            )
            f_cost = self.ecost(dis)
            energy_list.append(f_cost)
            E_total += max(0.01, (self.fuav_list[i].energy - f_cost))

        for i in range(self.fuav_num):
            k = match_result[i]
            dn = self.data_list[k]
            if dn <= 0:
                dn = 0.0001
            En = self.fuav_list[i].energy - energy_list[i]
            if En <= 0:
                En = 0.01

            # print("match")
            # print(dn)
            # print(dtotal)
            # print(En)
            # print(E_total)
            if dtotal <= 0:
                print(dtotal)
            if E_total <= 0:
                print(E_total)
            kl_DE += dn / dtotal * math.log((dn / dtotal) / (En / E_total))
            # kl_DE += En / E_total * math.log((En / E_total) / (dn / dtotal))

        return energy_list, kl_DE

    def ecost(self, x):
        c_1 = self.cfg.flight_c_1
        c_2 = self.cfg.flight_c_2
        c_3 = self.cfg.flight_c_3
        energy_cost = (
            self.flight_cost + c_1 * x + c_2 * math.pow(x, 2) + c_3 * math.pow(x, 3)
        )
        return energy_cost

    def get_flight_reward(self):
        # 飞行能耗 kl散度
        # 能耗 与速度/距离相关？ 存距离
        E_cost = 0  # ？
        E_list = []
        for uav in self.fuav_list:
            dist = uav.data_v
            e = self.ecost(dist)
            E_list.append(e)
            E_cost += e

        # kl散度
        dtotal = 0
        E_total = 0
        kl_DE = 0
        kl_DE_1 = 0
        r_TP = 0
        for sn in self.sn_list:
            dtotal += sn.packet[0]
        for uav in self.fuav_list:
            E_total += max(0.01, uav.energy)

        for uav in self.fuav_list:
            dn = 0
            for i in range(len(uav.sn_list)):
                c_id = uav.sn_list[i]
                dn += self.sn_list[c_id].packet[0]
            En = uav.energy
            if dn == 0:
                dn = 0.0001
            if En <= 0:
                En = 0.01
            # print("reward")
            # print(dn)
            # print(dtotal)
            # print(En)
            # print(E_total)
            kl_DE += dn / dtotal * math.log((dn / dtotal) / (En / E_total))
            kl_DE_1 += En / E_total * math.log((En / E_total) / (dn / dtotal))
        # print(kl_DE)
        E_cost = E_cost / self.cfg.normalize_energy

        if self.cfg.obj == "energy":
            if self.cfg.obj_type == "exp":
                r_TP = math.exp(-E_cost)
            elif self.cfg.obj_type == "linear":
                r_TP = 1 - E_cost
            elif self.cfg.obj_type == "non_linear":
                r_TP = 1 / E_cost
            else:
                print("能量奖励函数类型错误")
                exit()
        elif self.cfg.obj == "kl":
            if self.cfg.obj_type == "exp":
                r_TP = math.exp(-kl_DE * 10)
            elif self.cfg.obj_type == "linear":
                r_TP = 1 - kl_DE * 10
            else:
                print("KL散度奖励函数类型错误")
                exit()
        elif self.cfg.obj == "all":
            r_TP = math.exp(-E_cost - kl_DE)
        else:
            print("优化目标错误")
            exit()

        self.kl_list.append(kl_DE)
        self.r_tp_list.append(r_TP)
        return (r_TP, E_cost, kl_DE)

    def get_collect_reward(self, end):
        # 收集数据量 AR 结束
        r_RA = []
        for uav in self.fuav_list:
            r1 = 0
            r2 = 0
            r3 = 0
            r = 0
            # 收集数据量
            if uav.work:
                r1 = uav.data_v

            # AR /S
            for i in range(len(uav.act)):
                if uav.act[i]:
                    q_id = uav.sn_list[i]
                    # q_data = self.sn_list[q_id].packet[0]
                    q_aoi = self.sn_list[q_id].packet[1]
                    if uav.work:
                        r2 = (
                            (self.slot_step_num + 1 - self.step_c)
                            * (2 * q_aoi + self.step_c / self.slot_step_num + 1)
                            / 2
                        )

            # end time step
            if end:
                for sn in uav.sn_list:
                    sn_aoi = self.sn_list[sn].packet[1]
                    r3 += math.exp(-sn_aoi)

            r = r1 + r2 + r3
            uav.sub_reward = [r1, r2, r3]
            uav.reward = r
            r_RA.append(r)

        return r_RA

    def step(self, control_step, actions, new=None):
        """Returns reward, terminated, info"""

        flag = 0  # slot结束标志
        l_reward = 0
        f_reward = []
        freward_total = 0

        # print(f"luav slot {self.slot} run")
        # print(f"luav step {self.step_c} run")
        assert control_step == self.step_c

        # 清理
        if self.step_c == 0:  # 第一个step
            self.env_slot_clear()
        else:
            self.env_step_clear()

        if self.cfg.large_timescale:
            if self.step_c == 0:  # 第一个step
                luavaction = self.laction_convert(actions)
                self.lact = luavaction

                for luav in self.luav_list:
                    luav.action = self.lact
                    luav.act_id = actions

                    luav.act_check(self.cell_map)
                    if not luav.act_legal:
                        # luav.action = (0.0, 0.0)
                        self.lact = (0.0, 0.0)
                        # print(f"luav target false\n")
                        luav.slot_reward = -5.0

                    step_dir, step_dis = self.lact[0], self.lact[1]

                    new_pos = (
                        round(luav.pos[0] + math.cos(step_dir) * step_dis),
                        round(luav.pos[1] + math.sin(step_dir) * step_dis),
                        luav.pos[2],
                    )
                    if (
                        new_pos[0] < 0
                        or new_pos[1] < 0
                        or new_pos[0] > self.map_length
                        or new_pos[1] > self.map_width
                    ):
                        print("luav pos error")
                        raise RuntimeError("luav pos error")

                    if self.cell_map[new_pos[0]][new_pos[1]].obs == 1 and step_dis != 0:
                        print("luav pos obs error")
                        raise RuntimeError("luav pos obs error")

                    self.slot_target = new_pos[:2]

                    luav.pos = new_pos
                    luav.dis_total += step_dis
                    luav.step += 1  # step好像没用

                    # 更新fuav相对位置
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        fuav_x, fuav_y = fuav.pos_abs[0], fuav.pos_abs[1]
                        fuav.pos_rela = (fuav_x - luav.pos[0], fuav_y - luav.pos[1])

                    flag = 1  # slot结束
                    luav.slot += 1
                    l_reward = self.get_luav_reward()

                    # luav.data_add(env_uav_data=self.uav_data)

                    # 检查fuav是否在luav范围内，给予奖励
                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0

        else:
            if self.step_c == 0:  # 第0个frame 飞行
                # 动作：n个索引，对应新位置和连接的SN的id
                # 更新fuav位置
                # 更新fuav连接的SN列表
                # 更新fuav能量：飞行
                luavaction = actions[0]

                # 匹配算法对比！
                # self.match(luavaction)

                for luav in self.luav_list:
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        c_id = luavaction[fuav_id]
                        dist = math.sqrt(
                            (fuav.pos_abs[0] - self.cluster_center[c_id][0]) ** 2
                            + (fuav.pos_abs[1] - self.cluster_center[c_id][1]) ** 2
                        )
                        fuav.data_v = dist
                        fuav.pos_abs = self.cluster_center[c_id]
                        fuav.cluster = c_id
                        fuav.sn_list = self.cluster_list[c_id]
                        # fuav.energy -= self.flight_cost * dist
                        fuav.energy -= self.ecost(dist)
                        if fuav.energy < 0:
                            fuav.energy = 0

                    flag = 0  # slot进行中
                    freward_total = 0
                    self.step_c += 1
                    f_reward = 0
                    l_reward, sub1, sub2 = self.get_flight_reward()
                    # luav.data_add(env_uav_data=self.uav_data)
                    for fuav in self.fuav_list:
                        fuav.l_reward = l_reward
                        fuav.sub_reward = [sub1, sub2]
                        fuav.data_add(env_uav_data=self.uav_data)

            elif self.step_c > 0:  # 第1~k个frame 收集数据
                # 动作：n个无人机与连接列表内SN的关联（收集）q  n*sn
                # 收集数据：判断SN是否收集完，更新SN数据包信息和收集状态
                # 更新能量：悬停（+收集）
                fuavaction = actions[0]
                for luav in self.luav_list:
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        l = len(fuav.sn_list)
                        fact = self.faction_convert(fuavaction[fuav_id], l)

                        fuav.frame_run(
                            act=fact,
                            env_uav_data=self.uav_data,
                            env_sn_list=self.sn_list,
                        )

                # 判断slot是否结束
                if self.step_c == self.slot_step_num:  # 最后一个frame
                    # 更新SN数据包信息：所有数据包非0的信息年龄+1，为0的生成新数据包

                    # print(f"current step = {self.step_c}")
                    # print(f"slot{self.slot} run over\n")
                    flag = 1  # slot结束
                    luav.slot += 1

                    for sn in self.sn_list:
                        sn.data_add(
                            self.uav_data, self.slot, self.step_c, self.episode_limit
                        )

                    self.sn_slot_update()
                    l_reward = 0
                    f_reward = self.get_collect_reward(1)
                    for fuav in self.fuav_list:
                        freward_total += fuav.reward
                        fuav.data_add(env_uav_data=self.uav_data)

                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0
                else:
                    flag = 0  # slot进行中
                    l_reward = 0
                    f_reward = self.get_collect_reward(0)

                    for fuav in self.fuav_list:
                        freward_total += fuav.reward
                        fuav.data_add(env_uav_data=self.uav_data)
                    for sn in self.sn_list:
                        sn.data_add(
                            self.uav_data, self.slot, self.step_c, self.episode_limit
                        )
                    self.step_c += 1
            else:
                # print("输入错误")
                assert 0

        terminated = self.slot == self.episode_limit
        env_info = self.get_env_info(new)

        return (l_reward, f_reward, freward_total, flag, terminated, env_info)

    def tp_test_recoed(self, t_env: int = None, path: str = None):
        path = Path(path)
        csv_path = f"{path}/{self.cfg.obj}_{self.cfg.obj_type}.csv"

        fuav_eng_list = []
        for fuav in self.fuav_list:
            fuav_eng_list.append(fuav.energy)

        row = [
            t_env,
            np.mean(self.kl_list),
            np.mean(fuav_eng_list),
            np.mean(self.r_tp_list),
            self.kl_list,
            fuav_eng_list,
            self.r_tp_list,
        ]
        # 检查文件是否存在，如果不存在则写入表头
        file_exists = False
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                file_exists = True
        except FileNotFoundError:
            file_exists = False

        # 打开 CSV 文件并追加新的数据
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 如果文件不存在，则写入表头
            if not file_exists:
                writer.writerow(
                    [
                        "t_env",
                        "kl",
                        "energy",
                        "tp_reward",
                        "kl_list",
                        "energy_list",
                        "tp_reward_list",
                    ]
                )  # 写入表头

            # 写入新的数据行
            writer.writerow(row)

    def record(self, t_env: int = None, path: str = None):
        """
        记录环境运行产生的数据
        """
        if path is None:
            path = Path.cwd() / "record" / "TPRA" / str(os.getppid()) / "data"
            if t_env is None:
                cur_time = datetime.now().strftime("%m_%d_%H_%M_%S")
                path = path / cur_time
            # else:
            #     path = path / str(t_env)
            # save_dir = Path.cwd() / "record" / "TPRA" / str(os.getppid()) / "fig"
        else:
            path = Path(path)
            # save_dir = path / "fig"
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        excel_file = f"{path}/{t_env}.xlsx"
        data_save.save_data_to_excel(self, excel_file)
        self.tp_test_recoed(t_env, path)

        print(f"数据已保存到 '{excel_file}'")

        # data_file = pd.read_excel(excel_file, sheet_name=None)
        # data_map = data_file["map"].to_numpy()

        # self.trajectory_alive(data_file, save_dir, t_env)

    # -------------- OBSERVATION METHODS --------------
    def plot_static(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.set_xlim(0, self.map_length)
        ax.set_aspect("equal")

        obstacle_list = np.array(self.cfg.obstacle_list, dtype=object)
        ue_pos = np.array(self.cfg.ue_cluster_center_list[0], dtype=object)
        ue_pos = ue_pos[0]
        # plt.scatter(ue_pos[:, 0], ue_pos[:, 1], c="gray", s=10, label="UE")
        ax.scatter(
            ue_pos[0] + 0.5,
            ue_pos[1] + 0.5,
            c="gray",
            s=10,
            label="UE",
        )
        # ax.scatter(luav_pos[0] + 0.5, luav_pos[1] + 0.5, c="gray", s=10)

        for pos, a, b in obstacle_list:
            x, y = pos[0], pos[1]
            points = [[x, y], [x + a, y], [x + a, y + b], [x, y + b], [x, y]]
            points = np.array(points).transpose()
            ax.fill(points[0], points[1], c="black", linestyle="-", linewidth=0.3)

        return fig, ax

    def trajectory_alive(self, data: pd.DataFrame, save_dir_uav, data_slot):
        if not os.path.exists(save_dir_uav):
            os.makedirs(save_dir_uav)

        fig, ax = self.plot_static()

        for luav_id in range(self.luav_num):
            x, y = [], []
            luav_data = data[f"luav{luav_id}"]
            for i in range(luav_data.shape[0]):
                pos = eval(luav_data["pos"][i])[:2]
                color = uav_color_list[-1]
                x.append(pos[0] + 0.5)
                y.append(pos[1] + 0.5)
                # c.append(color)
            ax.plot(x, y, color=color, linestyle="-", linewidth=1)

        fuav_alive = np.ones(self.fuav_num)

        for fuav_id in range(self.fuav_num):
            x, y, c = [], [], []
            fuav_data = data[f"fuav{fuav_id}"]

            for i in range(fuav_data.shape[0]):
                # fuav_data = fuav_data.iloc[i]
                pos = eval(fuav_data["pos_abs"][i])[:2]
                color = uav_color_list[fuav_id] if fuav_data["formation"][i] else "grey"
                x.append(pos[0] + 0.5)
                y.append(pos[1] + 0.5)
                if not fuav_data["formation"][i]:
                    fuav_alive[fuav_id] = 0
                    # print(i)
                    # print(f"fuav id:{fuav_id}")
                    # print(fuav_data["pos_abs"][i])
                    # print(fuav_data["pos_rela"][i])
                    break
            ax.plot(x, y, color=color, linestyle=":", linewidth=0.5)

        luav_data_f = luav_data.iloc[-1]
        x, y = eval(luav_data_f["pos"])[:2]

        color = uav_color_list[-1]

        ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="10")

        for fuav in range(self.fuav_num):
            if fuav_alive[fuav]:
                fuav_data = data[f"fuav{fuav}"]
                fuav_data = fuav_data.iloc[-1]
                x, y = eval(fuav_data["pos_abs"])[:2]
                color = uav_color_list[fuav]
                ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

        luav_data_s = luav_data.iloc[0]
        x, y = eval(luav_data_s["pos"])[:2]
        color = uav_color_list[-1]
        ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="10")

        for fuav in range(self.fuav_num):
            fuav_data = data[f"fuav{fuav}"]
            fuav_data = fuav_data.iloc[0]
            x, y = eval(fuav_data["pos_abs"])[:2]
            color = uav_color_list[fuav]
            ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

        plt.savefig(f"{save_dir_uav}/{data_slot}.pdf")
        plt.close(fig)
        print(f"Finish plotting alive trajectory.")

    def trajectory(self, data: pd.DataFrame, save_dir_uav, data_slot):
        if not os.path.exists(save_dir_uav):
            os.makedirs(save_dir_uav)

        fig, ax = self.plot_static()

        for luav_id in range(self.luav_num):
            x, y = [], []
            luav_data = data[f"luav{luav_id}"]
            for i in range(luav_data.shape[0]):
                pos = eval(luav_data["pos"][i])[:2]
                color = uav_color_list[-1]
                x.append(pos[0] + 0.5)
                y.append(pos[1] + 0.5)
            ax.plot(x, y, color=color, linestyle="-", linewidth=1)

        for fuav_id in range(self.fuav_num):
            x, y, c = [], [], []
            fuav_data = data[f"fuav{fuav_id}"]

            for i in range(fuav_data.shape[0]):
                # fuav_data = fuav_data.iloc[i]
                pos = eval(fuav_data["pos_abs"][i])[:2]
                color = uav_color_list[fuav_id] if fuav_data["formation"][i] else "grey"
                x.append(pos[0] + 0.5)
                y.append(pos[1] + 0.5)
            ax.plot(x, y, color=color, linestyle=":", linewidth=0.5)

        luav_data_f = luav_data.iloc[-1]
        x, y = eval(luav_data_f["pos"])[:2]
        color = uav_color_list[-1]
        ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="10")

        for fuav in range(self.fuav_num):
            fuav_data = data[f"fuav{fuav}"]
            fuav_data = fuav_data.iloc[-1]
            x, y = eval(fuav_data["pos_abs"])[:2]
            color = uav_color_list[fuav]
            ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

        luav_data_s = luav_data.iloc[0]
        x, y = eval(luav_data_s["pos"])[:2]
        color = uav_color_list[-1]
        ax.plot([x + 0.5], [y + 0.5], c=color, marker="*", markersize="10")

        for fuav in range(self.fuav_num):
            fuav_data = data[f"fuav{fuav}"]
            fuav_data = fuav_data.iloc[0]
            x, y = eval(fuav_data["pos_abs"])[:2]
            color = uav_color_list[fuav]
            ax.plot([x + 0.5], [y + 0.5], c=color, marker="^")

        plt.savefig(f"{save_dir_uav}/{data_slot}.pdf")
        plt.close(fig)
        print(f"Finish plotting trajectory.")

    def get_luav_obs(self):
        # 静态s：聚类位置 数据量 aoi
        # 动态d：匹配cov、A 无人机位置 能量
        luav_s_observation = []
        luav_d_observation = []
        sn_c = int(len(self.sn_list) / self.fuav_num)
        for cluster in self.bkm_clusters:
            luav_s_observation.append(cluster.center[0] / self.map_length)
        for cluster in self.bkm_clusters:
            luav_s_observation.append(cluster.center[1] / self.map_width)
        for cluster in self.bkm_clusters:
            luav_s_observation.append(cluster.get_data() / (sn_c * 10))
        for cluster in self.bkm_clusters:
            luav_s_observation.append(cluster.get_aoi() / (sn_c * self.episode_limit))

        for luav in self.luav_list:
            for fuav_id in luav.fuav_list:
                uav = self.fuav_list[fuav_id]
                luav_d_observation.append(uav.pos_abs[0] / self.map_length)
        for luav in self.luav_list:
            for fuav_id in luav.fuav_list:
                uav = self.fuav_list[fuav_id]
                luav_d_observation.append(uav.pos_abs[1] / self.map_width)
        for luav in self.luav_list:
            for fuav_id in luav.fuav_list:
                uav = self.fuav_list[fuav_id]
                luav_d_observation.append(max(0, uav.energy) / self.init_energy)
        for luav in self.luav_list:
            for fuav_id in luav.fuav_list:
                uav = self.fuav_list[fuav_id]
                luav_d_observation.append(0)
                luav_d_observation.append(0)

        luav_s_observation.extend(luav_d_observation)
        assert len(luav_s_observation) == self.get_l_obs_size()
        return luav_s_observation

    def get_luav_obs_new(self):
        # 静态s：聚类位置 数据量 aoi 无人机位置 能量
        # 动态d：匹配cov、A
        luav_s_observation = []
        luav_d_observation = []
        sn_c = int(len(self.sn_list) / self.fuav_num)
        for cluster in self.bkm_clusters:
            luav_s_observation.append(cluster.center[0] / self.map_length)
        for cluster in self.bkm_clusters:
            luav_s_observation.append(cluster.center[1] / self.map_width)
        # for cluster in self.bkm_clusters:
        #     luav_s_observation.append(cluster.get_data() / (sn_c * 10))
        for cluster in self.bkm_clusters:
            luav_s_observation.append(cluster.get_aoi() / (sn_c * self.episode_limit))
        for luav in self.luav_list:
            for fuav_id in luav.fuav_list:
                uav = self.fuav_list[fuav_id]
                luav_s_observation.append(uav.pos_abs[0] / self.map_length)
        for luav in self.luav_list:
            for fuav_id in luav.fuav_list:
                uav = self.fuav_list[fuav_id]
                luav_s_observation.append(uav.pos_abs[1] / self.map_width)
        for luav in self.luav_list:
            for fuav_id in luav.fuav_list:
                uav = self.fuav_list[fuav_id]
                luav_s_observation.append(max(0, uav.energy) / self.init_energy)

        luav_d_observation.extend(
            [1] * sum(len(luav.fuav_list) for luav in self.luav_list)
        )
        luav_d_observation.extend(
            [0] * sum(len(luav.fuav_list) for luav in self.luav_list)
        )

        luav_s_observation.extend(luav_d_observation)
        assert len(luav_s_observation) == self.get_l_obs_size_new()
        return luav_s_observation

    def get_l_obs_size(self):
        return 9 * self.fuav_num

    def get_l_s_obs_size(self):
        return 4 * self.fuav_num

    def get_l_d_obs_size(self):
        return 5 * self.fuav_num

    def get_l_obs_size_new(self):
        return 8 * self.fuav_num

    def get_l_s_obs_size_new(self):
        return 6 * self.fuav_num

    def get_l_d_obs_size_new(self):
        return 2 * self.fuav_num

    def get_fuav_obs(self):
        # 无人机能量，子目标
        # SN位置，数据量，aoi，aipv
        fuav_full_observation = []
        for fuav in self.fuav_list:
            fuav_obs = []
            fuav_obs = fuav.get_observation(self.sn_list)

            # 确认fuav observation 的形状为 a
            a = 3 + 5 * len(self.sn_list) / self.fuav_num
            assert a == len(fuav_obs)

            fuav_full_observation.append(fuav_obs)

        return np.array(fuav_full_observation)

    def get_f_obs_size(self):
        # return (self.fuav_num * 3 + self.cfg.sn_num * 5)
        return 3 + int(5 * len(self.sn_list) / self.fuav_num)

    def env_nodes_obs(self, l, f):
        if l:
            # 获取领航无人机信息
            return self.get_luav_obs()
        if f:
            #  获取跟随无人机信息
            return self.get_fuav_obs()

    def get_state(self):
        self.state = np.ones(1)

        return self.state

    def get_state_size(self):
        """Returns the shape of the state"""
        return 1

    def get_l_avail_actions(self):
        return np.ones(self.get_l_actions())

    def get_f_avail_actions(self):
        return np.ones(shape=(len(self.fuav_list), self.get_f_actions()))

    def get_l_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.fuav_num
        # return math.factorial(self.fuav_num)

    def get_f_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return int(len(self.sn_list) / self.fuav_num)

    def render(self):
        raise NotImplementedError

    def close(self):
        # TODO!
        pass

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self, new=None):
        if new:
            env_info = {
                "state_shape": self.get_state_size(),
                "l_obs_shape": self.get_l_obs_size_new(),
                "l_s_obs_shape": self.get_l_s_obs_size_new(),
                "l_d_obs_shape": self.get_l_d_obs_size_new(),
                "l_n_actions": self.get_l_actions(),
                "l_n_agents": self.l_n_agents,
                "episode_limit": self.episode_limit,
            }
        else:
            env_info = {
                "state_shape": self.get_state_size(),
                "l_obs_shape": self.get_l_obs_size(),
                "l_s_obs_shape": self.get_l_s_obs_size(),
                "l_d_obs_shape": self.get_l_d_obs_size(),
                "f_obs_shape": self.get_f_obs_size(),
                "l_n_actions": self.get_l_actions(),
                "f_n_actions": self.get_f_actions(),
                "l_n_agents": self.l_n_agents,
                "f_n_agents": self.f_n_agents,
                "episode_limit": self.episode_limit,
            }
        return env_info

    def greedy_data_fuav(self, fuav):
        sn_data_list = []
        for sn_id in fuav.sn_list:
            sn = self.sn_list[sn_id]
            sn_data_list.append(sn.packet[0])
        return sn_data_list.index(max(sn_data_list))

    def ra(self, slot_step):
        assert slot_step == self.step_c

        fuav_actions = []
        for fuav in self.fuav_list:
            fuav_actions.append(self.greedy_data_fuav(fuav))

        self.step(slot_step, [fuav_actions])

    def tp(self, slot_step):

        uav_pos = []
        uav_energy = []
        for uav in self.fuav_list:
            uav_pos.append(uav.pos_abs)
            uav_energy.append(max(0, uav.energy))

        greedy_matching_result, greedy_total_cost = greedy_matching(
            uav_pos, self.cluster_center
        )
        self.step(slot_step, [greedy_matching_result])
