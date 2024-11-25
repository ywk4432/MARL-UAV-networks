#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt

# from data_save import *
from . import data_save
from .element import *
from ..multiagentenv import MultiAgentEnv

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


class LUAVEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        cfg = SimpleNamespace(**kwargs)
        self.name = cfg.env_name
        self.luav_num = cfg.luav_num
        self.fuav_num = cfg.fuav_num

        self.l_n_agents = cfg.luav_num
        self.f_n_agents = cfg.fuav_num

        # self.map_size = cfg.map_size
        self.map_length = cfg.map_length  # 行数 50
        self.map_width = cfg.map_width  # 列数 20

        # self.obstacle_list = cfg.obstacle_list
        self.obstacle_list = [Obstocal(*obs) for obs in cfg.obstacle_list]

        self.slot = -1
        self.step_c = -1
        self.slot_step_num = cfg.slot_step_num

        self.luav_list = []
        self.fuav_list = []
        self.luav_init_pos_list = cfg.luav_init_pos_list
        self.obs_size = cfg.luav_observation_size

        self.luav_obs_size = 2 * cfg.luav_observation_size + 1
        self.fuav_obs_size = 2 * cfg.fuav_observation_size + 1

        # 地面用户分布生成
        # self.ue_cluster_center_list = cfg.ue_cluster_center_list
        self.ue_cluster_center_list = [
            UECluster(*target) for target in cfg.ue_cluster_center_list
        ]
        self.ue_init_pos_list = []

        self.fuav_acts = cfg.fuav_acts  # 9维

        self.FUAV_mecost = cfg.FUAV_mecost
        self.LUAV_mecost = cfg.LUAV_mecost

        self.cell_map = []
        self.cfg = cfg
        self.uav_data = {}  # 存储所有无人机在各时隙的信息

        self.done = False

        self.lact = (0.0, 0.0)
        self.slot_target = (0, 0)

        self.episode_limit = cfg.episode_limit
        self.state = []

        self.fuav_init_pos_rela = cfg.fuav_init_pos_list

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

    def generate_pos_around_center(self, center, radius, n_points):
        # 解包圆心坐标
        cx, cy = center

        # 生成 n 个随机距离，距离圆心越近的点距离越小
        distances = np.random.uniform(0, radius, n_points)

        # 生成 n 个随机角度（弧度）
        angles = np.random.uniform(0, 2 * np.pi, n_points)

        # 将极坐标转换为二维坐标
        x_coords = cx + distances * np.cos(angles)
        y_coords = cy + distances * np.sin(angles)

        # 将坐标四舍五入为整数
        integer_points = np.column_stack(
            (np.round(x_coords).astype(int), np.round(y_coords).astype(int))
        )

        return integer_points

    def cell_map_init(self):
        map_size = (self.map_length, self.map_width)
        self.cell_map = [
            [CellNode(id=i + j) for i in range(map_size[1])] for j in range(map_size[0])
        ]

    def ue_cluster_init(self):
        """
        在center_pos里的各中心点附近生成ue集群，同时更新cell_map的信息
        """
        # 生成数据簇
        for ue_center in self.ue_cluster_center_list:
            # 设置中心点
            ue_poses = self.generate_pos_around_center(
                center=ue_center.pos, radius=ue_center.range, n_points=ue_center.ue_num
            )
            for ue_pos in ue_poses:
                self.cell_map[ue_pos[0]][ue_pos[1]].ue_num += 1
            self.cell_map[ue_center.pos[0]][
                ue_center.pos[1]
            ].ue_num = self.cfg.max_ue_num_in_a_cell

            # 引力
            tx, ty = ue_center.pos[0], ue_center.pos[1]
            for i in range(self.map_length):
                for j in range(self.map_width):
                    dis = math.sqrt((i - tx) ** 2 + (j - ty) ** 2)
                    if dis != 0:
                        self.cell_map[i][j].apf = 100 / dis
                    else:
                        self.cell_map[i][j].apf = 105
                #     print(round(self.cell_map[i][j].apf,2),end=' ')
                # print(i)

    def obstacle_init(self):
        """
        生成障碍物
        """
        # 生成数据簇
        for obstacle in self.obstacle_list:
            obs_pos, obs_x, obs_y = obstacle.pos, obstacle.x, obstacle.y
            for x in range(obs_pos[0], obs_pos[0] + obs_x):
                for y in range(obs_pos[1], obs_pos[1] + obs_y):
                    self.cell_map[x][y].obs = True
                    self.cell_map[x][y].apf -= 5
            # 斥力
            obs_range = round((obs_x + obs_y) / 2)
            if obs_range < 1:
                obs_range = 1
            for n in range(1, obs_range + 1):
                for x in range(obs_pos[0] - n, obs_pos[0] + obs_x + n):
                    for y in range(obs_pos[1] - n, obs_pos[1] + obs_y + n):
                        if not (
                            x < 0
                            or x >= self.map_length
                            or y < 0
                            or y >= self.map_width
                        ):
                            self.cell_map[x][y].apf -= 0.2 / n

            # for x in range(obs_pos[0]- 1, obs_pos[0] + obs_x + 1):
            #     for y in range(obs_pos[1] - 1, obs_pos[1] + obs_y + 1):
            #         if not (
            #             x < 0
            #             or x >= self.map_length
            #             or y < 0
            #             or y >= self.map_width
            #         ):
            #             self.cell_map[x][y].apf -= 1

        # for i in range(self.map_length):
        #     for j in range(self.map_width):
        #         print(self.cell_map[i][j].apf , end=' ')
        #     print(i)

    def uav_inf_init(self):
        """
        归零无人机位置
        """
        for i in range(self.map_length):
            for j in range(self.map_width):
                self.cell_map[i][j].uav_inf = ("n", 0)

    def uav_update(self):
        """
        更新无人机位置
        """
        # for luav in self.luav_list:
        #     luav_x, luav_y = luav.pos[0], luav.pos[1]
        #     new_luav_inf = ("l", 1)
        #     self.cell_map[luav_x][luav_y].uav_inf = new_luav_inf
        for fuav in self.fuav_list:
            fuav_x, fuav_y = fuav.pos_abs[0], fuav.pos_abs[1]
            # if self.cell_map[fuav_x][fuav_y].uav_inf[1] != 0: # 发生无人机碰撞
            #   设置 fuav.alive = False 或者 new_fuav_inf = ("？", 2)
            new_fuav_inf = ("f", 1)
            self.cell_map[fuav_x][fuav_y].uav_inf = new_fuav_inf

    def reset(self):
        """Returns initial observations and states"""
        """ env_init """
        """
        环境初始化: 环境中luav, fuav列表、ue列表、地图的创建
        """
        # 清理环境在上一回合的历史信息（如果有的话）
        self.luav_list.clear()
        self.fuav_list.clear()
        self.slot = 0
        self.step_c = 0
        self.uav_data = {}
        # self.fuav_init_pos_rela = []

        self.cell_map_init()
        self.ue_cluster_init()
        self.obstacle_init()
        self.uav_inf_init()

        # 环境中的luav、fuav列表初始化
        for i in range(self.luav_num):
            init_pos = self.luav_init_pos_list[i]
            target = self.ue_cluster_center_list[i]
            new_luav = LUAVNode(
                id=i, env_cfg=self.cfg, init_pos=init_pos, target_pos=target.pos
            )

            for j in range(self.fuav_num):
                init_pos_rela = self.fuav_init_pos_rela[j]
                # self.fuav_init_pos_rela.append(init_pos_rela)

                new_fuav = FUAVNode(
                    id=j,
                    luav_id=i,  # 领航无人机、跟随无人机相互建立连接关系
                    env_cfg=self.cfg,
                    env_acts=self.fuav_acts,
                    init_pos_abs=init_pos,
                    init_pos_rela=init_pos_rela,
                    target_pos=target.pos,
                )
                new_fuav.data_add(self.uav_data)
                # 领航无人机、跟随无人机相互建立连接关系
                self.fuav_list.append(new_fuav)
                new_luav.fuav_list.append(new_fuav.id)

            new_luav.fuav_num = len(new_luav.fuav_list)
            new_luav.data_add(self.uav_data)
            self.luav_list.append(new_luav)

        self.uav_update()

    # -------------- INTERACTION METHODS --------------
    def laction_convert(self, id):
        id = id[0]
        assert id <= self.get_l_actions()
        act_list = [(0, 0)]
        for dis in self.cfg.dis[1:]:
            for dir in self.cfg.dir:
                act = (dir, dis)
                act_list.append(act)

        return act_list[id]

    def get_luav_reward(self):
        for luav in self.luav_list:
            dis = math.sqrt(
                (luav.pos[0] - luav.target_pos[0]) ** 2
                + (luav.pos[1] - luav.target_pos[1]) ** 2
            )
            if dis != 0:
                r1 = 10 / dis
                r2 = 0
            else:
                r1 = 15
                r2 = 30 / luav.dis_total
            # r1 到目标距离
            r3 = luav.slot_reward  # 非法动作 -5
            # r4 fuav死亡数

        luav_reward = r1 + r2 + r3
        # luav_reward = r2 + r3
        luav.sub_reward = [r1, r2, r3]
        luav.reward = luav_reward

        return luav_reward

    def get_freward_total(self):
        freward_total = 0.0
        for fuav in self.fuav_list:
            dis = math.sqrt(
                (fuav.pos_abs[0] - fuav.act_rela[0]) ** 2
                + (fuav.pos_abs[1] - fuav.act_rela[1]) ** 2
            )
            if dis != 0:
                r1 = 5 / dis
                r3 = 0
            else:
                r1 = 6
                r3 = 1

            # r1 到目标点距离 2 or 1/dis
            r2 = fuav.step_reward  # 非法动作 -5

            r4 = fuav.reward_slot_end  # 抵达范围 3 or -5
            r = r1 + r2 + r3 + r4
            fuav.sub_reward = [r1, r2, r3, r4]
            fuav.reward = r
            freward_total += r

        for fuav in self.fuav_list:
            fuav.reward_total = freward_total

        return freward_total

    def get_freward(self):
        freward_total = 0
        for ue_center in self.ue_cluster_center_list:
            ue_x, ue_y = ue_center.pos[0], ue_center.pos[1]
        for fuav in self.fuav_list:
            target_x = ue_x + fuav.init_rela[0]
            target_y = ue_y + fuav.init_rela[1]
            assert target_x == fuav.target_pos[0]
            assert target_y == fuav.target_pos[1]

            dis = math.sqrt(
                (fuav.pos_abs[0] - target_x) ** 2 + (fuav.pos_abs[1] - target_y) ** 2
            )
            if dis != 0:
                r1 = 30 / dis
                r3 = 0
            else:
                r1 = 31
                r3 = 20 / self.slot  # 没统计总距离

            r2 = fuav.step_reward  # 非法动作 -5

            r = r1 + r2 + r3
            fuav.sub_reward = [r1, r2, r3]
            fuav.reward = r
            freward_total += r

        for fuav in self.fuav_list:
            fuav.reward_total = freward_total

        return freward_total

    def select_target(self):
        poses_rela_able = []
        for i in range(-self.cfg.luav_connect_dis, self.cfg.luav_connect_dis):
            for j in range(-self.cfg.luav_connect_dis, self.cfg.luav_connect_dis):
                pos_x = i + self.slot_target[0]
                pos_y = j + self.slot_target[1]
                if not (
                    pos_x < 0
                    or pos_x >= self.map_length
                    or pos_y < 0
                    or pos_y >= self.map_width
                ):

                    if not (self.cell_map[pos_x][pos_y].obs):
                        poses_rela_able.append((i, j))

        for fuav in self.fuav_list:
            fuav.luav_pos_abs = self.slot_target
            point = self.fuav_init_pos_rela[fuav.id]
            dist = lambda x: math.sqrt((x[0] - point[0]) ** 2 + (x[1] - point[1]) ** 2)
            match = min(poses_rela_able, key=dist)
            rela_x, rela_y = (
                self.slot_target[0] + match[0],
                self.slot_target[1] + match[1],
            )
            fuav.act_rela = (rela_x, rela_y)
            poses_rela_able.remove(match)

    def step(self, control_step, uav_type, actions):
        """Returns reward, terminated, info"""

        flag = 0
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
            if self.step_c == 0 and uav_type == "l":  # 第一个step
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

                    luav.data_add(env_uav_data=self.uav_data)

                    # 检查fuav是否在luav范围内，给予奖励
                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0

        else:
            if self.step_c == 0 and uav_type == "l":  # 第一个step
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

                    # l_x, l_y = luav.pos[0], luav.pos[1]
                    # if self.cell_map[l_x][l_y].uav_inf[0] == "l":
                    #     self.cell_map[l_x][l_y].uav_inf = ("n", 0)

                    # l_x_n, l_y_n = new_pos[0], new_pos[1]
                    # self.cell_map[l_x_n][l_y_n].uav_inf = ("l", 1)

                    self.slot_target = new_pos[:2]

                    luav.pos = new_pos
                    luav.dis_total += step_dis
                    luav.step += 1  # step好像没用
                    # print(f"luav slot target {luav.pos}")

                    # 更新fuav相对位置
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        fuav_x, fuav_y = fuav.pos_abs[0], fuav.pos_abs[1]
                        fuav.pos_rela = (fuav_x - luav.pos[0], fuav_y - luav.pos[1])

                    # 给fuav选择合适目标点
                    self.select_target()

                    flag = 0  # slot进行中
                    l_reward = 0
                    freward_total = 0
                    self.step_c += 1
                    # f_reward

            elif self.step_c > 0 and uav_type == "f":
                fuavaction = actions
                for luav in self.luav_list:
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        # 给各跟随无人机配置绝对动作

                        # 以下部分放入select_target，选择合理目标点act_rela
                        # fuav.luav_pos_abs = self.slot_target
                        # rela_x, rela_y = (
                        #     self.slot_target[0] + self.fuav_init_pos_rela[fuav_id][0],
                        #     self.slot_target[1] + self.fuav_init_pos_rela[fuav_id][1],
                        # )
                        # fuav.act_rela = (rela_x, rela_y)

                        f_x, f_y = fuav.pos_abs[0], fuav.pos_abs[1]
                        if self.cell_map[f_x][f_y].uav_inf[0] == "f":
                            self.cell_map[f_x][f_y].uav_inf = ("n", 0)
                        else:
                            print("cell map false")
                            raise RuntimeError("cell map false")

                        fuav.step_run(
                            env_cell_map=self.cell_map,
                            env_luav_list=self.luav_list,
                            act=fuavaction[fuav_id],
                            env_uav_data=self.uav_data,
                        )
                        # f_reward.append(fuav.step_reward)

                        # 更新uav位置
                        # self.uav_inf_init()
                        # self.uav_update()
                        f_x_n, f_y_n = fuav.pos_abs[0], fuav.pos_abs[1]
                        self.cell_map[f_x_n][f_y_n].uav_inf = ("f", 1)

                # 判断slot是否结束
                if self.step_c == self.slot_step_num:
                    # print(f"current step = {self.step_c}")
                    # print(f"slot{self.slot} run over\n")
                    flag = 1  # slot结束
                    luav.slot += 1
                    luav.energy_update()
                    l_reward = self.get_luav_reward()
                    luav.check_fuav_list(self.fuav_list)

                    freward_total = self.get_freward_total()

                    for fuav in self.fuav_list:
                        fuav.data_add(env_uav_data=self.uav_data)
                        f_reward.append(fuav.reward)

                    luav.data_add(env_uav_data=self.uav_data)

                    # 更新接入的跟随无人机列表
                    luav.update_fuav_list(self.fuav_list, self.cell_map)

                    # self.uav_inf_init()
                    # self.uav_update()

                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0
                else:
                    flag = 0  # slot进行中
                    l_reward = 0
                    freward_total = self.get_freward_total()
                    for fuav in self.fuav_list:
                        fuav.data_add(env_uav_data=self.uav_data)
                        f_reward.append(fuav.reward)
                    self.step_c += 1
            else:
                # print("输入错误")
                assert 0

        terminated = self.slot == self.episode_limit
        env_info = self.get_env_info()

        return (l_reward, f_reward, freward_total, flag, terminated, env_info)

    def step_dmtd(self, control_step, uav_type, actions):
        """Returns reward, terminated, info"""

        flag = 0
        l_reward = 0
        f_reward = []
        freward_total = 0
        assert control_step == self.step_c

        # 清理
        if self.step_c == 0:  # 第一个step
            self.env_slot_clear()
        else:
            self.env_step_clear()

        if self.cfg.large_timescale:
            if self.step_c == 0 and uav_type == "l":  # 第一个step
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

                    luav.data_add(env_uav_data=self.uav_data)

                    # 检查fuav是否在luav范围内，给予奖励
                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0

        else:
            if self.step_c == 0 and uav_type == "l":  # 第一个step
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
                    luav.step += 1

                    # 更新fuav相对位置
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        fuav_x, fuav_y = fuav.pos_abs[0], fuav.pos_abs[1]
                        fuav.pos_rela = (fuav_x - luav.pos[0], fuav_y - luav.pos[1])

                    flag = 0  # slot进行中
                    l_reward = 0
                    freward_total = 0
                    self.step_c += 1

            elif self.step_c > 0 and uav_type == "f":
                fuavaction = actions
                for luav in self.luav_list:
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        fuav.luav_pos_abs = self.slot_target
                        rela_x, rela_y = (
                            self.slot_target[0] + self.fuav_init_pos_rela[fuav_id][0],
                            self.slot_target[1] + self.fuav_init_pos_rela[fuav_id][1],
                        )
                        fuav.act_rela = (rela_x, rela_y)

                        f_x, f_y = fuav.pos_abs[0], fuav.pos_abs[1]
                        if self.cell_map[f_x][f_y].uav_inf[0] == "f":
                            self.cell_map[f_x][f_y].uav_inf = ("n", 0)
                        else:
                            print("cell map false")
                            raise RuntimeError("cell map false")

                        fuav.step_run(
                            env_cell_map=self.cell_map,
                            env_luav_list=self.luav_list,
                            act=fuavaction[fuav_id],
                            env_uav_data=self.uav_data,
                        )

                        f_x_n, f_y_n = fuav.pos_abs[0], fuav.pos_abs[1]
                        self.cell_map[f_x_n][f_y_n].uav_inf = ("f", 1)

                # 判断slot是否结束
                if self.step_c == self.slot_step_num:
                    flag = 1  # slot结束
                    luav.slot += 1
                    luav.energy_update()
                    l_reward = self.get_luav_reward()

                    freward_total = self.get_freward()

                    for fuav in self.fuav_list:
                        fuav.data_add(env_uav_data=self.uav_data)
                        f_reward.append(fuav.reward)

                    luav.data_add(env_uav_data=self.uav_data)

                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0
                else:
                    flag = 0  # slot进行中
                    l_reward = 0
                    freward_total = self.get_freward()
                    for fuav in self.fuav_list:
                        fuav.data_add(env_uav_data=self.uav_data)
                        f_reward.append(fuav.reward)
                    self.step_c += 1
            else:
                # print("输入错误")
                assert 0

        terminated = self.slot == self.episode_limit
        env_info = self.get_env_info()

        return (l_reward, f_reward, freward_total, flag, terminated, env_info)

    def step_unselect(self, control_step, uav_type, actions):
        """Returns reward, terminated, info"""

        flag = 0
        l_reward = 0
        f_reward = []
        freward_total = 0

        assert control_step == self.step_c

        # 清理
        if self.step_c == 0:  # 第一个step
            self.env_slot_clear()
        else:
            self.env_step_clear()

        if self.cfg.large_timescale:
            if self.step_c == 0 and uav_type == "l":  # 第一个step
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

                    luav.data_add(env_uav_data=self.uav_data)

                    # 检查fuav是否在luav范围内，给予奖励
                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0

        else:
            if self.step_c == 0 and uav_type == "l":  # 第一个step
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
                    # print(f"luav slot target {luav.pos}")

                    # 更新fuav相对位置
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]
                        fuav_x, fuav_y = fuav.pos_abs[0], fuav.pos_abs[1]
                        fuav.pos_rela = (fuav_x - luav.pos[0], fuav_y - luav.pos[1])

                    flag = 0  # slot进行中
                    l_reward = 0
                    freward_total = 0
                    self.step_c += 1
                    # f_reward

            elif self.step_c > 0 and uav_type == "f":
                fuavaction = actions
                for luav in self.luav_list:
                    for fuav_id in luav.fuav_list:
                        fuav = self.fuav_list[fuav_id]

                        fuav.luav_pos_abs = self.slot_target
                        rela_x, rela_y = (
                            self.slot_target[0] + self.fuav_init_pos_rela[fuav_id][0],
                            self.slot_target[1] + self.fuav_init_pos_rela[fuav_id][1],
                        )
                        fuav.act_rela = (rela_x, rela_y)

                        f_x, f_y = fuav.pos_abs[0], fuav.pos_abs[1]
                        if self.cell_map[f_x][f_y].uav_inf[0] == "f":
                            self.cell_map[f_x][f_y].uav_inf = ("n", 0)
                        else:
                            print("cell map false")
                            raise RuntimeError("cell map false")

                        fuav.step_run(
                            env_cell_map=self.cell_map,
                            env_luav_list=self.luav_list,
                            act=fuavaction[fuav_id],
                            env_uav_data=self.uav_data,
                        )
                        # f_reward.append(fuav.step_reward)

                        # 更新uav位置
                        # self.uav_inf_init()
                        # self.uav_update()
                        f_x_n, f_y_n = fuav.pos_abs[0], fuav.pos_abs[1]
                        self.cell_map[f_x_n][f_y_n].uav_inf = ("f", 1)

                # 判断slot是否结束
                if self.step_c == self.slot_step_num:
                    # print(f"current step = {self.step_c}")
                    # print(f"slot{self.slot} run over\n")
                    flag = 1  # slot结束
                    luav.slot += 1
                    luav.energy_update()
                    l_reward = self.get_luav_reward()
                    luav.check_fuav_list(self.fuav_list)

                    freward_total = self.get_freward_total()

                    for fuav in self.fuav_list:
                        fuav.data_add(env_uav_data=self.uav_data)
                        f_reward.append(fuav.reward)

                    luav.data_add(env_uav_data=self.uav_data)

                    # 更新接入的跟随无人机列表
                    luav.update_fuav_list(self.fuav_list, self.cell_map)

                    # self.uav_inf_init()
                    # self.uav_update()

                    self.slot += 1  # 进入新一个slot
                    self.step_c = 0
                else:
                    flag = 0  # slot进行中
                    l_reward = 0
                    freward_total = self.get_freward_total()
                    for fuav in self.fuav_list:
                        fuav.data_add(env_uav_data=self.uav_data)
                        f_reward.append(fuav.reward)
                    self.step_c += 1
            else:
                # print("输入错误")
                assert 0

        terminated = self.slot == self.episode_limit
        env_info = self.get_env_info()

        return (l_reward, f_reward, freward_total, flag, terminated, env_info)

    def record(self, t_env: int = None, path: str = None):
        """
        记录环境运行产生的数据
        """
        if path is None:
            path = Path.cwd() / "record" / "formation" / str(os.getppid()) / "data"
            if t_env is None:
                cur_time = datetime.now().strftime("%m_%d_%H_%M_%S")
                path = path / cur_time
            # else:
            #     path = path / str(t_env)
            save_dir = Path.cwd() / "record" / "formation" / str(os.getppid()) / "fig"
        else:
            path = Path(path)
            save_dir = path / "fig"
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        excel_file = f"{path}/{t_env}.xlsx"
        data_save.save_data_to_excel(self, excel_file)
        print(f"数据已保存到 '{excel_file}'")

        data_file = pd.read_excel(excel_file, sheet_name=None)
        data_map = data_file["map"].to_numpy()

        self.trajectory_alive(data_file, save_dir, t_env)

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
        for luav in self.luav_list:
            luav.get_observation(self.cell_map)
            # 展开数组
            obs_obs_flat = luav.observation_obs.flatten()
            obs_uav_flat = luav.observation_uav.flatten()
            obs_ue_flat = luav.observation_ue.flatten()

            # 将展开后的数组连接成一个 luav_observation 数组
            luav_observation = np.concatenate((obs_obs_flat, obs_uav_flat, obs_ue_flat))

            # 确认 luav_observation 的形状为 a=3n^2
            a = 3 * self.luav_obs_size**2
            assert a == len(luav_observation)
            if self.cfg.onehot:
                lpos_x = np.zeros(2 * self.map_length + 1)
                lpos_y = np.zeros(2 * self.map_width + 1)
                lpos_x[luav.pos[0] - luav.target_pos[0] + self.map_length] = 1
                lpos_y[luav.pos[1] - luav.target_pos[1] + self.map_width] = 1
                luav_observation = np.concatenate((luav_observation, lpos_x, lpos_y))
            else:
                luav_observation = np.append(
                    luav_observation,
                    values=(luav.pos[0] - luav.target_pos[0]) / self.map_length,
                )
                luav_observation = np.append(
                    luav_observation,
                    values=(luav.pos[1] - luav.target_pos[1]) / self.map_width,
                )
            # luav_observation = np.append(luav_observation, values=luav.energy)

        assert len(luav_observation) == self.get_l_obs_size()
        return luav_observation

    def get_l_obs_size(self):
        return (
            3 * self.luav_obs_size**2 + 2 * self.map_length + 2 * self.map_width + 2
            if self.cfg.onehot
            else 3 * self.luav_obs_size**2 + 2
        )

    def get_fuav_obs(self):
        fuav_full_observation = []
        for fuav in self.fuav_list:
            fuav_obs = []
            fuav.get_observation(self.cell_map)
            # 展开数组
            obs_flat = fuav.observation_obs.flatten()
            uav_flat = fuav.observation_uav.flatten()

            # 将展开后的数组连接成一个 luav_extra_observation 数组
            fuav_obs = np.concatenate((obs_flat, uav_flat))

            # 确认 luav_extra_observation 的形状为 a=3n^2
            a = 2 * self.fuav_obs_size**2
            assert a == len(fuav_obs)

            f_rela_x = fuav.pos_rela[0] - self.fuav_init_pos_rela[fuav.id][0]
            f_rela_y = fuav.pos_rela[1] - self.fuav_init_pos_rela[fuav.id][1]
            l = (
                2 * self.cfg.luav_connect_dis
                + self.cfg.slot_step_num
                + self.cfg.dis[-1]
            )

            # if self.cfg.onehot:
            #     # # 相对位置
            #     # fpos_x = np.zeros(2 * (self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1]) + 1)
            #     # fpos_y = np.zeros(2 * (self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1]) + 1)
            #     # # print(f"fuav_id{fuav.id}, {fuav.pos_rela[0]}, {fuav.pos_rela[1]}")

            #     # if(fuav.pos_rela[0] + self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1] < 0
            #     #    or fuav.pos_rela[0] > self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1]
            #     #    or fuav.pos_rela[1] + self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1] < 0
            #     #    or fuav.pos_rela[1] > self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1]
            #     #    ):
            #     #     print("pos_rela false")
            #     #     print(f"fuav.pos_rela:{fuav.pos_rela}")

            #     # fpos_x[int(fuav.pos_rela[0] + self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1])] = 1
            #     # fpos_y[int(fuav.pos_rela[1] + self.cfg.luav_connect_dis + self.cfg.slot_step_num + self.cfg.dis[-1])] = 1
            #     # fuav_obs = np.concatenate((fuav_obs, fpos_x, fpos_y))

            #     # # 初始相对位置
            #     # fpos_t_x = np.zeros(2 * self.cfg.luav_connect_dis + 1)
            #     # fpos_t_y = np.zeros(2 * self.cfg.luav_connect_dis + 1)
            #     # fpos_t_x[
            #     #     int(self.fuav_init_pos_rela[fuav.id][0] + self.cfg.luav_connect_dis)
            #     # ] = 1
            #     # fpos_t_y[
            #     #     int(self.fuav_init_pos_rela[fuav.id][1] + self.cfg.luav_connect_dis)
            #     # ] = 1
            #     # fuav_obs = np.concatenate((fuav_obs, fpos_t_x, fpos_t_y))

            #     # 相对位置之差
            #     fpos_x = np.zeros(
            #         2
            #         * (
            #             2 * self.cfg.luav_connect_dis
            #             + self.cfg.slot_step_num
            #             + self.cfg.dis[-1]
            #         )
            #         + 1
            #     )
            #     fpos_y = np.zeros(
            #         2
            #         * (
            #             2 * self.cfg.luav_connect_dis
            #             + self.cfg.slot_step_num
            #             + self.cfg.dis[-1]
            #         )
            #         + 1
            #     )

            #     if (
            #         f_rela_x
            #         + 2 * self.cfg.luav_connect_dis
            #         + self.cfg.slot_step_num
            #         + self.cfg.dis[-1]
            #         < 0
            #         or f_rela_x
            #         > 2 * self.cfg.luav_connect_dis
            #         + self.cfg.slot_step_num
            #         + self.cfg.dis[-1]
            #         or f_rela_y
            #         + 2 * self.cfg.luav_connect_dis
            #         + self.cfg.slot_step_num
            #         + self.cfg.dis[-1]
            #         < 0
            #         or f_rela_y
            #         > 2 * self.cfg.luav_connect_dis
            #         + self.cfg.slot_step_num
            #         + self.cfg.dis[-1]
            #     ):
            #         print("pos_rela false")
            #         print(
            #             f"fuav.pos_rela:{fuav.pos_rela} init_pos_rela:{self.fuav_init_pos_rela[fuav.id]}"
            #         )

            #     fpos_x[
            #         int(
            #             f_rela_x
            #             + 2 * self.cfg.luav_connect_dis
            #             + self.cfg.slot_step_num
            #             + self.cfg.dis[-1]
            #         )
            #     ] = 1
            #     fpos_y[
            #         int(
            #             f_rela_y
            #             + 2 * self.cfg.luav_connect_dis
            #             + self.cfg.slot_step_num
            #             + self.cfg.dis[-1]
            #         )
            #     ] = 1
            #     fuav_obs = np.concatenate((fuav_obs, fpos_x, fpos_y))

            # else:
            #     # fuav_obs = np.append(fuav_obs, values=fuav.pos_rela[0])
            #     # fuav_obs = np.append(fuav_obs, values=fuav.pos_rela[1])
            #     # fuav_obs = np.append(
            #     #     fuav_obs, values=self.fuav_init_pos_rela[fuav.id][0]
            #     # )
            #     # fuav_obs = np.append(
            #     #     fuav_obs, values=self.fuav_init_pos_rela[fuav.id][1]
            #     # )
            #     fuav_obs = np.append(fuav_obs, values=f_rela_x / l)
            #     fuav_obs = np.append(fuav_obs, values=f_rela_y / l)

            fuav_obs = np.append(
                fuav_obs,
                values=(fuav.pos_abs[0] - fuav.target_pos[0]) / self.map_length,
            )
            fuav_obs = np.append(
                fuav_obs, values=(fuav.pos_abs[1] - fuav.target_pos[1]) / self.map_width
            )

            # fuav_obs = np.append(fuav_obs, values=fuav.energy)

            fuav_full_observation.append(fuav_obs)

        return np.array(fuav_full_observation)

    def get_f_obs_size(self):
        # return (
        #     3 * self.fuav_obs_size**2 + 8 * self.cfg.luav_connect_dis + 4 * self.cfg.slot_step_num + 4 * self.cfg.dis[-1] + 4
        #     if self.cfg.onehot
        #     else 3 * self.fuav_obs_size**2 + 4
        # )
        return (
            2 * self.fuav_obs_size**2
            + 8 * self.cfg.luav_connect_dis
            + 4 * self.cfg.slot_step_num
            + 4 * self.cfg.dis[-1]
            + 4
            if self.cfg.onehot
            else 2 * self.fuav_obs_size**2 + 2
        )

    def env_nodes_obs(self, l, f):
        if l:
            # 获取领航无人机信息
            return self.get_luav_obs()
        if f:
            #  获取跟随无人机信息
            return self.get_fuav_obs()

    def get_state(self):
        uav_pos = []
        for luav in self.luav_list:
            luav_x, luav_y = luav.pos[0], luav.pos[1]
            uav_pos.append(luav_x)
            uav_pos.append(luav_y)
        for fuav in self.fuav_list:
            fuav_x, fuav_y = fuav.pos_abs[0], fuav.pos_abs[1]
            uav_pos.append(fuav_x)
            uav_pos.append(fuav_y)

        obs_pos = []
        for obs in self.obstacle_list:
            pos_x, pos_y, x, y = obs.pos[0], obs.pos[1], obs.x, obs.y
            obs_pos.append(pos_x)
            obs_pos.append(pos_y)
            obs_pos.append(x)
            obs_pos.append(y)

        target_pos = []
        for target in self.ue_cluster_center_list:
            pos_x, pos_y = target.pos[0], target.pos[1]
            target_pos.append(pos_x)
            target_pos.append(pos_y)

        self.state = np.array(uav_pos + obs_pos + target_pos)

        return self.state

    def get_state_size(self):
        """Returns the shape of the state"""
        return 2 * (
            self.luav_num + self.fuav_num + len(self.ue_cluster_center_list)
        ) + 4 * len(self.obstacle_list)

    def get_l_avail_actions(self):
        return np.ones(self.get_l_actions())

    def get_f_avail_actions(self):
        return np.ones(shape=(len(self.fuav_list), self.get_f_actions()))

    def get_l_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return (len(self.cfg.dis) - 1) * len(self.cfg.dir) + 1

    def get_f_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return len(self.fuav_acts)

    def render(self):
        raise NotImplementedError

    def close(self):
        # TODO!
        pass

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "l_obs_shape": self.get_l_obs_size(),
            "f_obs_shape": self.get_f_obs_size(),
            "l_n_actions": self.get_l_actions(),
            "f_n_actions": self.get_f_actions(),
            "l_n_agents": self.l_n_agents,
            "f_n_agents": self.f_n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info
