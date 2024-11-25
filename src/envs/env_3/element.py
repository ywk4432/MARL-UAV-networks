"""
 # @ Author: Wenke
 # @ Create Time: 2023-09-18 11:34:09
 # @ Modified by: Wenke
 # @ Modified time: 2023-09-19 06:47:49
 # @ Description: 系统中各类元素：用户集群、障碍物、地面cell格、领航无人机、跟随无人机
 """

import math

from . import supple

PI = math.pi

"""
 # @ Author: Wenke
 # @ Create Time: 2023-09-18 11:34:09
 # @ Modified by: Wenke
 # @ Modified time: 2023-09-19 06:47:49
 # @ Description: 系统中各类元素：用户集群、障碍物、地面cell格、领航无人机、跟随无人机
 """

import copy

import pandas as pd

from .supple import *

PI = math.pi


class UECluster:
    def __init__(self, init_pos, range, ue_num) -> None:
        self.pos = tuple(init_pos)
        self.range = range
        self.ue_num = ue_num


class Obstocal:
    def __init__(self, init_pos, x, y) -> None:
        self.pos = init_pos
        self.x = x
        self.y = y


class CellNode:
    def __init__(self, id=-1, init_pos=(0, 0), env_cfg=None, elements=None):
        if elements is None:
            elements = [
                0,
                0,
                ("n", 0),
            ]  # [用户、障碍物、无人机信息(无人机类型，无人机数量)]
        self.id = id
        self.ue_num = elements[0]
        self.obs = elements[1]
        self.uav_inf = elements[2]
        self.apf = 0


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
        self.energy = env_cfg.luav_init_energy

        self.luav_connect_dis = env_cfg.luav_connect_dis
        self.fuav_observation_size = env_cfg.fuav_observation_size
        self.fuav_list = []  # 只保存每个slot保持连接的跟随无人机的ID
        self.fuav_num = 0  # 每个slot所连接跟随无人机的数量
        self.fuav_absact_kill_num = 0  # 每个slot因领航无人机的绝对动作造成的跟随无人机消，只会在跟随无人机的非法相对动作时发生

        self.target_pos = target_pos
        self.slot_reward = 0.0
        self.reward = 0.0
        self.sub_reward = []

        self.luav_observation_size = env_cfg.luav_observation_size
        self.observation_obs = np.zeros(
            (2 * self.luav_observation_size + 1, 2 * self.luav_observation_size + 1)
        )  # 局部观测 - 障碍物
        self.observation_uav = np.zeros(
            (2 * self.luav_observation_size + 1, 2 * self.luav_observation_size + 1)
        )  # 局部观测 - 无人机
        self.observation_ue = np.zeros(
            (2 * self.luav_observation_size + 1, 2 * self.luav_observation_size + 1)
        )  # 局部观测 - 地面用户
        self.extra_observation_obs = np.zeros(
            (
                2 * (self.luav_connect_dis + self.fuav_observation_size) + 1,
                2 * (self.luav_connect_dis + self.fuav_observation_size) + 1,
            )
        )  # 扩展局部观测 - 障碍物
        self.extra_observation_uav = np.zeros(
            (
                2 * (self.luav_connect_dis + self.fuav_observation_size) + 1,
                2 * (self.luav_connect_dis + self.fuav_observation_size) + 1,
            )
        )  # 扩展局部观测 - 无人机
        self.extra_observation_ue = np.zeros(
            (
                2 * (self.luav_connect_dis + self.fuav_observation_size) + 1,
                2 * (self.luav_connect_dis + self.fuav_observation_size) + 1,
            )
        )  # 扩展局部观测 - 地面用户

        self.env_cfg = env_cfg
        self.agent = None
        self.dis_total = 0

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

    def get_extra_observation(self, env_fuav_list):
        self.extra_observation_obs[:] = 0
        self.extra_observation_uav[:] = 0
        self.extra_observation_ue[:] = 0

        for i in range(-self.luav_observation_size, self.luav_observation_size):
            for j in range(-self.luav_observation_size, self.luav_observation_size):
                self.extra_observation_obs[i][j] = self.observation_obs[i][j]
                self.extra_observation_uav[i][j] = self.observation_uav[i][j]
                self.extra_observation_ue[i][j] = self.observation_ue[i][j]

        for fuav_id in self.fuav_list:
            fuav = env_fuav_list[fuav_id]
            for i in range(-fuav.fuav_observation_size, fuav.fuav_observation_size):
                for j in range(-fuav.fuav_observation_size, fuav.fuav_observation_size):
                    x, y = int(fuav.pos_rela[0] + i), int(fuav.pos_rela[1] + j)
                    self.extra_observation_obs[x][y] = fuav.observation_obs[i][j]
                    self.extra_observation_uav[x][y] = fuav.observation_uav[i][j]
                    self.extra_observation_ue[x][y] = fuav.observation_ue[i][j]

    def check_fuav_list(self, env_fuav_list):
        self.fuav_list.clear()
        for fuav_id, fuav in enumerate(env_fuav_list):
            if (
                fuav.alive
                and abs(fuav.pos_abs[0] - self.pos[0]) <= self.luav_connect_dis
                and abs(fuav.pos_abs[1] - self.pos[1]) <= self.luav_connect_dis
            ):
                fuav.reward_slot_end = 3
                self.fuav_list.append(fuav_id)
            else:
                # 给跟随无人机惩罚
                fuav.reward_slot_end = -5
        self.fuav_num = len(self.fuav_list)

    def update_fuav_list(self, env_fuav_list, env_cell_map):
        self.fuav_list.clear()
        poses_abs_able = []
        poses_rela_able = []
        self.get_observation(env_cell_map)
        fuav_pos_z = env_fuav_list[0].pos_abs[2]
        for i in range(-self.luav_observation_size, self.luav_observation_size):
            for j in range(-self.luav_observation_size, self.luav_observation_size):
                index_i = i + self.luav_observation_size
                index_j = j + self.luav_observation_size
                if not (
                    self.observation_obs[index_i][index_j]
                    or self.observation_uav[index_i][index_j]
                ):
                    pos_x, pos_y = self.pos[0] + i, self.pos[1] + j
                    if not (
                        pos_x < 0
                        or pos_x >= self.env_cfg.map_length
                        or pos_y < 0
                        or pos_y >= self.env_cfg.map_width
                    ):
                        poses_abs_able.append(
                            (self.pos[0] + i, self.pos[1] + j, fuav_pos_z)
                        )
                        poses_rela_able.append((i, j, fuav_pos_z))
        for fuav_id, fuav in enumerate(env_fuav_list):
            if fuav.reward_slot_end == -5:
                f_x, f_y = fuav.pos_abs[0], fuav.pos_abs[1]
                if env_cell_map[f_x][f_y].uav_inf[0] == "f":
                    env_cell_map[f_x][f_y].uav_inf = ("n", 0)
                else:
                    print("update_fuav_list cell map false")
                    raise RuntimeError("update_fuav_list cell map false")

                # 给新位置
                fuav.pos_abs = poses_abs_able[fuav_id]
                fuav.pos_rela = poses_rela_able[fuav_id]
                x, y = fuav.pos_abs[0], fuav.pos_abs[1]
                env_cell_map[x][y].uav_inf = ("f", 1)

                if (
                    x < 0
                    or x >= self.env_cfg.map_length
                    or y < 0
                    or y >= self.env_cfg.map_width
                ):
                    print("拉取位置错误")

            self.fuav_list.append(fuav_id)
        self.fuav_num = len(self.fuav_list)

    def clear(self):
        # self.fuav_list.clear()
        self.action = (0, 0)
        self.fuav_absact_kill_num = 0
        self.slot_reward = 0
        self.reward = 0
        self.act_id = 0

    def energy_update(self):
        """更新无人机能耗"""
        self.slot_ecost = self.env_cfg.LUAV_mecost * self.action[1]
        self.energy = self.energy - self.slot_ecost

    def act_check(self, env_cell_map):
        """
        只负责检查领航无人机动作的合法性，不执行
        """
        self.act_legal = True

        dir, dis = self.action[0], self.action[1]
        new_end = (
            round(self.pos[0] + math.cos(dir) * dis),
            round(self.pos[1] + math.sin(dir) * dis),
            self.pos[2],
        )
        pos_x, pos_y = new_end[0], new_end[1]
        pos_z = self.pos[2]

        if (
            pos_x < self.env_cfg.safe_dis
            or pos_x >= self.env_cfg.map_length - self.env_cfg.safe_dis
            or pos_y < self.env_cfg.safe_dis
            or pos_y >= self.env_cfg.map_width - self.env_cfg.safe_dis
            or pos_z not in self.env_cfg.alts
        ):
            # print("luav即将飞出边界 -> illegal action 0")
            self.act_legal = False

        else:
            if env_cell_map[pos_x][pos_y].obs == 1:
                # print("luav即将前往区域存在障碍物 -> illegal action 3")
                self.act_legal = False
            # if env_cell_map[pos_x][pos_y].uav_inf[1] > 0:
            #     # print(f"luav {self.id} 即将前往区域存在其他fuav -> illegal action 2")
            #     self.act_legal = False

        new_poses = supple.ac_detect(self.pos, self.action)

        for new_pos in new_poses:
            # pos_x, pos_y, pos_z = new_pos[0], new_pos[1], new_pos[2]
            pos_x, pos_y = round(new_pos[0]), round(new_pos[1])
            pos_z = self.pos[2]

            if (
                pos_x < 0
                or pos_x >= self.env_cfg.map_length
                or pos_y < 0
                or pos_y >= self.env_cfg.map_width
                or pos_z not in self.env_cfg.alts
            ):
                # print("luav即将飞出边界 -> illegal action 0")
                self.act_legal = False

            else:
                if env_cell_map[pos_x][pos_y].obs == 1:
                    # print("luav即将前往区域存在障碍物 -> illegal action 3")
                    self.act_legal = False
                # if env_cell_map[pos_x][pos_y].uav_inf[1] > 0:
                #     # print(f"luav {self.id} 即将前往区域存在其他fuav -> illegal action 2")
                #     self.act_legal = False

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
                "act_legal": [self.act_legal],
                "slot_reward": [self.slot_reward],
                "reward": [self.reward],
                "sub_reward": [copy.deepcopy(self.sub_reward)],
                "fuav_list": [copy.deepcopy(self.fuav_list)],
                "fuav_num": [self.fuav_num],
                "fuav_absact_kill_num": [self.fuav_absact_kill_num],
            }
        )
        if uav_name in env_uav_data:
            env_uav_data[uav_name] = pd.concat(
                [env_uav_data[uav_name], data], ignore_index=True
            )
        else:
            env_uav_data[uav_name] = data

    def act_make(
        self, action, env_cell_map, env_fuav_list, env_luav_list, env_uav_data
    ):
        """
        每个slot起始，luav做绝对飞行控制决策，并在该slot的steps里，控制跟随无人机执行
        """
        self.action = action
        self.act_check(env_cell_map)
        if not self.act_legal:
            self.action = (0.0, 0.0)
            self.slot_reward = -5.0

        # 如果领航无人机动作合法，则执行，否则直接返回负的奖励
        for s in range(self.slot_step_num):
            self.step_run()
            if s != self.slot_step_num - 1:
                self.data_add(env_uav_data=env_uav_data)
            for fuav_id in self.fuav_list:
                fuav = env_fuav_list[fuav_id]
                # 给各跟随无人机配置绝对动作
                fuav.luav_pos_abs = self.action
                # 产生各跟随无人机配置相对动作
                fuav_action = 1
                # fuav_action = np.random.randint(0, 9)
                fuav.step_run(
                    env_cell_map=env_cell_map,
                    env_luav_list=env_luav_list,
                    act=fuav_action,
                    env_uav_data=env_uav_data,
                )

        self.energy_update()

        # 更新接入的跟随无人机列表
        self.update_fuav_list(env_fuav_list)
        self.get_observation(env_cell_map)
        self.get_extra_observation(env_fuav_list)
        self.data_add(env_uav_data=env_uav_data)


class FUAVNode:
    def __init__(
        self,
        id=-1,
        luav_id=-1,
        init_pos_abs=(0, 0, 0),
        init_pos_rela=(0, 0),
        env_cfg=None,
        env_acts=None,
        target_pos=(0, 0),
    ):
        if env_acts is None:
            env_acts = []
        self.id = id
        self.alive = True
        self.pos_abs = (
            init_pos_abs[0] + init_pos_rela[0],
            init_pos_abs[1] + init_pos_rela[1],
            init_pos_abs[2],
        )  # 绝对位置坐标 - 相对于整个地图坐标系
        self.pos_rela = init_pos_rela  # 相对位置坐标 - 相对于领航无人机
        self.move_dis_abs = 0  # 绝对移动距离
        self.move_dis_rela = 0  # 相对移动距离
        self.slot = 0  # slot数
        self.step = 0  # step数

        self.fuav_nei_dis = env_cfg.fuav_nei_dis
        self.nei_uav_list = []
        self.nei_uav_connect = True
        self.nei_uav_num = 0

        self.luav_connect = True
        self.luav_id = luav_id

        self.luav_pos_abs = (0, 0)  # 领航无人机的目标点
        self.act_rela = (0, 0)  # 跟随无人机相对位置目标点
        self.act_abs_legal = True  # 领航无人机动作的合法性
        self.act = 0  # 无人机的动作
        self.act_legal = True  # 无人机的动作检查
        self.energy = env_cfg.uav_init_energy  # 无人机可用储能
        self.step_ecost = 0.0  # 单个step内耗能

        self.state_size = env_cfg.lagent_state_size
        self.obs_size = env_cfg.lagent_obs_size
        self.act_dim = env_cfg.lagent_act_dim

        self.fuav_observation_size = env_cfg.fuav_observation_size
        self.observation_obs = np.zeros(
            (2 * self.fuav_observation_size + 1, 2 * self.fuav_observation_size + 1)
        )  # 局部观测 - 障碍物
        self.observation_uav = np.zeros(
            (2 * self.fuav_observation_size + 1, 2 * self.fuav_observation_size + 1)
        )  # 局部观测 - 无人机
        self.observation_ue = np.zeros(
            (2 * self.fuav_observation_size + 1, 2 * self.fuav_observation_size + 1)
        )  # 局部观测 - 地面用户
        self.state = np.array([0.0 for _ in range(self.state_size)])
        self.next_state = np.array([0.0 for _ in range(self.state_size)])
        self.step_reward = 0  # 无人机在单个step内的动作合法奖励值
        self.reward_slot_end = 0  # 无人机在slot中是否抵达目标位置的奖励值

        self.env_cfg = env_cfg  # 存储系统性参数
        self.agent = None

        self.init_rela = init_pos_rela
        self.formation = True

        self.reward_total = 0.0
        self.reward = 0.0
        self.sub_reward = []
        self.target_pos = (
            target_pos[0] + init_pos_rela[0],
            target_pos[1] + init_pos_rela[1],
        )

    def data_add(self, env_uav_data):
        uav_name = f"fuav{self.id}"
        data = pd.DataFrame(
            {
                "ID": [self.id],
                "slot": [self.slot],
                "step": [self.step],
                "pos_abs": [self.pos_abs],
                "pos_rela": [self.pos_rela],
                "init_pos_rela": [self.init_rela],
                "luav_pos_abs": [self.luav_pos_abs],
                "goal_abs_pos": [copy.deepcopy(self.act_rela)],
                "action": [self.act],
                "act_legal": [self.act_legal],
                "alive": [self.alive],
                "formation": [self.formation],
                "reward_total": [self.reward_total],
                "reward": [self.reward],
                "sub_reward": [copy.deepcopy(self.sub_reward)],
            }
        )

        if uav_name in env_uav_data:
            env_uav_data[uav_name] = pd.concat(
                [env_uav_data[uav_name], data], ignore_index=True
            )
        else:
            env_uav_data[uav_name] = data

    def get_observation(self, env_cell_map):
        self.observation_obs[:] = 0
        self.observation_uav[:] = 0
        self.observation_ue[:] = 0

        for i in range(-self.fuav_observation_size, self.fuav_observation_size):
            for j in range(-self.fuav_observation_size, self.fuav_observation_size):
                index_i = i + self.fuav_observation_size
                index_j = j + self.fuav_observation_size
                if (
                    self.pos_abs[0] + i < 0
                    or self.pos_abs[0] + i >= self.env_cfg.map_length
                    or self.pos_abs[1] + j < 0
                    or self.pos_abs[1] + j >= self.env_cfg.map_width
                ):
                    self.observation_obs[index_i][index_j] = -1
                    self.observation_uav[index_i][index_j] = -1
                    self.observation_ue[index_i][index_j] = -1
                else:
                    self.observation_obs[index_i][index_j] = env_cell_map[
                        self.pos_abs[0] + i
                    ][self.pos_abs[1] + j].obs
                    self.observation_uav[index_i][index_j] = env_cell_map[
                        self.pos_abs[0] + i
                    ][self.pos_abs[1] + j].uav_inf[1]
                    self.observation_ue[index_i][index_j] = env_cell_map[
                        self.pos_abs[0] + i
                    ][self.pos_abs[1] + j].ue_num

    def clear(self):
        """
        每个step开始都要进行信息清理
        """
        self.act = 0
        self.act_legal = True
        self.step_ecost = 0.0
        self.step_reward = 0.0
        self.reward_slot_end = 0.0
        self.reward = 0
        self.formation = True

    def energy_update(self):
        """更新无人机能耗"""
        self.step_ecost = self.env_cfg.FUAV_mecost * (
            self.move_dis_abs + self.move_dis_rela
        )
        self.energy = self.energy - self.step_ecost

    def act_execute_2(self, env_cell_map, env_luav_list):
        """
        fuav执行action
        """

        self.move_dis_rela, self.pos_abs = self.act_check_2(env_cell_map)
        x, y = self.pos_abs[0], self.pos_abs[1]
        self.pos_rela = (x - self.luav_pos_abs[0], y - self.luav_pos_abs[1])

        pos_x, pos_y, pos_z = self.pos_abs[0], self.pos_abs[1], self.pos_abs[2]
        if (
            pos_x < 0
            or pos_x >= self.env_cfg.map_length
            or pos_y < 0
            or pos_y >= self.env_cfg.map_width
            or pos_z not in self.env_cfg.alts
        ):
            print("fuav飞行超出边界 -> 死亡 0")
            self.alive = False

        # 检查是否在luav覆盖范围内
        if self.step % self.env_cfg.slot_step_num == 0:
            if (
                abs(self.pos_rela[0]) > self.env_cfg.luav_connect_dis
                or abs(self.pos_rela[1]) > self.env_cfg.luav_connect_dis
            ):
                # print(f"fuav {self.id} abs:{self.pos_abs}  rela:{self.pos_rela}飞行即将超出领航无人机覆盖范围 -> 死亡 1 (不应该发生)")
                self.formation = False

        if env_cell_map[self.pos_abs[0]][self.pos_abs[1]].uav_inf[1] > 0:
            print(f"fuav {self.id} 即将前往区域存在其他fuav -> 死亡 2 (不应该发生)")
            self.alive = False

        if env_cell_map[self.pos_abs[0]][self.pos_abs[1]].obs == 1:
            print(f"fuav {self.id} 即将前往区域存在障碍物 -> 死亡 3(不应该发生)")
            self.alive = False

        if self.alive:
            # 更新能耗
            self.energy_update()
        else:
            # 领导无人机因绝对动作决策导致跟随无人机死亡统计
            env_luav_list[self.luav_id].fuav_absact_kill_num += 1

    def act_check_2(self, env_cell_map):
        """
        对动作进行检查：
            若合法，则返回动作执行后的绝对、相对移动距离、新绝对、相对坐标。
            若不合法，则相对动作不执行，只执行绝对动作，并返回动作执行后的绝对、相对移动距离、新绝对、相对坐标。
        """
        self.act_legal = True

        if self.act == 0:
            """相对坐标不变，绝对坐标跟随领航无人机发生变化"""
            # print(f"fuav {self.id} -> 不相对移动")
            move_dis_rela = 0
            move_rela = (0, 0)
        else:
            # 确定新的 (x,y,z)
            dir = self.env_cfg.fuav_acts[self.act]
            if dir in [2 * PI, PI / 2, PI, PI * 3 / 2]:
                move_dis_rela = 1
            elif dir in [PI / 4, PI * 3 / 4, PI * 5 / 4, PI * 7 / 4]:
                move_dis_rela = math.sqrt(2)
            else:
                print("错误 - 0")
                exit()

            # print(
            #     f"fuav {self.id} 相对移动方向 -> {dir}；相对移动距离 -> {move_dis_rela}"
            # )
            move_rela = (move_dis_rela * math.cos(dir), move_dis_rela * math.sin(dir))
        # 未来位置
        pos_abs = (
            round(self.pos_abs[0] + move_rela[0]),
            round(
                self.pos_abs[1] + move_rela[1],
            ),
            self.pos_abs[2],
        )

        pos_x, pos_y, pos_z = pos_abs[0], pos_abs[1], pos_abs[2]
        if (
            pos_x < 0
            or pos_x >= self.env_cfg.map_length
            or pos_y < 0
            or pos_y >= self.env_cfg.map_width
            or pos_z not in self.env_cfg.alts
        ):
            # print("fuav飞行超出边界 -> illegal action 0")
            self.act_legal = False
        else:
            if env_cell_map[pos_x][pos_y].uav_inf[1] > 0:
                # print(f"fuav {self.id} 即将前往区域存在其他fuav -> illegal action 2")
                self.act_legal = False
            if env_cell_map[pos_x][pos_y].obs == 1:
                # print(f"fuav {self.id} 即将前往区域存在障碍物 -> illegal action 3")
                self.act_legal = False

        if self.act_legal:
            return move_dis_rela, pos_abs
        else:
            # print(f"fuav {self.id} -> 不相对移动")
            self.step_reward = -5.0
            return 0, self.pos_abs

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
