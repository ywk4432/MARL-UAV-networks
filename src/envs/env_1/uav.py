import math

import numpy as np


class UAV:
    def __init__(
        self,
        index: int = 0,
        pos: list = None,
        obs_radius: list = None,
        cover_radius: list = None,
        action_space: list = None,
        horizon_speed: float = None,
        vertical_speed: float = None,
        max_serve_capacity: int = None,
        max_energy: float = None,
        epsilon: list = None,
        slot_length: float = None,
        max_obs_radius: int = None,
        obs_radius_change: bool = None,
        action_size: int = None,
        specify_initial_pos: bool = None,
        initial_pos: list = None,
        comm_radius: int = 5,
    ):
        """
        :param index: 序号
        :param pos: 初始位置
        :param obs_radius: 离散化的观测半径数组
        :param cover_radius: 离散化的覆盖半径数组
        :param action_space: [[角度离散值], [水平位移离散值], [垂直位移离散值]]
        :param horizon_speed: 无人机的水平飞行速度
        :param vertical_speed: 无人机的垂直飞行速度
        :param max_serve_capacity: 无人机的最大服务用户数目
        :param max_energy: 无人机的最大能量
        :param epsilon: 能量消耗相关的超参数，5 维向量
        :param slot_length: 每个时隙的长度
        :param max_obs_radius: 最大观测半径
        :param obs_radius_change: 无人机的观测半径是否随高度变化
        :param action_size: 动作空间大小
        :param specify_initial_pos: 是否指定无人机的初始位置
        :param initial_pos: 指定的初始位置
        :param comm_radius: 无人机之间的通信距离
        """
        self.id = index
        self.alive = True
        self.action_valid = True
        self.pos = [0, 0, 0] if pos is None else pos
        self.serve_ue_count = 0
        self.energy = max_energy
        self.slot_energy = 0  # 无人机本时隙消耗的能量
        self.current_action = [0, 0, 0]
        self._self_action = [0, 0, 0]  # 无人机自身决策的动作
        self.try_pos = [0, 0, 0]  # 当前动作执行完后无人机的位置
        self._try_energy = 0  # 当前动作执行消耗的能量
        self.max_ue_count = 0  # 回合开始以来无人机最多服务的 UE 数目

        # 以下信息在每次训练过程中保持不变
        self.obs_radius: list = obs_radius
        self.cover_radius: list = cover_radius
        self.action_space: list = action_space
        self.horizon_speed: float = horizon_speed
        self.vertical_speed: float = vertical_speed
        self.max_serve_capacity: int = max_serve_capacity
        self.max_energy: float = max_energy
        self.epsilon: list = epsilon
        self.slot_length: float = slot_length
        self.max_obs_radius: int = max_obs_radius
        self.obs_radius_change: bool = obs_radius_change
        self.action_size: int = action_size
        self.max_energy_in_a_slot: float = self.get_max_energy_in_a_slot()
        self.specify_initial_pos: bool = specify_initial_pos
        self.initial_pos: list = initial_pos
        self.comm_radius: int = comm_radius

    def take_action(self, action):
        """
        计算这一个 action 执行结束时的位置，时耗与能耗
        :param action: [角度离散序号, 水平位移离散序号, 垂直位移离散序号]
        :return: 目标位置, 消耗的时间
        """
        r = self.action_space[1][action[1]]
        theta = self.action_space[0][action[0]]
        h = self.action_space[2][action[2]]
        delta = [r * math.cos(theta), r * math.sin(theta), h]
        self._self_action = [r, theta, h]
        self.try_pos = [round(self.pos[i] + delta[i]) for i in range(3)]
        time_fly = r / self.horizon_speed + abs(h) / self.vertical_speed
        time_hover = self.slot_length - time_fly
        e_flight = (
            self.epsilon[0] * r / self.horizon_speed
            + self.epsilon[1] * abs(h)
            + self.epsilon[2]
        )
        e_hover = self.epsilon[3] * time_hover
        e_comm = self.epsilon[4] * time_hover
        self._try_energy = e_flight + e_hover + e_comm
        return self.try_pos, time_fly

    def update(self, valid=True):
        """
        更新无人机的状态
        """
        self.action_valid = valid
        if valid:
            self.pos = self.try_pos
            self.slot_energy = self._try_energy
            self.current_action = self._self_action
        else:
            self.slot_energy = (self.epsilon[3] + self.epsilon[4]) * self.slot_length
            self.current_action = [0, 0, 0]
        self.energy -= self.slot_energy
        self.max_ue_count = max(self.serve_ue_count, self.max_ue_count)

    def reset(self, pos: list = None):
        """
        重置无人机的状态
        """
        self.alive = True
        self.action_valid = True
        self.pos = self.initial_pos.copy() if pos is None else pos
        self.serve_ue_count = 0
        self.energy = self.max_energy
        self.slot_energy = 0

    def get_array_obs(self, map_size: list = None, use_onehot: bool = False):
        if map_size is None:
            return self.pos + [self.slot_energy]
        elif use_onehot:
            onehot_pos = []
            for i in range(3):
                onehot_base = np.eye(map_size[i])
                onehot_pos += onehot_base[self.pos[i]].tolist()
            return onehot_pos + [self.slot_energy]
        return [
            self.pos[0] / map_size[0],
            self.pos[1] / map_size[1],
            self.pos[2] / map_size[2],
            self.slot_energy,
        ]

    def get_max_energy_in_a_slot(self):
        time_fly = (
            np.max(self.action_space[1]) / self.horizon_speed
            + np.max(self.action_space[2]) / self.vertical_speed
        )
        time_hover = self.slot_length - time_fly
        e_flight = (
            self.epsilon[0] * np.max(self.action_space[1]) / self.horizon_speed
            + self.epsilon[1] * np.max(self.action_space[2]) / self.vertical_speed
            + self.epsilon[2]
        )
        e_hover = self.epsilon[3] * time_hover
        e_comm = self.epsilon[4] * time_hover
        return e_flight + e_hover + e_comm

    def record(self) -> dict:
        return dict(
            pos=self.pos,
            cover_radius=self.cover_radius[self.pos[2]],
            action=self._self_action,
            validity=self.action_valid,
            slot_energy=self.slot_energy,
            serve_ue_count=self.serve_ue_count,
        )
