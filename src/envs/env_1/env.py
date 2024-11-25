import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .cellmap import CellMap
from .uav import UAV
from .ue import UE


class Env:
    class Cluster:
        def __init__(self, index: int, center, radius: float):
            self.id: int = index
            self.center: list = center
            self.radius: float = radius
            self.ue_num: int = 0
            self.ue_list: list = []
            self.tilde_C: list = []

    class Obstacle:
        def __init__(self, index: int, pos: list, shape: list):
            self.id: int = index
            self.pos: list = pos
            self.shape: list = shape

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: 配置文件字典
        """
        self.config: dict = kwargs
        self.run_id = (
            self.config["run_id"]
            if "run_id" in self.config
            else datetime.now().strftime("%m_%d_%H_%M_%S")
        )
        self.map_name: str = self.config["map_name"]
        self.map_length: int = self.config["map_length"]
        self.map_width: int = self.config["map_width"]
        self.map_height: int = self.config["map_height"]
        self.uav_num: int = self.config["uav_num"]
        self.use_onehot: bool = (
            self.config["use_onehot"] if "use_onehot" in self.config else False
        )
        self.use_novelty: bool = (
            self.config["use_novelty"] if "use_novelty" in self.config else False
        )
        self.use_hybrid_novelty: bool = (
            self.config["use_hybrid_novelty"]
            if "use_hybrid_novelty" in self.config
            else False
        )
        self.obstacle_list = []
        if "obs_list_file" in self.config:
            obs_data = pd.read_csv(self.config["obs_list_file"]).values.tolist()
            for obs in obs_data:
                self.obstacle_list.append(self.Obstacle(obs[0], obs[1:3], obs[3:]))
        if self.config["uav"]["specify_initial_pos"]:
            self.uav_initial_pos = self.config["uav"]["initial_pos"]
        else:
            self.uav_initial_pos = self.random_initialize_uav()
        del self.config["uav"]["initial_pos"]
        self.uav_list = [
            UAV(
                index=i,
                pos=self.uav_initial_pos[i].copy(),
                initial_pos=self.uav_initial_pos[i].copy(),
                slot_length=self.config["slot_length"],
                **self.config["uav"],
            )
            for i in range(self.config["uav_num"])
        ]
        if not self.config["specify_ue_pos"]:
            cluster_centers = (
                self.config["ue"]["cluster_centers"]
                if self.config["ue"]["specify_cluster_centers"]
                else None
            )
            ue_pos, clusters = self.random_initialize_ue(cluster_centers)
        else:
            ue_pos = pd.read_csv(self.config["ue_pos_file"]).values
            clusters = pd.read_csv(self.config["ue_cluster_file"]).values
        self.cluster_list = [
            self.Cluster(
                index=int(cluster[0]),
                center=[cluster[1], cluster[2]],
                radius=cluster[3],
            )
            for cluster in clusters
        ]
        self.ue_list = [
            UE(index=ue[0], cluster=ue[1], pos=list(ue[2:])) for ue in ue_pos
        ]
        for ue in self.ue_list:
            cluster = ue.cluster
            self.cluster_list[cluster].ue_num += 1
            self.cluster_list[cluster].ue_list.append(ue.id)
        self.cell_map = CellMap(
            map_length=self.map_length,
            map_width=self.map_width,
            ue_list=self.ue_list,
            obs_list=self.obstacle_list,
            uav_init_pos=self.uav_initial_pos,
        )
        self.current_slot = 1
        self.max_slot_num = self.config["slot_num"]
        self._record_current_slot = {}  # 用于收集本时隙的相关数据
        self._record_data = []  # 用于收集本 episode 的相关数据
        self.fairness = 1  # 本时隙的公平指数
        self.adjacency = self.get_adjacency()  # 无人机拓扑图的邻接矩阵

    def random_initialize_uav(self) -> list:
        while True:
            x = np.random.uniform(0, self.map_length - 1, self.uav_num).astype(int)
            y = np.random.uniform(0, self.map_width - 1, self.uav_num).astype(int)
            z = np.random.uniform(0, self.map_height - 1, self.uav_num).astype(int)
            poses = np.transpose(np.stack((x, y, z))).tolist()
            flag = True
            for pos in poses:
                for obs in self.obstacle_list:
                    if (
                        abs(obs.pos[0] - pos[0]) < obs.shape[0]
                        and abs(obs.pos[1] - pos[1]) < obs.shape[1]
                    ):
                        flag = False
            if flag:
                return poses

    def random_initialize_ue(self, cluster_centers: list = None):
        """
        随机初始化 UE 位置，cluster 中心在区域内部均匀分布，半径服从 0 到最大半径的均匀分布
        UE 离圆心的距离服从正态分布，均值为 0，标准差为半径的三分之一，取绝对值
        UE 相对于圆心的方位角服从 0 到 2pi 的均匀分布
        :return: [[cluster, x, y]], [Cluster]
        """
        ue_pos = []
        clusters = []
        for i in range(self.config["ue"]["cluster_num"]):
            if cluster_centers is None:
                center = np.array(
                    [
                        i,
                        np.random.uniform(0, self.map_length),
                        np.random.uniform(0, self.map_width),
                    ],
                    dtype=np.float32,
                )
            else:
                center = np.array(
                    [i, cluster_centers[i][0], cluster_centers[i][1]],
                    dtype=np.float32,
                )
            # radius = np.random.uniform(
            #    low=0, high=self.config["ue"]["max_cluster_size"]
            # )
            radius = self.config["ue"]["max_cluster_size"]
            clusters.append([i, center[1], center[2], radius])
            ue_num = self.config["ue"]["num_in_a_cluster"]
            theta = np.random.uniform(low=0, high=2 * np.pi, size=ue_num)
            # sigma = radius / 3
            # r = np.abs(np.random.normal(loc=0, scale=sigma, size=ue_num))
            r = np.abs(np.random.uniform(low=0, high=radius, size=ue_num))
            relative = np.stack(
                [np.zeros(ue_num), r * np.cos(theta), r * np.sin(theta)], axis=1
            )
            pos = center + relative
            ue_pos.append(
                pos[
                    np.logical_and(pos[:, 1] >= 0, pos[:, 1] <= self.map_length)
                    & np.logical_and(pos[:, 2] >= 0, pos[:, 2] <= self.map_width)
                ].astype(np.int32)
            )
        ue_pos = np.vstack(ue_pos)
        ue_id = np.linspace(0, len(ue_pos) - 1, len(ue_pos), dtype=np.int32).reshape(
            (-1, 1)
        )
        return np.concatenate([ue_id, ue_pos], axis=1), clusters

    def get_image_obs(self) -> np.ndarray:
        res = []
        for uav in self.uav_list:
            uav_obs = []
            radius = (
                uav.obs_radius[uav.pos[2]]
                if uav.obs_radius_change
                else uav.max_obs_radius
            )
            x0, y0 = uav.pos[:2]
            for x in range(int(x0 - radius), int(x0 + radius)):
                if x < 0 or x >= self.map_length:
                    uav_obs.append([[-1] * 4] * (2 * int(radius)))
                    continue
                x_obs = []
                for y in range(int(y0 - radius), int(y0 + radius)):
                    if y < 0 or y >= self.map_width:
                        x_obs.append([-1] * 4)
                    else:
                        cell = self.cell_map[x, y]
                        if cell.ue_num != 0 and cell.ue_list[0].cover_uav == uav.id:
                            serve_ue_num = cell.ue_num
                        else:
                            serve_ue_num = 0
                        x_obs.append(
                            [
                                # cell.ue_num,
                                serve_ue_num,
                                cell.obstacle != -1,
                                cell.uav_id != -1 and cell.uav_id != uav.id,
                                cell.novelty[uav.id],
                            ]
                        )
                uav_obs.append(x_obs)
            if uav.obs_radius_change:  # 将观测 padding 到最大范围
                pad_size = int(uav.max_obs_radius - radius)
                uav_obs = np.pad(
                    np.array(uav_obs),
                    pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                    mode="constant",
                    constant_values=-2,
                ).tolist()
            res.append(uav_obs)
        return np.array(res, dtype=np.float32)

    def get_array_obs(self) -> np.ndarray:
        res = [
            uav.get_array_obs(
                [self.map_length, self.map_width, self.map_height], self.use_onehot
            )
            for uav in self.uav_list
        ]
        return np.array(res, dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        """
        返回当前的观测
        image_obs: [智能体][x][y][UE, 障碍物, 临近无人机（有无）, Novelty]
        array_obs: [智能体][无人机的位置, 剩余能量]
        :return: [智能体][拼接之后的向量]
        """
        image_obs = self.get_image_obs()
        array_obs = self.get_array_obs()
        image_obs = image_obs.reshape(len(image_obs), -1)
        return np.concatenate((image_obs, array_obs), axis=1)

    def get_obs_size(self) -> int:
        if self.use_onehot:
            return int(
                self.config["channel_num"]
                * ((2 * self.uav_list[0].max_obs_radius) ** 2)
                + self.map_length
                + self.map_width
                + self.map_height
                + 1
            )
        else:
            return int(
                self.config["channel_num"]
                * ((2 * self.uav_list[0].max_obs_radius) ** 2)
                + 4
            )

    def get_raw_state(self, uav_id: int):
        """
        返回 HAP 编码使用的原始观测
        :param uav_id: 无人机 id
        :return: image (50, 50, 4) [ue_num, obstacle, uav_status, novelty],
                 vector (324, ) 所有 UE 的位置以及无人机的位置、当前时隙能耗
        """
        image = []
        for i in range(self.map_length):
            line = []
            for j in range(self.map_width):
                cell = self.cell_map[i, j]
                line.append(
                    [
                        cell.ue_num,
                        cell.obstacle,
                        cell.uav_id != -1,
                        cell.novelty[uav_id],
                    ]
                )
            image.append(line)
        vector = []
        for uav in self.uav_list:
            vector += uav.get_array_obs()
        for ue in self.ue_list:
            vector += ue.pos
        return image, vector

    def get_adjacency(self) -> np.ndarray:
        """
        返回无人机拓扑图的连接矩阵
        Returns:
            np.ndarray(uav_num, uav_num)
        """
        points = np.array([uav.pos for uav in self.uav_list])
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distance = np.sqrt(np.sum(np.square(diff), axis=-1))
        adjacency = (distance < self.uav_list[0].comm_radius).astype(np.float32)
        return np.tile(adjacency, (self.uav_num, 1, 1))

    def get_state(self) -> np.ndarray:
        """
        返回 HAP 编码之后的状态
        """
        if self.use_onehot:
            length_onehot_base = np.eye(self.map_length)
            width_onehot_base = np.eye(self.map_width)
            height_onehot_base = np.eye(self.map_height)
            onehot_bases = [length_onehot_base, width_onehot_base, height_onehot_base]
            cluster_centers = [
                np.concatenate(
                    [
                        length_onehot_base[cluster.center[0]],
                        width_onehot_base[cluster.center[1]],
                    ]
                )
                for cluster in self.cluster_list
            ]
            uav_pos = [
                np.concatenate([onehot_bases[i][uav.pos[i]] for i in range(3)])
                for uav in self.uav_list
            ]
            obstacles = [
                np.concatenate(
                    [
                        length_onehot_base[obs.pos[0]],
                        width_onehot_base[obs.pos[1]],
                        length_onehot_base[obs.shape[0]],
                        width_onehot_base[obs.shape[1]],
                    ]
                )
                for obs in self.obstacle_list
            ]
        else:
            cluster_centers = [
                [
                    cluster.center[0] / self.map_length,
                    cluster.center[1] / self.map_width,
                ]
                for cluster in self.cluster_list
            ]
            uav_pos = [
                [
                    uav.pos[0] / self.map_length,
                    uav.pos[1] / self.map_width,
                    uav.pos[2] / self.map_height,
                ]
                for uav in self.uav_list
            ]
            obstacles = [
                [
                    obs.pos[0] / self.map_length,
                    obs.pos[1] / self.map_width,
                    obs.shape[0] / self.map_length,
                    obs.shape[1] / self.map_width,
                ]
                for obs in self.obstacle_list
            ]
        cluster_centers = np.array(cluster_centers).reshape(-1)
        uav_pos = np.array(uav_pos).reshape(-1)
        obstacles = np.array(obstacles).reshape(-1)
        return np.concatenate((cluster_centers, uav_pos, obstacles))

    def get_state_size(self) -> int:
        if self.use_onehot:
            return (
                (self.map_length + self.map_width) * len(self.cluster_list)
                + (self.map_length + self.map_width + self.map_height)
                * len(self.uav_list)
                + (self.map_length + self.map_width) * 2 * len(self.obstacle_list)
            )
        else:
            return (
                2 * len(self.cluster_list)
                + 3 * len(self.uav_list)
                + 4 * len(self.obstacle_list)
            )

    def get_avail_actions(self) -> np.ndarray:
        return np.ones(shape=(len(self.uav_list), self.config["uav"]["action_size"]))

    def get_total_actions(self) -> int:
        return self.config["action_size"]

    def action_check(self, action: list) -> tuple:
        """
        检查所有无人机的动作是否合法
        :param action: 按编号顺序给出每个 uav 的动作序号 [[theta, horizon, vertical]]
        :return: [bool] 表示每个无人机的动作是否合法, [[id, pos, fly_time]] 表示无人机最终的行为
        """
        act_results = [
            [i] + list(self.uav_list[i].take_action(action[i]))
            for i in range(len(self.uav_list))
        ]
        res = [True] * len(self.uav_list)
        while True:
            act_results.sort(key=lambda item: item[2])
            flag = [True] * len(self.uav_list)
            uav_pos_set = set()
            for index, (uav_id, try_pos, _) in enumerate(act_results):
                try_pos = tuple(try_pos)
                if try_pos in uav_pos_set:  # 检查无人机之间的碰撞
                    res[uav_id] = False
                    flag[uav_id] = False
                else:
                    uav_pos_set.add(try_pos)
                if (
                    try_pos[0] < 0
                    or try_pos[0] >= self.map_length
                    or try_pos[1] < 0
                    or try_pos[1] >= self.map_width
                    or try_pos[2] < 0
                    or try_pos[2] >= self.map_height
                ):  # 检查越界
                    res[uav_id] = False
                    flag[uav_id] = False
                elif try_pos[2] <= self.cell_map[try_pos].obstacle:  # 检查障碍碰撞
                    res[uav_id] = False
                    flag[uav_id] = False
                if not flag[uav_id]:
                    uav_pos_set.add(
                        tuple(self.uav_list[uav_id].pos)
                    )  # 能够减少循环次数
                    # 将无人机的行为修改为保持不动
                    act_results[index][1] = self.uav_list[uav_id].pos
                    act_results[index][2] = 0
            if np.sum(flag) == len(flag):  # 直到所有无人机的行为都不会导致碰撞
                break
        return res, act_results

    def get_r1(self) -> float:
        """
        计算所有无人机通用的 r1
        """
        r1 = 0
        for uav in self.uav_list:
            r1 += uav.serve_ue_count
        r1 /= len(self.ue_list)
        return r1

    def get_r2(self) -> float:
        """
        计算所有无人机通用的 r2：cluster 历史覆盖时隙占比的公平指数 + 当前时隙 cluster 覆盖率
        """
        tilde_C_m = []
        for cluster in self.cluster_list:
            cover_ue_num = 0
            for ue in cluster.ue_list:
                cover_ue_num += float(self.ue_list[ue].cover_uav != -1)
            cluster.tilde_C.append(
                float(
                    cover_ue_num / cluster.ue_num
                    > self.config["ue"]["cluster_cover_threshold"]
                )
            )
            tilde_C_m.append(cluster.tilde_C)
        tilde_C_m = np.array(tilde_C_m, dtype=np.float32)
        cluster_cover_rate = tilde_C_m[:, -1].sum() / len(self.cluster_list)
        # cluster_cover_rate = tilde_C_m[:, -1].sum()
        jain_x = tilde_C_m.sum(axis=1) / self.current_slot
        sum_square = jain_x.sum() ** 2
        square_sum = np.square(jain_x).sum()
        if square_sum < 1e-4:
            F_t = 1
        else:
            F_t = sum_square / (len(self.cluster_list) * square_sum)
        r2 = F_t + cluster_cover_rate
        self.fairness = F_t
        self._record_current_slot["fairness"] = F_t
        self._record_current_slot["cluster_cover"] = tilde_C_m[:, -1]
        return r2

    def get_sub_rewards(self) -> np.ndarray:
        """
        返回子奖励
        r1: 全局 UE 覆盖率
        r2: 公平指数 * 所有 Cluster 覆盖率
        r3: 负归一化能耗
        r4: 无人机服务 UE 数/服务上限
        r5: 非法动作惩罚项
        r6: 当前时隙覆盖 UE 数相对于历史覆盖最大 UE 覆盖数的差值 / 当前时隙覆盖 UE 数 * 2
        r7: 与相邻最近的 UE cluster 是否被覆盖
        r8: 观测 cell 的 novelty 平均值
        :return: [UAV][r1, r2, r3, r4, r5, r6, r7, r8]
        """
        r1 = self.get_r1() * 2
        self._record_current_slot["ue_cover_rate"] = r1 / 2
        r2 = self.get_r2()
        res = []
        for uav in self.uav_list:
            r3 = -uav.slot_energy / uav.max_energy_in_a_slot
            r4 = uav.serve_ue_count / uav.max_serve_capacity * 2
            r5 = float(not uav.action_valid) * self.config["penalty"]
            r6 = max(uav.serve_ue_count - uav.max_ue_count, 0) * 2
            if uav.serve_ue_count != 0:
                r6 /= uav.serve_ue_count
            r6 *= 2
            dis = [
                (uav.pos[0] - cluster.center[0]) ** 2
                + (uav.pos[1] - cluster.center[1]) ** 2
                for cluster in self.cluster_list
            ]
            uav_cluster = self.cluster_list[np.argmin(dis)]
            r7 = uav_cluster.tilde_C[-1] * 2
            radius = (
                uav.obs_radius[uav.pos[2]]
                if uav.obs_radius_change
                else uav.max_obs_radius
            )
            x, y = uav.pos[0], uav.pos[1]
            novelty = [
                self.cell_map[i, j].novelty[uav.id]
                for i in range(
                    round(max(0, x - radius)),
                    round(min(x + radius, self.map_length)),
                )
                for j in range(
                    round(max(0, y - radius)),
                    round(min(y + radius, self.map_width)),
                )
            ]
            r8 = np.mean(novelty) * 2
            # r8 = min(np.sum(novelty), self.config["max_novelty_reward"])
            res.append([r1, r2, r3, r4, r5, r6, r7, r8])
        return np.array(res, dtype=np.float32)

    def get_reward(
        self,
        sub_rewards: np.ndarray = None,
        model_type: str = "default",
    ) -> np.ndarray:
        """
        按无人机 id 顺序返回求和后的奖励
        :param sub_rewards: 分项奖励（来自 get_sub_rewards）
        :param model_type: ue | energy | illegal | default
        :return: [r]
        """
        if sub_rewards is None:
            sub_rewards = self.get_sub_rewards()
        if model_type == "ue":
            rewards = sub_rewards[:, 3]
        elif model_type == "energy":
            rewards = sub_rewards[:, 2]
        elif model_type == "illegal":
            rewards = sub_rewards[:, 4]
        else:
            rewards = sub_rewards[:, 2:5].sum(axis=1)
        if self.use_novelty:
            novelty = sub_rewards[:, 5:].sum(axis=1)
            rewards += novelty if model_type == "default" else novelty / 3
        return rewards

    def get_reward_total(self, sub_rewards: np.ndarray) -> float:
        sub_reward_total = [
            sub_rewards[0][0],  # 总 UE 覆盖率
            sub_rewards[0][1],  # 公平指数 + 当前时隙 Cluster 覆盖率
            sub_rewards[:, 2].mean(),  # UAV 平均负归一化能耗
            sub_rewards[:, 4].mean(),  # UAV 平均惩罚项
        ]
        if self.use_novelty:
            sub_reward_total.append(
                sub_rewards[:, 7].mean()
            )  # UAV 观测范围内的平均新颖性
        reward_total = np.sum(sub_reward_total)
        self._record_current_slot["sub_reward_total"] = sub_reward_total
        return reward_total

    def action_convert(self, action) -> list:
        """
        将 [value] 格式的 action 转换成 [[theta, horizon, vertical]]
        :param action: 一维动作列表
        :return: [[theta, horizon, vertical]] 格式的 action
        """
        # 编码方式
        # 水平位移为 0, 垂直飞行位移
        # 角度（水平位移为 1），垂直飞行位移
        action_space = self.config["uav"]["action_space"]
        l_vertical = len(action_space[2])
        res = []
        for act in action:
            index = act // l_vertical
            vertical = act % l_vertical
            theta, horizon = (0, 0) if index == 0 else (index - 1, 1)
            res.append([theta, horizon, vertical])
        return res

    def step(
        self,
        action,
        request_sub_rewards: bool = False,
        model_type: str = "default",
        dry_run: bool = False,
    ) -> tuple:
        """
        :param action: 按编号顺序给出每个 uav 的动作序号
        :param request_sub_rewards: 是否要求返回子奖励
        :param model_type: ue | energy | illegal | default
        :param dry_run: 如果为 True，则不会真的执行这个动作
        :return: (reward, reward_total, done, info)
        """
        uav_snap = []
        ue_snap = []
        cell_snap = []
        if dry_run:
            # 记录快照
            uav_snap = [[uav.pos.copy(), uav.energy] for uav in self.uav_list]
            ue_snap = [ue.cover_slot for ue in self.ue_list]
            cell_snap = [
                [
                    [self.cell_map[i, j].uav_id, self.cell_map[i, j].access_cnt.copy()]
                    for j in range(self.map_width)
                ]
                for i in range(self.map_length)
            ]
        action = self.action_convert(action)
        validity, act_results = self.action_check(action)
        for i in range(len(self.uav_list)):
            uav = self.uav_list[i]
            if validity[i]:
                origin_cell = self.cell_map[uav.pos[:2]]
                if origin_cell.uav_id == i:
                    origin_cell.uav_id = -1
                self.cell_map[uav.try_pos].uav_id = i
            uav.update(validity[i])
            radius = (
                uav.obs_radius[uav.pos[2]]
                if uav.obs_radius_change
                else uav.max_obs_radius
            )
            x0, y0 = uav.pos[:2]
            for x in range(
                round(max(0, x0 - radius)),
                round(min(x0 + radius, self.map_length)),
            ):
                for y in range(
                    round(max(0, y0 - radius)),
                    round(min(y0 + radius, self.map_width)),
                ):
                    cell = self.cell_map[x, y]
                    if cell.obstacle <= 0:
                        cell.access_cnt[i] += 1
                        cell.novelty[i] = 1 / (
                            np.sum(cell.access_cnt)
                            if self.use_hybrid_novelty
                            else cell.access_cnt[i]
                        )
        for ue in self.ue_list:
            ue.cover_uav = -1
        for uav in self.uav_list:
            uav.serve_ue_count = 0
        for uav_id, pos, _ in act_results:

            def get_dis(ue_pos, uav_pos):
                return np.sqrt(
                    (uav_pos[0] - ue_pos[0]) ** 2 + (uav_pos[1] - ue_pos[1]) ** 2
                )

            for ue in self.ue_list:
                if ue.cover_uav != -1:
                    continue
                uav = self.uav_list[uav_id]
                if get_dis(ue.pos, pos) <= uav.cover_radius[uav.pos[2]]:
                    ue.cover_uav = uav_id
                    ue.cover_slot += 1
                    uav.serve_ue_count += 1
                if uav.serve_ue_count == uav.max_serve_capacity:
                    break
        if not dry_run:
            for ue in self.ue_list:
                ue.cover_uav_record.append(ue.cover_uav)
        sub_rewards = self.get_sub_rewards()
        rewards = self.get_reward(sub_rewards, model_type)
        reward_total = self.get_reward_total(sub_rewards)
        self.adjacency = self.get_adjacency()
        done = self.current_slot == self.max_slot_num
        uav_data = [uav.record() for uav in self.uav_list]
        for i in range(len(self.uav_list)):
            uav_data[i]["sub_reward"] = sub_rewards[i]
            uav_data[i]["reward"] = rewards[i]
            uav_data[i]["connected_uav"] = np.where(self.adjacency[i][i] > 0.5)[0]
        self._record_current_slot["reward_total"] = reward_total
        self._record_current_slot["uav_data"] = uav_data
        self._record_current_slot["ue_data"] = [
            [ue.cover_uav, ue.cover_slot] for ue in self.ue_list
        ]
        if not dry_run:
            self._record_data.append(self._record_current_slot)
            self._record_current_slot = {}
            self.current_slot += 1
        else:
            # 恢复快照
            for i in range(self.uav_num):
                uav = self.uav_list[i]
                uav.pos, uav.energy = uav_snap[i]
            for i in range(len(self.ue_list)):
                self.ue_list[i].cover_slot = ue_snap[i]
            for i in range(self.map_length):
                for j in range(self.map_width):
                    cell = self.cell_map[i, j]
                    cell.uav_id, cell.access_cnt = cell_snap[i][j]
        return_reward = sub_rewards if request_sub_rewards else rewards
        return return_reward, reward_total, done, self.get_env_info()

    def reset(self):
        for i in range(len(self.uav_list)):
            self.uav_list[i].reset()
        for ue in self.ue_list:
            ue.reset()
        for cluster in self.cluster_list:
            cluster.tilde_C.clear()
        self.current_slot = 1
        self._record_current_slot.clear()
        self._record_data.clear()
        self.cell_map.reset()
        self.adjacency = self.get_adjacency()

    def record(self, t_env: int = None, path: str = None, record_ue_pos: bool = False):
        """
        记录环境运行产生的数据
        """
        if path is None:
            path = Path.cwd() / "record" / self.run_id / str(os.getppid())
            if t_env is None:
                cur_time = datetime.now().strftime("%m_%d_%H_%M_%S")
                path = path / cur_time
            else:
                path = path / str(t_env)
        else:
            path = Path(path)
        uav_path = path / "uav"
        ue_path = path / "ue"
        if not uav_path.exists():
            uav_path.mkdir(parents=True, exist_ok=True)
        if not ue_path.exists():
            ue_path.mkdir(parents=True, exist_ok=True)
        if record_ue_pos:
            self.record_ue_pos(ue_path)
        self.record_ue_status(ue_path)
        self.record_uav_init_pos(uav_path)
        self.record_uav_status(uav_path)
        self.record_system_status(path)

    def get_stats(self):
        pass

    def record_ue_pos(self, path: Path):
        """
        将 UE 的位置写入 path/ue_pos.csv，将 Cluster 相关信息写入 path/ue_clusters.csv
        :param path: 保存文件的路径
        """
        ue_pos = [[ue.id, ue.cluster, ue.pos[0], ue.pos[1]] for ue in self.ue_list]
        clusters = [
            [cluster.id, cluster.center[0], cluster.center[1], cluster.radius]
            for cluster in self.cluster_list
        ]
        pd.DataFrame(ue_pos, columns=["ue_id", "cluster_id", "x", "y"]).to_csv(
            path / "ue_pos.csv", index=False
        )
        pd.DataFrame(
            clusters, columns=["cluster_id", "center_x", "center_y", "radius"]
        ).to_csv(path / "ue_clusters.csv", index=False)

    def record_ue_status(self, path: Path):
        columns = [f"ue_{i}" for i in range(len(self.ue_list))]
        data = [slot["ue_data"] for slot in self._record_data]
        pd.DataFrame(data, columns=columns).to_csv(path / "ue_status.csv", index=False)

    def record_uav_init_pos(self, path: Path):
        columns = ["uav_id", "init_pos"]
        data = [[i, self.uav_list[i].initial_pos] for i in range(len(self.uav_list))]
        pd.DataFrame(data, columns=columns).to_csv(path / "init_pos.csv", index=False)

    def record_uav_status(self, path: Path):
        columns = [
            "pos",
            "action",
            "slot_energy",
            "serve_ue_count",
            "connected_uav",
            "cover_radius",
            "validity",
            "sub_reward",
            "reward",
        ]
        for i in range(len(self.uav_list)):
            data = [slot["uav_data"][i] for slot in self._record_data]
            pd.DataFrame(data, columns=columns).to_csv(
                path / f"uav_{i}.csv", index=False
            )

    def record_system_status(self, path: Path):
        columns = [
            "ue_cover_rate",
            "fairness",
            "sub_reward_total",
            "reward_total",
            "cluster_cover_num",
            "cluster_covered",
        ]
        data = []
        for slot in self._record_data:
            cluster_cover = slot["cluster_cover"]
            cluster_cover_num = cluster_cover.sum()
            data.append(
                dict(
                    ue_cover_rate=slot["ue_cover_rate"],
                    fairness=slot["fairness"],
                    sub_reward_total=slot["sub_reward_total"],
                    reward_total=slot["reward_total"],
                    cluster_cover_num=cluster_cover_num,
                    cluster_covered=cluster_cover.astype(bool).tolist(),
                )
            )
        pd.DataFrame(data, columns=columns).to_csv(
            path / "system_status.csv", index=False
        )

    def get_env_info(self) -> dict:
        return dict(
            state_shape=self.get_state_size(),
            obs_shape=self.get_obs_size(),
            n_actions=self.config["uav"]["action_size"],
            n_agents=self.config["uav_num"],
            episode_limit=self.max_slot_num,
        )
