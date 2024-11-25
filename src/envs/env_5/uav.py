from typing import List, Tuple

import numpy as np


class UAV:
    def __init__(self, tcs_pos: list, **config):
        self.pos = np.array(config["initial_pos"], dtype=float)
        self.energy: float = float(config["max_energy"])  # 无人机当前剩余能量
        self.fly_time: float = 0.0  # 无人机当前时隙飞行时间
        self.low_power_penalty: float = 0.0  # 无人机因能量过低收到的惩罚
        self.energy_harvest: float = 0.0
        self.energy_consume: float = 0.0

        # 以下信息在每次训练过程中保持不变
        self.tcs_pos = np.array(tcs_pos, dtype=float)
        self.speed: float = config["speed"]
        self.height: float = config["height"]
        self.initial_pos: list = config["initial_pos"]
        self.max_energy = self.energy
        self.action_space: List[Tuple[float, float]] = config["action_space"]
        self.low_power_thresh: float = config["low_power_thresh"]
        self.low_tr_thresh: float = config["low_tr_thresh"]

        # 公式中的超参数
        # (8)
        self.B: float = float(config["B"])  # 链路带宽
        self.P_m: float = config["P_m"]  # SN 的传输功率
        self.beta_0: float = config["beta_0"]  # 参考距离上的功率增益
        self.sigma_square: float = config["sigma_square"]  # 接收器的白高斯噪声功率
        self.a_1: float = config["a_1"]  # logistic 回归参数
        self.a_2: float = config["a_2"]  # logistic 回归参数
        self.b_1: float = config["b_1"]  # logistic 回归参数
        self.b_2: float = config["b_2"]  # logistic 回归参数
        # (13)
        self.eta: float = config["eta"]  # 能量转换效率
        self.P_0: float = config["P_0"]  # 激光功率
        self.tau: float = config["tau"]  # 时隙长度
        self.A: float = config["A"]  # 激光接收器的面积
        self.V: float = config["V"]  # 接收器光学效率
        self.alpha: float = config["alpha"]  # 单位米内的链路衰减效率
        self.D: float = config["D"]  # 激光束的尺寸
        self.beta: float = config["beta"]  # 激光束的角度扩展
        # (16)
        self.P_blade: float = config["P_blade"]  # 悬停时的叶片剖面功率
        self.P_i: float = config["P_i"]  # 悬停时的叶片诱导功率
        self.U_tip: float = config["U_tip"]  # 旋翼叶尖速度
        self.v_0: float = config["v_0"]  # 悬停时的平均诱导速度
        self.d_0: float = config["d_0"]  # 机身阻力比
        self.rho: float = config["rho"]  # 空气密度
        self.s: float = config["s"]  # 转子坚固度
        self.A_roter: float = config["A_roter"]  # 转子盘面积
        self.P_c: float = config["P_c"]  # 无人机传输功率

        self.hover_power = self.P_blade + self.P_i
        self.fly_power = (
            self.P_blade * (1 + 3 * np.square(self.speed / self.U_tip))
            + self.P_i
            * np.sqrt(
                np.sqrt(1 + np.square(np.square(self.speed / self.v_0)) / 4)
                - np.square(self.speed / self.v_0) / 2
            )
            + self.d_0 * self.rho * self.s * self.A_roter * (self.speed**3)
        )

    def calc_trans_volume(self, ue_pos: np.ndarray):
        """
        计算无人机对于每个UE的数据传输量
        :return: 数据传输量，覆盖情况
        """
        distance = np.sum(np.square(self.pos - ue_pos), axis=1) + np.square(self.height)
        alpha_m = self.height / np.sqrt(distance)
        f_m = self.a_1 + self.a_2 / (1 + np.exp(-(self.b_1 + self.b_2 * alpha_m)))
        trans_rate = self.B * np.log2(
            1 + (self.P_m * self.beta_0 * f_m) / (self.sigma_square * distance)
        )
        connected = trans_rate >= self.low_tr_thresh
        trans_rate[connected] = 0
        cover_num = np.sum(connected)
        if cover_num == 0:
            return np.zeros(len(ue_pos)), np.zeros_like(connected)
        return trans_rate * (self.tau - self.fly_time) / cover_num, connected

    def calc_energy_harvest(self) -> float:
        d_tcs = np.sqrt(
            np.sum(np.square(self.tcs_pos - self.pos)) + np.square(self.height)
        )
        g_tcs = (self.A * self.V * np.exp(-self.alpha * d_tcs)) / np.square(
            self.D + self.beta * d_tcs
        )
        return self.eta * g_tcs * self.P_0 * self.tau

    def calc_energy_consume(self) -> float:
        return self.fly_power * self.fly_time + (self.hover_power + self.P_c) * (
            self.tau - self.fly_time
        )

    def step(self, action: int) -> None:
        r, theta = self.action_space[action]
        self.fly_time = r / self.speed
        delta = np.array([r * np.cos(theta), r * np.sin(theta)])
        self.pos += delta
        self.energy_harvest = self.calc_energy_harvest()
        self.energy_consume = self.calc_energy_consume()
        energy_delta = self.energy_harvest - self.energy_consume
        self.energy = min(self.energy + energy_delta, self.max_energy)
        self.low_power_penalty = (
            energy_delta / self.max_energy
            if self.energy < self.low_power_thresh * self.max_energy
            else 0
        )

    def reset(self):
        self.pos = np.array(self.initial_pos, dtype=float)
        self.energy = self.max_energy
