from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .uav import UAV
from .ue import UE


class Env:
    def __init__(self, **config):
        """
        适用于数据采集任务的单无人机环境
        Args:
            config: 配置文件字典
        """
        if "seed" in config:
            np.random.seed(config["seed"])
        self.map_length: int = config["map_length"]
        self.map_width: int = config["map_width"]
        self.slot_num: int = config["slot_num"]
        self.penalty: float = config["penalty"]

        # 初始化 UAV
        if not config["uav"]["specify_initial_pos"]:
            config["uav"]["initial_pos"] = [
                np.random.uniform(0, self.map_length),
                np.random.uniform(0, self.map_width),
            ]
        self.tcs_pos = np.array(config["tcs_pos"], dtype=float)
        self.uav = UAV(config["tcs_pos"], **config["uav"])
        self._uav_act_valid = True

        # 初始化 UE
        self.ue = UE(self.map_length, self.map_width, **config["ue"])

        self.current_slot = 0
        self.action_history = []

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        uav_pos = [self.uav.pos[0] / self.map_length, self.uav.pos[1] / self.map_width]
        ue_pos = self.ue.pos[:, 0] / self.map_length, self.ue.pos[:, 1] / self.map_width
        ue_pos = np.concatenate(np.stack(ue_pos, axis=1))
        uav_state = np.concatenate(
            ([self.uav.energy / self.uav.max_energy], uav_pos, ue_pos, self.ue.latency)
        )
        d_t = (
            np.sqrt(np.sum(np.square((self.uav.pos - self.tcs_pos))))
            / np.sum(np.square([self.map_length, self.map_width]))
            - self.uav.energy / self.uav.max_energy
        )
        c_t = (
            self.uav.energy / self.uav.max_energy
            - (
                np.sum(self.ue.payload[self.ue.covered])
                / (np.sum(self.ue.covered) * self.ue.max_payload)
            )
            if np.sum(self.ue.covered) != 0
            else 1 / self.ue.max_payload
        )
        high_state = np.concatenate(
            (
                [d_t, c_t, self.uav.energy / self.uav.max_energy],
                uav_pos,
                [self.get_fairness(), np.mean(self.ue.latency)],
            )
        )
        return uav_state, high_state

    def get_state_size(self) -> Tuple[int, int]:
        return 3 + 3 * self.ue.num, 7

    def action_check(self, action: int) -> int:
        r, theta = self.uav.action_space[action]
        delta = np.array([r * np.cos(theta), r * np.sin(theta)])
        x, y = self.uav.pos + delta
        if x >= self.map_length or y >= self.map_width or x < 0 or y < 0:
            return 1
        return 0

    def get_fairness(self) -> float:
        if np.sum(self.ue.cover_slot) == 0:
            return 1
        c_m = self.ue.cover_slot
        return np.square(np.sum(c_m)) / (len(c_m) * np.sum(np.square(c_m)))

    def get_reward(self) -> List[float]:
        fairness = self.get_fairness()
        a = np.mean(self.ue.latency)
        if self.current_slot == self.slot_num - 1:
            top_reward = fairness * self.uav.energy * np.exp(-a)
        elif self._uav_act_valid != 0:
            top_reward = self.penalty
        else:
            top_reward = 0
        if self._uav_act_valid != 0:
            return [self.penalty] * 3 + [top_reward]
        else:
            return [
                fairness,
                -a / self.current_slot,
                self.uav.energy / self.uav.max_energy + self.uav.low_power_penalty,
                top_reward,
            ]

    def step(self, action: int) -> bool:
        self.action_history.append(action)
        self._uav_act_valid = self.action_check(action)
        if self._uav_act_valid != 0:
            action = 0
        self.uav.step(action)
        volume, connected = self.uav.calc_trans_volume(self.ue.pos)
        self.ue.step(self.current_slot, volume, connected)
        self.current_slot += 1
        return self.current_slot == self.slot_num

    def reset(self):
        self.current_slot = 0
        self.uav.reset()
        self.ue.reset()
        self.action_history.clear()

    def get_env_info(self) -> dict:
        return {
            "n_agents": 1,
            "n_actions": len(self.uav.action_space),
            "state_shape": self.get_state_size(),
        }

    def save_replay(self, path: str):
        action_history = self.action_history.copy()
        self.reset()
        uav_data = {
            "pos": [],
            "energy": [],
            "action": [],
            "validity": [],
            "rewards": [],
            "harvest": [],
            "consume": [],
        }
        ue_data = {
            "pos": self.ue.pos,
            "payload": [],
            "latency": [],
            "max_latency": [],
            "cover": [],
        }
        max_latency = np.zeros_like(self.ue.latency)
        for action in action_history:
            self.step(action)
            uav_data["pos"].append(self.uav.pos.tolist())
            uav_data["action"].append(self.uav.action_space[action])
            uav_data["validity"].append(self._uav_act_valid)
            uav_data["rewards"].append(self.get_reward())
            uav_data["energy"].append(self.uav.energy)
            uav_data["harvest"].append(self.uav.energy_harvest)
            uav_data["consume"].append(self.uav.energy_consume)
            ue_data["payload"].append(self.ue.payload.tolist())
            ue_data["latency"].append(self.ue.latency.tolist())
            max_latency = np.maximum(max_latency, self.ue.latency)
            ue_data["max_latency"].append(max_latency.tolist())
            ue_data["cover"].append(
                np.stack(
                    [self.ue.covered, self.ue.cover_slot / self.current_slot], axis=1
                ).tolist()
            )
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        pd.DataFrame(uav_data).to_csv(path / "uav_data.csv", index=False)
        for key in ue_data:
            pd.DataFrame(ue_data[key]).to_csv(
                path / f"ue_{key}.csv", index=False, header=False
            )
