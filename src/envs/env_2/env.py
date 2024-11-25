import numpy as np

from ..env_1 import Env as Env1


class Env(Env1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_obs(self) -> np.ndarray:
        """
        返回当前的观测
        image_obs: [智能体][x][y][UE, 临近无人机（有无）, Novelty]
        :return: [智能体][拼接之后的向量]
        """
        image_obs = np.delete(self.get_image_obs(), 1, axis=3)
        return image_obs.reshape(image_obs.shape[0], -1)

    def get_obs_size(self) -> int:
        return int(
            self.config["channel_num"] * ((2 * self.uav_list[0].max_obs_radius) ** 2)
        )

    def get_r2(self) -> float:
        items = np.array([ue.cover_slot for ue in self.ue_list], dtype=np.float64)
        items /= self.current_slot
        denominator = len(items) * np.sum(np.square(items))
        jain = 1 if denominator < 1e-6 else np.sum(items) ** 2 / denominator
        self.fairness = jain
        self._record_current_slot["fairness"] = jain
        return jain

    def get_sub_rewards(self) -> np.ndarray:
        """
        返回子奖励
        r1: 全局 UE 覆盖率
        r2: UE 覆盖公平指数，单项为开始到现在的被服务时隙数/当前时隙数
        r3: 惩罚项
        r4: 无人机服务ue数/服务上限
        r5: 无人机观测范围内的地图新颖性
        :return: [UAV][r1, r2, r3, r4, r5]
        """
        r1 = self.get_r1()
        self._record_current_slot["ue_cover_rate"] = r1
        r2 = self.get_r2()
        self._record_current_slot["cluster_cover"] = np.array(
            [float(r1 > self.config["ue"]["cluster_cover_threshold"])]
        )
        res = []
        for uav in self.uav_list:
            r3 = float(not uav.action_valid) * self.config["penalty"]
            r4 = uav.serve_ue_count / uav.max_serve_capacity
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
            r5 = np.mean(novelty)
            res.append([r1, r2, r3, r4, r5])
        return np.array(res, dtype=np.float64)

    def get_reward_total(self, sub_rewards: np.ndarray) -> float:
        r1 = sub_rewards[0, 0]
        r2 = sub_rewards[0, 1]
        r3 = np.sum(sub_rewards[:, 2])
        r4 = np.mean(sub_rewards[:, 3])
        r5 = np.mean(sub_rewards[:, 4])
        self._record_current_slot["sub_reward_total"] = [r1, r2, r3, r4, r5]
        reward_total = np.sum([r1, r2, r3, r4])
        if self.use_novelty:
            reward_total += r5
        return reward_total

    def get_reward(
        self,
        sub_rewards: np.ndarray = None,
        model_type: str = "default",
    ) -> np.ndarray:
        """
        For DMTD
        """
        if sub_rewards is None:
            sub_rewards = self.get_sub_rewards()
        return sub_rewards[:, :4].sum(axis=1)

    def get_state(self) -> np.ndarray:
        if self.use_onehot:
            length_onehot_base = np.eye(self.map_length)
            width_onehot_base = np.eye(self.map_width)
            height_onehot_base = np.eye(self.map_height)
            onehot_bases = [length_onehot_base, width_onehot_base, height_onehot_base]
            uav_pos = [
                np.concatenate([onehot_bases[i][uav.pos[i]] for i in range(3)])
                for uav in self.uav_list
            ]
            ue_pos = [
                np.concatenate([onehot_bases[i][ue.pos[i]] for i in range(2)])
                for ue in self.ue_list
            ]
        else:
            uav_pos = [
                [
                    uav.pos[0] / self.map_length,
                    uav.pos[1] / self.map_width,
                    uav.pos[2] / self.map_height,
                ]
                for uav in self.uav_list
            ]
            ue_pos = [
                [
                    ue.pos[0] / self.map_length,
                    ue.pos[1] / self.map_width,
                ]
                for ue in self.ue_list
            ]
        uav_pos = np.array(uav_pos).reshape(-1)
        ue_pos = np.array(ue_pos).reshape(-1)
        return np.concatenate((uav_pos, ue_pos))

    def get_state_size(self) -> int:
        if self.use_onehot:
            return (self.map_length + self.map_width) * len(self.ue_list) + (
                self.map_length + self.map_width + self.map_height
            ) * len(self.uav_list)
        else:
            return 2 * len(self.ue_list) + 3 * len(self.uav_list)
