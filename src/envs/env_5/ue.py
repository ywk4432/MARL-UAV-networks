import numpy as np


class UE:
    def __init__(self, map_length: int, map_width: int, **config):
        self.num: int = config["num"]
        self.max_payload: float = config["max_payload"]
        self.max_generate_time: int = config[
            "max_generate_time"
        ]  # 在此时刻之后不会再生成 payload
        self.payload_generate_time = np.random.randint(
            0, self.max_generate_time, self.num
        )
        self.payload_generate_volume = np.random.uniform(0, self.max_payload, self.num)
        pos_x = np.random.uniform(0, map_length, self.num)
        pos_y = np.random.uniform(0, map_width, self.num)
        self.pos = np.stack([pos_x, pos_y], axis=1)
        self.cover_slot = np.zeros(self.num, dtype=int)
        self.covered = np.zeros(self.num, dtype=bool)
        self.payload = np.zeros(self.num, dtype=float)  # 数据量或计算负载
        self.latency = np.zeros(self.num, dtype=int)  # 信息年龄或等待时延

    def step(
        self, current_slot: int, trans_volume: np.ndarray, connected: np.ndarray
    ) -> None:
        generated_payload = (
            np.logical_not(self.payload_generate_time - current_slot).astype(int)
            * self.payload_generate_volume
        )
        self.payload = np.maximum(self.payload + generated_payload - trans_volume, 0)
        self.covered = connected
        self.cover_slot += connected
        self.latency[self.payload > 0] += 1
        self.latency[self.payload == 0] = 0

    def reset(self):
        self.cover_slot = np.zeros(self.num, dtype=int)
        self.payload = np.zeros(self.num, dtype=float)  # 数据量或计算负载
        self.latency = np.zeros(self.num, dtype=int)  # 信息年龄或等待时延
