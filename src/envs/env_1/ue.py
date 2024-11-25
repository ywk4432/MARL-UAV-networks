class UE:
    def __init__(
        self, index: int = 0, cluster: int = 0, pos: list = None, cover_uav: int = -1
    ):
        """
        :param index: UE 的编号
        :param cluster: UE 从属的 cluster
        :param pos: UE 当前的位置
        :param cover_uav: 当前覆盖这个 UE 的无人机
        """
        self.id = index
        self.cluster = cluster
        self.pos = [0, 0] if pos is None else pos
        self.cover_uav = cover_uav
        self.cover_slot = 0  # 开始至当前时隙被覆盖的时隙数目
        self.cover_uav_record = []  # 保存每个时隙覆盖本 UE 的无人机 id

    def reset(self):
        self.cover_uav = -1
        self.cover_slot = 0
        self.cover_uav_record.clear()
