class CellMap:
    class Cell:
        def __init__(
            self,
            uav_id: int = -1,
            ue_num: int = 0,
            obstacle: int = -1,
            uav_num: int = 6,
        ):
            self.uav_id = uav_id
            self.ue_num = ue_num
            self.ue_list = []
            self.obstacle = obstacle
            self.novelty = [1.0] * uav_num
            self.access_cnt = [0] * uav_num

    def __init__(
        self,
        map_length: int,
        map_width: int,
        ue_list: list,
        obs_list: list,
        uav_init_pos: list,
    ):
        self.uav_num = len(uav_init_pos)
        self.data = [
            [self.Cell(uav_num=self.uav_num) for _ in range(map_width)]
            for _ in range(map_length)
        ]
        self.uav_init_pos = uav_init_pos
        for ue in ue_list:
            cell = self.data[ue.pos[0]][ue.pos[1]]
            cell.ue_num += 1
            cell.ue_list.append(ue)
        for obs in obs_list:
            x, y = obs.pos
            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    self.data[x + i][y + j].obstacle = obs.shape[2]
                    self.data[x + i][y + j].novelty = [0.0] * self.uav_num
        for uav_id, pos in enumerate(self.uav_init_pos):
            self.data[pos[0]][pos[1]].uav_id = uav_id

    def __getitem__(self, item):
        return self.data[item[0]][item[1]]

    def reset(self):
        for line in self.data:
            for cell in line:
                cell.uav_id = -1
                cell.novelty = [1.0 if cell.obstacle <= 0 else 0.0] * self.uav_num
                cell.access_cnt = [0] * self.uav_num
        for uav_id, pos in enumerate(self.uav_init_pos):
            self.data[pos[0]][pos[1]].uav_id = uav_id
